import runpod
import torch
import re
import json
import time
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM


def log(msg):
    print(f"[LOG] {msg}", flush=True)


# ===============================
# CUDA SETUP
# ===============================
log("Starting anonymization worker")
log(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ===============================
# LOAD QWEN 2.5 7B
# ===============================
MODEL_PATH = "/models/qwen"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
log("Qwen 2.5-7B-Instruct loaded")


# ===============================
# REGEX PATTERNS FOR RUSSIAN COMPANIES
# ===============================
COMPANY_REGEX_PATTERNS = [
    r'ООО\s*[«\""]([^»\""]+)[»\"""]',
    r'АО\s*[«\""]([^»\""]+)[»\"""]',
    r'ПАО\s*[«\""]([^»\""]+)[»\"""]',
    r'ЗАО\s*[«\""]([^»\""]+)[»\"""]',
    r'ИП\s+([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.)',
    r'ИП\s+([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)',
    r'Акционерное\s+общество\s*[«\""]([^»\""]+)[»\"""]',
    r'Общество\s+с\s+ограниченной\s+ответственностью\s*[«\""]([^»\""]+)[»\"""]',
    r'Публичное\s+акционерное\s+общество\s*[«\""]([^»\""]+)[»\"""]',
]

# Full-match patterns (match the entire entity string including prefix)
COMPANY_FULL_PATTERNS = [
    r'ООО\s*[«\""][^»\""]+[»\"""]',
    r'АО\s*[«\""][^»\""]+[»\"""]',
    r'ПАО\s*[«\""][^»\""]+[»\"""]',
    r'ЗАО\s*[«\""][^»\""]+[»\"""]',
    r'ИП\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.',
    r'ИП\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+',
    r'Акционерное\s+общество\s*[«\""][^»\""]+[»\"""]',
    r'Общество\s+с\s+ограниченной\s+ответственностью\s*[«\""][^»\""]+[»\"""]',
    r'Публичное\s+акционерное\s+общество\s*[«\""][^»\""]+[»\"""]',
]


# ===============================
# CORE FUNCTIONS
# ===============================

def combine_pages(pages):
    """Combine all pages into a single text string, preserving page order."""
    sorted_pages = sorted(pages, key=lambda p: p["page"])
    return "\n\n".join(p["text"] for p in sorted_pages)


def chunk_text_with_overlap(text, max_tokens=3000, overlap_tokens=300):
    """Split text into overlapping chunks based on tokenizer token count."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += max_tokens - overlap_tokens
    log(f"Split text into {len(chunks)} chunks ({len(tokens)} total tokens)")
    return chunks


SYSTEM_PROMPT = """You are a named entity recognition assistant for Russian legal contracts.

Your task: Extract ALL person names and organization names from the provided text.

Rules:
- Extract FULL names of persons (e.g., "Иванов Иван Иванович")
- Extract FULL organization names including their legal form (e.g., "ООО «Ромашка»")
- Include short and full forms of organizations if both appear
- Do NOT include positions, titles, or roles
- Do NOT include addresses or dates
- Output ONLY valid JSON, no explanations

Output format:
{
  "persons": ["Person Name 1", "Person Name 2"],
  "organizations": ["Organization 1", "Organization 2"]
}

If no entities found, return:
{
  "persons": [],
  "organizations": []
}"""


def extract_entities_llm(text_chunk):
    """Use LLM to extract person and organization entities from a text chunk."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract all person and organization names from this Russian legal contract text:\n\n{text_chunk}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1
        )

    # Free input tensors immediately
    del inputs
    torch.cuda.empty_cache()

    raw_output = tokenizer.decode(
        output[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    log(f"LLM raw output: {raw_output[:500]}")

    # Parse JSON from output
    try:
        # Try to find JSON in the output
        json_match = re.search(r'\{[^{}]*"persons"\s*:\s*\[.*?\]\s*,\s*"organizations"\s*:\s*\[.*?\]\s*\}', raw_output, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw_output)

        persons = result.get("persons", [])
        organizations = result.get("organizations", [])

        # Clean up: remove empty strings and strip whitespace
        persons = [p.strip() for p in persons if p and p.strip()]
        organizations = [o.strip() for o in organizations if o and o.strip()]

        return persons, organizations

    except (json.JSONDecodeError, AttributeError) as e:
        log(f"Failed to parse LLM output: {e}")
        return [], []


def detect_companies_regex(text):
    """Detect Russian company names using regex patterns."""
    organizations = []
    for pattern in COMPANY_FULL_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            clean = match.strip()
            if clean and clean not in organizations:
                organizations.append(clean)

    log(f"Regex detected {len(organizations)} organizations")
    return organizations


def merge_entities(llm_persons, llm_orgs, regex_orgs):
    """Merge LLM and regex results, deduplicate."""
    # Deduplicate persons
    seen_persons = set()
    persons = []
    for p in llm_persons:
        normalized = p.strip()
        if normalized and normalized.lower() not in seen_persons:
            seen_persons.add(normalized.lower())
            persons.append(normalized)

    # Merge and deduplicate organizations
    all_orgs = llm_orgs + regex_orgs
    seen_orgs = set()
    organizations = []
    for o in all_orgs:
        normalized = o.strip()
        if normalized and normalized.lower() not in seen_orgs:
            seen_orgs.add(normalized.lower())
            organizations.append(normalized)

    log(f"Merged entities: {len(persons)} persons, {len(organizations)} organizations")
    return persons, organizations


def build_ordered_mapping(full_text, persons, organizations):
    """Build deterministic mapping based on first appearance order in document."""
    # Find first occurrence position of each entity
    person_positions = []
    for p in persons:
        pos = full_text.find(p)
        if pos == -1:
            # Try case-insensitive search
            pos = full_text.lower().find(p.lower())
        if pos == -1:
            pos = float('inf')
        person_positions.append((p, pos))

    org_positions = []
    for o in organizations:
        pos = full_text.find(o)
        if pos == -1:
            pos = full_text.lower().find(o.lower())
        if pos == -1:
            pos = float('inf')
        org_positions.append((o, pos))

    # Sort by first appearance
    person_positions.sort(key=lambda x: x[1])
    org_positions.sort(key=lambda x: x[1])

    mapping = {}

    # Assign company1, company2, etc.
    for idx, (org, _) in enumerate(org_positions, start=1):
        mapping[org] = f"company{idx}"

    # Assign user1, user2, etc.
    for idx, (person, _) in enumerate(person_positions, start=1):
        mapping[person] = f"user{idx}"

    log(f"Built mapping with {len(mapping)} entries")
    for entity, placeholder in mapping.items():
        log(f"  {entity} -> {placeholder}")

    return mapping


def safe_replace(text, mapping):
    """Replace entities in text safely, longest match first to avoid partial replacements."""
    # Sort by length descending to replace longest matches first
    sorted_entities = sorted(mapping.keys(), key=len, reverse=True)

    for entity in sorted_entities:
        placeholder = mapping[entity]
        # Use word-boundary-aware replacement
        # For Russian text, use a pattern that handles Cyrillic word boundaries
        escaped = re.escape(entity)
        text = re.sub(escaped, placeholder, text)

    return text


def anonymize_document(pages):
    """Main anonymization pipeline."""
    total_start = time.time()
    log(f"Processing document with {len(pages)} pages")

    # Safety limit
    if len(pages) > 100:
        return {"error": f"Document too large: {len(pages)} pages (max 100)"}

    # Step 1: Combine all pages into global text
    full_text = combine_pages(pages)
    log(f"Combined text length: {len(full_text)} chars")

    # Step 2: Chunk the text for LLM processing
    chunks = chunk_text_with_overlap(full_text)

    # Step 3: Extract entities from each chunk via LLM
    all_llm_persons = []
    all_llm_orgs = []
    for i, chunk in enumerate(chunks):
        chunk_start = time.time()
        log(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        persons, orgs = extract_entities_llm(chunk)
        all_llm_persons.extend(persons)
        all_llm_orgs.extend(orgs)
        chunk_time = time.time() - chunk_start
        log(f"Chunk {i+1} done in {chunk_time:.1f}s — found {len(persons)} persons, {len(orgs)} orgs")

        # Clear CUDA cache between chunks to prevent VRAM fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - total_start
    log(f"LLM extraction complete in {elapsed:.1f}s — {len(all_llm_persons)} persons, {len(all_llm_orgs)} orgs (before dedup)")

    # Step 4: Regex detection on full text
    regex_orgs = detect_companies_regex(full_text)

    # Step 5: Merge and deduplicate
    persons, organizations = merge_entities(all_llm_persons, all_llm_orgs, regex_orgs)

    # Step 6: Build ordered mapping based on first appearance
    mapping = build_ordered_mapping(full_text, persons, organizations)

    if not mapping:
        log("No entities found, returning original document")
        return {
            "pages": [{"page": p["page"], "text": p["text"]} for p in sorted(pages, key=lambda x: x["page"])],
            "mapping": {}
        }

    # Step 7: Anonymize each page individually using the global mapping
    anonymized_pages = []
    sorted_pages = sorted(pages, key=lambda p: p["page"])
    for page in sorted_pages:
        anonymized_text = safe_replace(page["text"], mapping)
        anonymized_pages.append({
            "page": page["page"],
            "text": anonymized_text
        })

    total_time = time.time() - total_start
    log(f"Anonymization complete in {total_time:.1f}s — {len(mapping)} entities replaced across {len(pages)} pages")
    return {
        "pages": anonymized_pages,
        "mapping": mapping
    }


# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    """RunPod serverless handler for contract anonymization."""
    try:
        pages = event["input"]["pages"]

        if not pages or not isinstance(pages, list):
            return {"error": "Invalid input: 'pages' must be a non-empty list"}

        # Validate page structure
        for p in pages:
            if "page" not in p or "text" not in p:
                return {"error": "Each page must have 'page' (number) and 'text' (string) fields"}

        result = anonymize_document(pages)
        return result

    except KeyError as e:
        return {"error": f"Missing required field: {e}"}
    except Exception as e:
        log(f"Error in handler: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
