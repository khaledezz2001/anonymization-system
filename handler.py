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

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
log("Qwen 2.5-14B-Instruct loaded")


# ===============================
# BLACKLIST: generic legal terms that are NOT entity names
# ===============================
GENERIC_TERMS_LOWER = {
    'общество', 'общества', 'обществу', 'обществом', 'обществе',
    'стороны', 'сторон', 'сторонам', 'сторонами', 'сторонах',
    'стороне', 'сторону', 'стороной', 'сторона',
    'компания', 'компании', 'компанию', 'компанией', 'компаний',
    'компании группы', 'компания группы', 'компаниям группы',
    'компаниями группы', 'компаниях группы',
    'займодавец', 'заемщик', 'займодавца', 'заемщика',
    'займодавцу', 'заемщику', 'займодавцем', 'заемщиком',
    'заёмщик', 'заёмщика', 'заёмщику', 'заёмщиком',
    'регистратор', 'регистратора', 'регистратору', 'регистратором',
    'дочерние общества', 'дочерних обществ', 'дочерним обществам',
    'председатель тк', 'секретарь тк',
    'генеральный директор', 'генерального директора',
    'единственный акционер', 'единственного акционера',
    'уполномоченное лицо', 'уполномоченного лица',
    'компании группа', 'компании группе',
}


# ===============================
# REGEX PATTERNS FOR RUSSIAN COMPANIES
# ===============================
COMPANY_FULL_PATTERNS = [
    r'ООО\s*[«""][^»""]+[»""]',
    r'МКАО\s*[«""][^»""]+[»""]',
    r'АО\s*[«""][^»""]+[»""]',
    r'ПАО\s*[«""][^»""]+[»""]',
    r'ЗАО\s*[«""][^»""]+[»""]',
    r'ИП\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.',
    r'ИП\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+',
    r'Акционерное\s+общество\s*[«""][^»""]+[»""]',
    r'Общество\s+с\s+ограниченной\s+ответственностью\s*[«""][^»""]+[»""]',
    r'Публичное\s+акционерное\s+общество\s*[«""][^»""]+[»""]',
    r'Международн\w+\s+компани\w+\s+акционерн\w+\s+общества?\s*[«""][^»""]+[»""]',
]


# ===============================
# ENTITY VALIDATION
# ===============================
def validate_entity(name):
    """Filter out generic terms, garbage, and invalid entities."""
    name = name.strip()
    if len(name) < 3:
        return False
    if name.lower() in GENERIC_TERMS_LOWER:
        return False
    if ',' in name and '«' in name:
        return False
    if name.count('«') > 1:
        return False
    return True


# ===============================
# ENTITY GROUPING (dedup variants)
# ===============================
def extract_quoted_name(org):
    """Extract name from inside «...» quotes."""
    match = re.search(r'[«""]([^»""]+)[»""]', org)
    return match.group(1).strip().lower() if match else None


def group_org_variants(organizations):
    """Group orgs sharing the same inner «name» into one entity. Keep longest as canonical."""
    groups = []  # [(canonical, [all_variants])]

    for org in organizations:
        inner = extract_quoted_name(org)
        matched_idx = None

        if inner:
            for i, (canonical, variants) in enumerate(groups):
                existing_inner = extract_quoted_name(canonical)
                if existing_inner and inner == existing_inner:
                    matched_idx = i
                    break

        if matched_idx is not None:
            groups[matched_idx][1].append(org)
            if len(org) > len(groups[matched_idx][0]):
                groups[matched_idx] = (org, groups[matched_idx][1])
        else:
            groups.append((org, [org]))

    return groups


def is_initial_match(name_a, name_b):
    """Check if one name is abbreviated form of the other (e.g. 'Сурова Е.Б.' vs 'Сурова Елена Борисовна')."""
    parts_a = [p.rstrip('.') for p in name_a.strip().split() if p.strip()]
    parts_b = [p.rstrip('.') for p in name_b.strip().split() if p.strip()]

    has_init_a = any(len(p) <= 2 for p in parts_a[1:]) if len(parts_a) > 1 else False
    has_init_b = any(len(p) <= 2 for p in parts_b[1:]) if len(parts_b) > 1 else False

    if has_init_a == has_init_b:
        return False
    if has_init_a:
        short_parts, long_parts = parts_a, parts_b
    else:
        short_parts, long_parts = parts_b, parts_a

    if len(short_parts) < 2 or len(long_parts) < 2:
        return False

    min_cmp = min(len(short_parts[0]), len(long_parts[0]), 4)
    if short_parts[0][:min_cmp].lower() != long_parts[0][:min_cmp].lower():
        return False

    for j in range(1, min(len(short_parts), len(long_parts))):
        init = short_parts[j] if len(short_parts[j]) <= 2 else long_parts[j]
        full = long_parts[j] if len(short_parts[j]) <= 2 else short_parts[j]
        if len(init) >= 1 and not full.lower().startswith(init[0].lower()):
            return False

    return True


def group_person_variants(persons):
    """Group abbreviated and full forms of same person. Keep longest as canonical."""
    groups = []

    for person in persons:
        matched_idx = None
        for i, (canonical, variants) in enumerate(groups):
            if is_initial_match(person, canonical):
                matched_idx = i
                break
        if matched_idx is not None:
            groups[matched_idx][1].append(person)
            if len(person) > len(groups[matched_idx][0]):
                groups[matched_idx] = (person, groups[matched_idx][1])
        else:
            groups.append((person, [person]))

    return groups


# ===============================
# RUSSIAN NAME DECLENSION (stem-based regex)
# ===============================
def build_name_pattern(name):
    """Build regex matching a Russian name in any grammatical case via stem matching."""
    parts = name.strip().split()
    if len(parts) < 2:
        return None

    has_initials = bool(re.search(r'\b[А-ЯЁ]\.', name))

    if has_initials:
        surname_match = re.match(r'([А-ЯЁа-яё]+)', name)
        initials = re.findall(r'([А-ЯЁ])\.', name)
        if not surname_match or not initials:
            return None
        surname = surname_match.group(1)
        stem_len = max(3, len(surname) - 2)
        pat = re.escape(surname[:stem_len]) + '[а-яёА-ЯЁ]{0,4}'
        for init in initials:
            pat += r'\s+' + re.escape(init) + r'\.?'
        return pat
    else:
        regex_parts = []
        for part in parts:
            if len(part) < 2:
                continue
            stem_len = max(3, len(part) - 2)
            regex_parts.append(re.escape(part[:stem_len]) + '[а-яёА-ЯЁ]{0,4}')
        return r'\s+'.join(regex_parts) if regex_parts else None


def build_person_patterns(person_groups, mapping):
    """Build list of (compiled_regex, placeholder) for declined name matching."""
    patterns = []
    for canonical, variants in person_groups:
        placeholder = mapping.get(canonical)
        if not placeholder:
            for v in variants:
                if v in mapping:
                    placeholder = mapping[v]
                    break
        if not placeholder:
            continue

        seen_patterns = set()
        for variant in variants:
            pat = build_name_pattern(variant)
            if pat and pat not in seen_patterns:
                seen_patterns.add(pat)
                full_pat = r'(?<![А-ЯЁа-яё])' + pat + r'(?![А-ЯЁа-яё])'
                try:
                    patterns.append((re.compile(full_pat), placeholder))
                except re.error:
                    pass
    return patterns


# ===============================
# CORE FUNCTIONS
# ===============================
def combine_pages(pages):
    sorted_pages = sorted(pages, key=lambda p: p["page"])
    return "\n\n".join(p["text"] for p in sorted_pages)


def chunk_text_with_overlap(text, max_tokens=3000, overlap_tokens=300):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += max_tokens - overlap_tokens
    log(f"Split into {len(chunks)} chunks ({len(tokens)} tokens)")
    return chunks


SYSTEM_PROMPT = """You are a named entity recognition assistant for Russian legal contracts.

Extract ALL person names and organization names from the text.

Rules:
- Extract FULL names of persons (e.g., "Иванов Иван Иванович")
- Extract abbreviated person names too (e.g., "Шатрова Ю.И.")
- Extract FULL organization names including legal form (e.g., "ООО «Ромашка»")
- Do NOT extract generic terms like "Общество", "Стороны", "Компании группы"
- Do NOT extract positions, titles, addresses, or dates
- Output ONLY valid JSON

Output format:
{
  "persons": ["Person Name 1"],
  "organizations": ["Org Name 1"]
}"""


def extract_entities_llm(text_chunk):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract all person and organization names:\n\n{text_chunk}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=2048,
            do_sample=False, temperature=1.0, repetition_penalty=1.1
        )

    del inputs
    torch.cuda.empty_cache()

    raw_output = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    del output
    log(f"LLM raw: {raw_output[:300]}")

    try:
        json_match = re.search(
            r'\{[^{}]*"persons"\s*:\s*\[.*?\]\s*,\s*"organizations"\s*:\s*\[.*?\]\s*\}',
            raw_output, re.DOTALL
        )
        result = json.loads(json_match.group()) if json_match else json.loads(raw_output)
        persons = [p.strip() for p in result.get("persons", []) if p and p.strip()]
        organizations = [o.strip() for o in result.get("organizations", []) if o and o.strip()]
        return persons, organizations
    except (json.JSONDecodeError, AttributeError) as e:
        log(f"Parse failed: {e}")
        return [], []


def detect_companies_regex(text):
    organizations = []
    for pattern in COMPANY_FULL_PATTERNS:
        for match in re.finditer(pattern, text):
            clean = match.group().strip()
            if clean and clean not in organizations:
                organizations.append(clean)
    log(f"Regex found {len(organizations)} orgs")
    return organizations


def merge_entities(llm_persons, llm_orgs, regex_orgs):
    """Merge, validate, deduplicate."""
    seen_p = set()
    persons = []
    for p in llm_persons:
        n = p.strip()
        if n and n.lower() not in seen_p and validate_entity(n):
            seen_p.add(n.lower())
            persons.append(n)

    seen_o = set()
    organizations = []
    for o in llm_orgs + regex_orgs:
        n = o.strip()
        if n and n.lower() not in seen_o and validate_entity(n):
            seen_o.add(n.lower())
            organizations.append(n)

    log(f"After validation: {len(persons)} persons, {len(organizations)} orgs")
    return persons, organizations


def build_ordered_mapping(full_text, person_groups, org_groups):
    """Build deterministic mapping from grouped entities, ordered by first appearance."""
    def find_earliest(variants):
        best = float('inf')
        for v in variants:
            pos = full_text.find(v)
            if pos == -1:
                pos = full_text.lower().find(v.lower())
            if pos != -1 and pos < best:
                best = pos
        return best

    org_positions = [(c, vs, find_earliest(vs)) for c, vs in org_groups]
    person_positions = [(c, vs, find_earliest(vs)) for c, vs in person_groups]

    org_positions.sort(key=lambda x: x[2])
    person_positions.sort(key=lambda x: x[2])

    mapping = {}
    for idx, (canonical, variants, _) in enumerate(org_positions, 1):
        placeholder = f"company{idx}"
        for v in variants:
            mapping[v] = placeholder

    for idx, (canonical, variants, _) in enumerate(person_positions, 1):
        placeholder = f"user{idx}"
        for v in variants:
            mapping[v] = placeholder

    log(f"Mapping ({len(mapping)} entries):")
    shown = set()
    for entity, ph in mapping.items():
        if ph not in shown:
            shown.add(ph)
            log(f"  {ph} <- {entity}")

    return mapping


def safe_replace(text, mapping, name_patterns=None):
    """Replace with Cyrillic word boundaries + declined name patterns."""
    sorted_entities = sorted(mapping.keys(), key=len, reverse=True)

    for entity in sorted_entities:
        placeholder = mapping[entity]
        escaped = re.escape(entity)
        pattern = r'(?<![А-ЯЁа-яёA-Za-z])' + escaped + r'(?![А-ЯЁа-яёA-Za-z])'
        text = re.sub(pattern, placeholder, text)

    if name_patterns:
        for compiled_re, placeholder in name_patterns:
            text = compiled_re.sub(placeholder, text)

    return text


def anonymize_document(pages):
    total_start = time.time()
    log(f"Processing {len(pages)} pages")

    if len(pages) > 100:
        return {"error": f"Too many pages: {len(pages)} (max 100)"}

    full_text = combine_pages(pages)
    log(f"Combined: {len(full_text)} chars")

    chunks = chunk_text_with_overlap(full_text)

    all_persons, all_orgs = [], []
    for i, chunk in enumerate(chunks):
        t = time.time()
        log(f"Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        persons, orgs = extract_entities_llm(chunk)
        all_persons.extend(persons)
        all_orgs.extend(orgs)
        log(f"Chunk {i+1} done in {time.time()-t:.1f}s — {len(persons)}p {len(orgs)}o")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    log(f"LLM total: {len(all_persons)}p {len(all_orgs)}o (raw)")

    regex_orgs = detect_companies_regex(full_text)
    persons, organizations = merge_entities(all_persons, all_orgs, regex_orgs)

    # Group variants
    person_groups = group_person_variants(persons)
    org_groups = group_org_variants(organizations)
    log(f"Grouped: {len(person_groups)} person groups, {len(org_groups)} org groups")

    mapping = build_ordered_mapping(full_text, person_groups, org_groups)

    if not mapping:
        log("No entities found")
        return {
            "pages": [{"page": p["page"], "text": p["text"]} for p in sorted(pages, key=lambda x: x["page"])],
            "mapping": {}
        }

    # Build declension patterns for persons
    name_patterns = build_person_patterns(person_groups, mapping)
    log(f"Built {len(name_patterns)} declension patterns")

    anonymized_pages = []
    for page in sorted(pages, key=lambda p: p["page"]):
        anon_text = safe_replace(page["text"], mapping, name_patterns)
        anonymized_pages.append({"page": page["page"], "text": anon_text})

    # Build clean display mapping (canonical -> placeholder)
    display_mapping = {}
    for canonical, variants in org_groups:
        ph = mapping.get(canonical) or mapping.get(variants[0])
        if ph:
            display_mapping[canonical] = ph
    for canonical, variants in person_groups:
        ph = mapping.get(canonical) or mapping.get(variants[0])
        if ph:
            display_mapping[canonical] = ph

    total = time.time() - total_start
    log(f"Done in {total:.1f}s — {len(display_mapping)} entities across {len(pages)} pages")
    return {"pages": anonymized_pages, "mapping": display_mapping}


# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    try:
        pages = event["input"]["pages"]
        if not pages or not isinstance(pages, list):
            return {"error": "'pages' must be a non-empty list"}
        for p in pages:
            if "page" not in p or "text" not in p:
                return {"error": "Each page needs 'page' and 'text' fields"}
        return anonymize_document(pages)
    except KeyError as e:
        return {"error": f"Missing field: {e}"}
    except Exception as e:
        log(f"Error: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
