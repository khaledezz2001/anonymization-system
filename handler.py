import runpod
import re
import json
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ===============================
# LOAD MODEL WITH vLLM
# ===============================
MODEL_PATH = "/app/models/Qwen3-14B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

llm = LLM(
    model=MODEL_PATH,
    dtype="float16",
    max_model_len=16384,        # Increased: long docs need more context room
    tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
    gpu_memory_utilization=0.90,
)

SAMPLING_PARAMS = SamplingParams(
    temperature=0,          # greedy decoding
    max_tokens=4096,        # Increased: dense docs produce many entities
    repetition_penalty=1.1,
)

print(f"[LOG] Qwen/Qwen3-14B loaded via vLLM", flush=True)


# ===============================
# CORE FUNCTIONS
# ===============================
def combine_pages(pages):
    sorted_pages = sorted(pages, key=lambda p: p["page"])
    return "\n\n".join(p["text"] for p in sorted_pages)


def chunk_text_with_overlap(text, max_tokens=3000, overlap_tokens=200):
    """Split text into token-limited chunks with sentence-boundary awareness.

    max_tokens is kept conservative (3000) so that after adding the system
    prompt (~1500 tokens) and reserving max_tokens for output (4096), the
    total stays safely within max_model_len (16384).

    Tries to break at sentence boundaries (. ! ?) to give the LLM better
    context per chunk, improving entity detection accuracy.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        actual_end = end

        # Try to break at sentence boundary if not at the end of the text
        if end < len(tokens) and len(chunk_text) > 200:
            sentence_enders = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
            best_break = -1
            for ender in sentence_enders:
                idx = chunk_text.rfind(ender)
                if idx > best_break:
                    best_break = idx

            # Only break at sentence if boundary is in the last 40% of chunk
            # (don't make chunks too small)
            min_break_pos = len(chunk_text) * 6 // 10
            if best_break >= min_break_pos:
                trimmed_text = chunk_text[:best_break + 1].strip()
                if trimmed_text:
                    trimmed_tokens = tokenizer.encode(trimmed_text, add_special_tokens=False)
                    chunk_text = trimmed_text
                    actual_end = start + len(trimmed_tokens)

        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(chunk_text)
        if actual_end >= len(tokens):
            break
        start = max(actual_end - overlap_tokens, start + 1)

    return chunks


MAX_CHUNKS = 120   # 40 pages ≈ 40-50 chunks at 3000 tok/chunk — give headroom


SYSTEM_PROMPT = """You are a multilingual named entity recognition (NER) assistant for legal and business documents.

Extract ALL of the following from the text:
1. Person names (actual human names only)
2. Organisation / company names (actual registered business names only)
3. Dates (specific calendar dates only)
4. Addresses (physical street/postal addresses in any language)
5. Phone numbers (phone and fax numbers)
6. Registration IDs (company registration numbers, tax IDs)
7. Bank accounts (IBAN numbers, bank account numbers, SWIFT/BIC codes)
8. Email addresses (email addresses of individuals or organizations)

CRITICAL RULES - what to extract:
- PERSONS: Only real human names, like "John Smith", "Andreas Menelaou"
  - Extract person names EVEN when they appear in an official capacity
  - Extract person names from witnesses, signatories, advocates
  - If a person's name is used as a business/firm name, extract it as BOTH a person AND an organisation
- ORGANISATIONS: Only actual named companies/firms that are REGISTERED BUSINESS ENTITIES
  - Extract ALL language variants of the same company
  - Include companies in any language: English, Greek, Russian (e.g., ООО, АО), French, German, etc.
  - A company must be a specific legal entity (e.g., "Altus Citadel Corporate Services Limited")
- DATES: Only specific calendar dates, like "01/09/2015", "24th of July, 2015"
  - Do NOT extract section or article numbers as dates (e.g. "2.2.11", "3.1.5" are section numbers, NOT dates)
- ADDRESSES: ONLY physical street/postal addresses with specific location details
  - An address MUST contain a street name, building number, postcode, or similar location identifier
  - Example addresses: "82 Akropoleos, 2nd floor, 1012 Acropolis, Cyprus", "191 ATHALASSIS AVE."
  - Extract addresses from EVERYWHERE: signature pages, witness sections, headers, body text
  - Addresses can be in any format and any language
- PHONES: Phone and fax numbers, like "+357 22 315161", "22314641"
- REGISTRATION IDS: Any company or entity identification numbers
  - Examples: "H.E.107777", "HE317807", "HRB 12345", "Company No. 12345678", "Reg. No. 123456", "Tax ID 123456"
- BANK ACCOUNTS: Bank account numbers, IBAN codes, SWIFT/BIC codes
  - Examples: "CY17 0020 0128 0000 0012 0052 7600", "BCYPCY2N", "Account No. 0120052760"
  - Include ANY numbers explicitly labeled as bank accounts, deposit accounts, or payment accounts
- EMAILS: Email addresses of individuals or organizations
  - Examples: "john@example.com", "info@company.com", "maria.smith@org.co.uk"
  - Extract email addresses found anywhere in the document

CRITICAL RULES - what NOT to extract:
- Do NOT extract role titles as persons: Chairman, Director, Secretary, Landlord, Tenant
- Do NOT extract generic legal terms as organisations: "the Company", "Board of Directors"
- Do NOT extract GOVERNMENT BODIES, INTERGOVERNMENTAL ORGANIZATIONS, or REGULATORY AUTHORITIES as organisations.
  These are NOT companies: "European Commission", "European Banking Authority", "FATF", "Financial Action Task Force", "United Nations", "MOKAS", "Unit for Combating Money Laundering"
- Do NOT extract EU DIRECTIVES or REGULATIONS as organisations.
  These are NOT companies: "Directive (EU) 2018/843", "Directive 2018/1673", "Council Regulation (EU) No. 833/2014", "General Data Protection Regulation (GDPR)", "EBA Guidelines"
- Do NOT extract COUNTRIES, JURISDICTIONS, or GEOGRAPHIC AREAS as organisations: "BVI", "European Economic Area"
- Do NOT extract INDICES or REPORTS as organisations: "Basel AML Index"
- Do NOT extract countries alone as organisations
- Do NOT extract time durations as dates: "fourteen days", "six months"
- Do NOT extract bare years as dates: "2014" alone is NOT a date
- Do NOT extract quarter references as dates: "Q2 2024" alone is a period, not a specific date
- Do NOT extract section/article numbers as dates: "2.2.11", "3.1.5" are NOT dates
- Do NOT extract sentence fragments as addresses
- Do NOT extract page numbers or section headers as addresses (e.g. "4 INTRODUCTION" is NOT an address)
- Do NOT extract duration phrases as addresses (e.g. "1 year for high risk customers" is NOT an address)
- Do NOT extract counts or quantities as addresses (e.g. "2 clients onboarded" is NOT an address)
- Do NOT extract legal references as addresses (e.g. "8 and Chapter VI of Directive" is NOT an address)
- Do NOT extract bank account numbers, IBAN codes, or reference numbers as phone numbers
- Do NOT extract URLs or website domain names as email addresses (e.g. "www.example.com" is NOT an email)

Output ONLY valid JSON with no explanation or thinking. Do not wrap in markdown.

{
  "persons": ["name1", "name2"],
  "organizations": ["org1", "org2"],
  "dates": ["date1", "date2"],
  "addresses": ["addr1", "addr2"],
  "phones": ["phone1", "phone2"],
  "registration_ids": ["H.E.107777"],
  "bank_accounts": ["CY17 0020 0128 0000 0012 0052 7600"],
  "emails": ["email1@example.com"]
}"""


def strip_thinking(text):
    """Remove <think>...</think> blocks from model output (safety net)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def extract_entities_batch(chunks):
    """Extract entities from all chunks using vLLM batch inference.

    Processes chunks in micro-batches of BATCH_SIZE to avoid GPU KV-cache
    OOM when there are many chunks from large documents.
    """
    BATCH_SIZE = 8  # Process 8 chunks at a time to avoid KV-cache OOM

    all_persons, all_orgs, all_dates = [], [], []
    all_addresses, all_phones, all_reg_ids, all_bank_accounts, all_emails = [], [], [], [], []

    total_chunks = len(chunks)
    for batch_start in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_chunks)
        batch_chunks = chunks[batch_start:batch_end]

        print(f"[LOG] Processing chunks {batch_start + 1}-{batch_end} of {total_chunks}", flush=True)

        prompts = []
        for chunk in batch_chunks:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract all persons, organizations, dates, addresses, phones, registration IDs, bank accounts, and emails:\n\n{chunk}"}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompts.append(prompt)

        outputs = llm.generate(prompts, SAMPLING_PARAMS)

        for output in outputs:
            raw = output.outputs[0].text.strip()
            cleaned = strip_thinking(raw)

            try:
                json_match = re.search(r'\{[^{}]*"persons"\s*:.*\}', cleaned, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                result = json.loads(json_match.group()) if json_match else json.loads(cleaned)

                all_persons.extend([p.strip() for p in result.get("persons", []) if p and p.strip()])
                all_orgs.extend([o.strip() for o in result.get("organizations", []) if o and o.strip()])
                all_dates.extend([d.strip() for d in result.get("dates", []) if d and d.strip()])
                all_addresses.extend([a.strip() for a in result.get("addresses", []) if a and a.strip()])
                all_phones.extend([p.strip() for p in result.get("phones", []) if p and p.strip()])
                all_reg_ids.extend([r.strip() for r in result.get("registration_ids", []) if r and r.strip()])
                all_bank_accounts.extend([b.strip() for b in result.get("bank_accounts", []) if b and b.strip()])
                all_emails.extend([e.strip() for e in result.get("emails", []) if e and e.strip()])
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"[WARN] Failed to parse chunk output: {e}", flush=True)
                print(f"[WARN] Raw output was: {raw[:500]}", flush=True)

    print(f"[LOG] Entity extraction complete. Found: {len(all_persons)} persons, "
          f"{len(all_orgs)} orgs, {len(all_dates)} dates, {len(all_addresses)} addresses, "
          f"{len(all_phones)} phones, {len(all_reg_ids)} reg_ids, {len(all_bank_accounts)} bank_accounts, "
          f"{len(all_emails)} emails",
          flush=True)

    return all_persons, all_orgs, all_dates, all_addresses, all_phones, all_reg_ids, all_bank_accounts, all_emails


# ===============================
# ANTI-HALLUCINATION VALIDATION
# ===============================
def verify_entity_in_text(entity, full_text_lower):
    """Check if an entity actually exists in the source document.

    Uses case-insensitive matching with flexible whitespace.
    Returns True if the entity (or a whitespace-flexible variant) is found.
    """
    entity_clean = entity.strip()
    if not entity_clean:
        return False

    # Fast path: direct case-insensitive substring match
    if entity_clean.lower() in full_text_lower:
        return True

    # Flexible whitespace: the entity might span a line break in the source
    # e.g., model says "John Smith" but text has "John\nSmith"
    words = entity_clean.split()
    if len(words) > 1:
        flexible = r'\s+'.join(re.escape(w) for w in words)
        if re.search(flexible, full_text_lower, re.IGNORECASE):
            return True

    return False


def validate_entities(entities, full_text_lower, entity_type):
    """Filter out hallucinated entities that don't appear in the source text."""
    valid = []
    removed = []
    for entity in entities:
        if verify_entity_in_text(entity, full_text_lower):
            valid.append(entity)
        else:
            removed.append(entity)

    if removed:
        print(f"[HALLUCINATION] Removed {len(removed)} fake {entity_type}: {removed[:10]}", flush=True)

    return valid


# ===============================
# REGEX BACKUP DETECTION
# ===============================
def regex_backup_detection(full_text):
    """Catch common PII patterns the LLM might have missed using regex.

    Runs as a safety net after LLM extraction to ensure high-confidence
    patterns like emails, phone numbers, and IBANs are never missed.
    """
    backup_emails = []
    backup_phones = []
    backup_ibans = []

    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    for match in re.finditer(email_pattern, full_text):
        backup_emails.append(match.group())

    # International phone pattern (requires 7-15 digits)
    phone_pattern = r'(?<!\d)(?:\+\d{1,3}[\s\-]?)?\(?\d{1,4}\)?[\s\-]?\d{2,4}[\s\-]?\d{3,4}(?!\d)'
    for match in re.finditer(phone_pattern, full_text):
        candidate = match.group().strip()
        digits = re.sub(r'\D', '', candidate)
        if 7 <= len(digits) <= 15:
            backup_phones.append(candidate)

    # IBAN pattern
    iban_pattern = r'\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,4}\b'
    for match in re.finditer(iban_pattern, full_text):
        backup_ibans.append(match.group())

    return backup_emails, backup_phones, backup_ibans


# ===============================
# DEDUPLICATION
# ===============================
def dedup_list(items):
    """Deduplicate a list while preserving order (case-insensitive)."""
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def dedup_substrings(items):
    """Remove items that are substrings of other items.

    Optimized: uses a set for O(1) exact-match skips and limits
    comparison window for large lists.
    """
    if not items:
        return items
    sorted_items = sorted(items, key=len, reverse=True)
    result = []
    result_lower = []  # parallel list to avoid repeated .lower() calls
    for item in sorted_items:
        item_lower = item.lower()
        is_sub = False
        for accepted_lower in result_lower:
            if item_lower in accepted_lower:
                is_sub = True
                break
        if not is_sub:
            result.append(item)
            result_lower.append(item_lower)
    return result


# ===============================
# REPLACEMENT FUNCTIONS
# ===============================
def _flexible_pattern(text_str):
    """Build a regex that matches text_str with flexible whitespace,
    but NEVER matches in the middle of a word."""
    escaped = re.escape(text_str)
    flexible = escaped.replace(r'\ ', r'\s+')
    return r'(?<!\w)' + flexible + r'(?!\w)'


def build_combined_pattern(mapping):
    """Build a single compiled regex that matches all entities at once.

    This is dramatically faster than running one regex per entity,
    especially when there are hundreds of entities across many pages.
    Entities are sorted longest-first so longer matches take priority.
    """
    sorted_entities = sorted(mapping.keys(), key=len, reverse=True)
    patterns = []
    for entity in sorted_entities:
        patterns.append(_flexible_pattern(entity))
    combined = '|'.join(f'({p})' for p in patterns)
    return re.compile(combined, re.IGNORECASE), sorted_entities


def safe_replace(text, mapping):
    """Replace all entities in text using a single-pass combined regex.

    For small mapping sets (< 5), falls back to sequential replacement
    since the overhead of building a combined pattern isn't worth it.
    """
    if not mapping:
        return text

    if len(mapping) < 5:
        # Small number of entities — sequential is fine
        sorted_entities = sorted(mapping.keys(), key=len, reverse=True)
        for entity in sorted_entities:
            placeholder = mapping[entity]
            pattern = _flexible_pattern(entity)
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
        return text

    # For many entities, use single-pass replacement
    sorted_entities = sorted(mapping.keys(), key=len, reverse=True)
    patterns = [_flexible_pattern(entity) for entity in sorted_entities]
    combined = '|'.join(f'({p})' for p in patterns)

    try:
        compiled = re.compile(combined, re.IGNORECASE)
    except re.error:
        # Fallback to sequential if regex is too complex
        for entity in sorted_entities:
            placeholder = mapping[entity]
            pattern = _flexible_pattern(entity)
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
        return text

    # Build a lookup: for each match, find which entity it matched
    entity_lower_map = {}
    for entity in sorted_entities:
        entity_lower_map[entity.lower()] = mapping[entity]

    def replace_match(match):
        matched_text = match.group(0)
        # Look up the matched text (case-insensitive)
        matched_lower = matched_text.lower().strip()
        # Try exact match first
        if matched_lower in entity_lower_map:
            return entity_lower_map[matched_lower]
        # Normalize whitespace and try again
        normalized = re.sub(r'\s+', ' ', matched_lower)
        if normalized in entity_lower_map:
            return entity_lower_map[normalized]
        # Fallback: find the entity that best matches
        for entity, placeholder in mapping.items():
            if re.fullmatch(_flexible_pattern(entity), matched_text, re.IGNORECASE):
                return placeholder
        return matched_text  # no match — return unchanged

    text = compiled.sub(replace_match, text)
    return text


# ===============================
# MAIN ANONYMIZATION PIPELINE
# ===============================
def anonymize_document(pages):

    full_text = combine_pages(pages)
    total_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
    print(f"[LOG] Document: {len(pages)} pages, ~{total_tokens} tokens", flush=True)

    chunks = chunk_text_with_overlap(full_text)
    print(f"[LOG] Split into {len(chunks)} chunks", flush=True)

    if len(chunks) > MAX_CHUNKS:
        return {
            "error": (
                f"Document too large: {len(chunks)} chunks required but the maximum is {MAX_CHUNKS}. "
                "Please split the document and process it in parts."
            )
        }

    # Batch inference: process chunks in micro-batches via vLLM
    all_persons, all_orgs, all_dates, all_addresses, all_phones, all_reg_ids, all_bank_accounts, all_emails = \
        extract_entities_batch(chunks)

    # ---- ANTI-HALLUCINATION: verify every entity exists in the source text ----
    full_text_lower = full_text.lower()
    all_persons = validate_entities(all_persons, full_text_lower, "persons")
    all_orgs = validate_entities(all_orgs, full_text_lower, "organizations")
    all_dates = validate_entities(all_dates, full_text_lower, "dates")
    all_addresses = validate_entities(all_addresses, full_text_lower, "addresses")
    all_phones = validate_entities(all_phones, full_text_lower, "phones")
    all_reg_ids = validate_entities(all_reg_ids, full_text_lower, "registration_ids")
    all_bank_accounts = validate_entities(all_bank_accounts, full_text_lower, "bank_accounts")
    all_emails = validate_entities(all_emails, full_text_lower, "emails")

    print(f"[LOG] After validation: {len(all_persons)} persons, {len(all_orgs)} orgs, "
          f"{len(all_dates)} dates, {len(all_addresses)} addresses, "
          f"{len(all_phones)} phones, {len(all_reg_ids)} reg_ids, "
          f"{len(all_bank_accounts)} bank_accounts, {len(all_emails)} emails", flush=True)

    # ---- REGEX BACKUP: catch patterns the LLM might have missed ----
    backup_emails, backup_phones, backup_ibans = regex_backup_detection(full_text)

    existing_emails_lower = {e.lower() for e in all_emails}
    for email in backup_emails:
        if email.lower() not in existing_emails_lower:
            all_emails.append(email)
            existing_emails_lower.add(email.lower())

    existing_phones_lower = {p.lower() for p in all_phones}
    for phone in backup_phones:
        if phone.lower() not in existing_phones_lower:
            all_phones.append(phone)
            existing_phones_lower.add(phone.lower())

    existing_ibans_lower = {b.lower() for b in all_bank_accounts}
    for iban in backup_ibans:
        if iban.lower() not in existing_ibans_lower:
            all_bank_accounts.append(iban)
            existing_ibans_lower.add(iban.lower())

    print(f"[LOG] After regex backup: {len(all_emails)} emails, "
          f"{len(all_phones)} phones, {len(all_bank_accounts)} bank_accounts", flush=True)

    # Deduplicate all entity lists
    unique_persons = dedup_substrings(dedup_list(all_persons))
    unique_orgs = dedup_substrings(dedup_list(all_orgs))
    unique_dates = dedup_substrings(dedup_list(all_dates))
    unique_addresses = dedup_substrings(dedup_list(all_addresses))
    unique_phones = dedup_substrings(dedup_list(all_phones))
    unique_reg_ids = dedup_substrings(dedup_list(all_reg_ids))
    unique_bank_accounts = dedup_substrings(dedup_list(all_bank_accounts))
    unique_emails = dedup_substrings(dedup_list(all_emails))

    print(f"[LOG] After dedup: {len(unique_persons)} persons, {len(unique_orgs)} orgs, "
          f"{len(unique_dates)} dates, {len(unique_addresses)} addresses, "
          f"{len(unique_phones)} phones, {len(unique_reg_ids)} reg_ids, "
          f"{len(unique_bank_accounts)} bank_accounts, {len(unique_emails)} emails", flush=True)

    # Build mapping ordered by first appearance in the document
    def find_first_pos(token):
        pos = full_text.find(token)
        if pos == -1:
            pos = full_text.lower().find(token.lower())
        return pos if pos != -1 else float('inf')

    mapping = {}

    for i, org in enumerate(sorted(unique_orgs, key=find_first_pos), 1):
        mapping[org] = f"[COMPANY{i}]"

    for i, person in enumerate(sorted(unique_persons, key=find_first_pos), 1):
        mapping[person] = f"[PERSON{i}]"

    date_map = {}
    for i, d in enumerate(sorted(unique_dates, key=find_first_pos), 1):
        date_map[d] = f"[DATE{i}]"

    addr_map = {}
    for i, a in enumerate(sorted(unique_addresses, key=find_first_pos), 1):
        addr_map[a] = f"[ADDRESS{i}]"

    phone_map = {}
    for i, p in enumerate(sorted(unique_phones, key=find_first_pos), 1):
        phone_map[p] = f"[PHONE{i}]"

    reg_id_map = {}
    for i, r in enumerate(sorted(unique_reg_ids, key=find_first_pos), 1):
        reg_id_map[r] = f"[REG_ID{i}]"

    bank_account_map = {}
    for i, ba in enumerate(sorted(unique_bank_accounts, key=find_first_pos), 1):
        bank_account_map[ba] = f"[BANK_ACCOUNT{i}]"

    email_map = {}
    for i, e in enumerate(sorted(unique_emails, key=find_first_pos), 1):
        email_map[e] = f"[EMAIL{i}]"

    # Combine all mappings for replacement
    all_mappings = {}
    all_mappings.update(mapping)
    all_mappings.update(addr_map)
    all_mappings.update(date_map)
    all_mappings.update(phone_map)
    all_mappings.update(reg_id_map)
    all_mappings.update(bank_account_map)
    all_mappings.update(email_map)

    print(f"[LOG] Total entities to replace: {len(all_mappings)}", flush=True)

    # Replace all entities page by page
    anonymized_pages = []
    for idx, page in enumerate(sorted(pages, key=lambda p: p["page"])):
        anon_text = safe_replace(page["text"], all_mappings)
        anonymized_pages.append({"page": page["page"], "text": anon_text})
        if (idx + 1) % 10 == 0:
            print(f"[LOG] Replaced entities in {idx + 1}/{len(pages)} pages", flush=True)

    print(f"[LOG] Anonymization complete for {len(pages)} pages", flush=True)

    display_mapping = dict(all_mappings)

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

        print(f"[LOG] Received request with {len(pages)} pages", flush=True)
        return anonymize_document(pages)
    except KeyError as e:
        return {"error": f"Missing field: {e}"}
    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}", flush=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
