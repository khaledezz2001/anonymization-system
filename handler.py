import runpod
import torch
import re
import json
import time
import gc
import os

from huggingface_hub import snapshot_download
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
# DOWNLOAD & LOAD MODEL
# ===============================
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_PATH = f"/runpod-volume/models/{MODEL_ID.replace('/', '_')}"

# Auto-download to network volume on first start
log(f"Ensuring model {MODEL_ID} is downloaded to {MODEL_PATH}...")
os.makedirs(MODEL_PATH, exist_ok=True)
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_PATH,
    local_dir_use_symlinks=False,
    resume_download=True
)
log("Model is present on volume.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
log(f"{MODEL_ID} loaded")


# ===============================
# BLACKLIST: generic legal terms that are NOT entity names
# ===============================
GENERIC_TERMS_LOWER = {
    'company', 'companies', 'the company', 'the companies',
    'party', 'parties', 'the party', 'the parties',
    'borrower', 'lender', 'lessor', 'lessee', 'tenant', 'landlord',
    'client', 'contractor', 'counterparty', 'agent', 'principal',
    'employer', 'employee', 'vendor', 'buyer', 'seller', 'purchaser',
    'assignor', 'assignee', 'guarantor', 'beneficiary',
    'director', 'directors', 'general director', 'ceo', 'shareholder', 'shareholders',
    'authorized person', 'representative', 'signatory',
    'group company', 'group companies', 'subsidiary', 'subsidiaries',
    'chairman', 'secretary', 'treasurer', 'president', 'vice president',
    'manager', 'administrator', 'auditor', 'inspector', 'officer',
    'member', 'members', 'board', 'board of directors', 'committee',
    'trustee', 'trustees', 'receiver', 'liquidator', 'executor',
    'personal representative', 'personal representatives',
    'curator bonis', 'requisitionists', 'proxy',
    'parent', 'child', 'children', 'spouse', 'husband', 'wife',
    'brother', 'sister', 'mother', 'father', 'son', 'daughter',
    'widow', 'widower', 'deceased', 'heir', 'heirs',
    'remote issue', 'issue',
    'registrar of companies', 'ministry', 'court', 'tribunal',
    'government', 'state', 'republic', 'authority',
    'cyprus', 'belize', 'republic of cyprus', 'united kingdom',
    'united states', 'germany', 'france', 'spain', 'italy',
}


# ===============================
# REGEX PATTERNS FOR COMPANIES
# ===============================
COMPANY_FULL_PATTERNS = [
    # English / international legal forms
    r'[A-Z0-9][A-Za-z0-9&\'\-\.]+(?:\s+[A-Za-z0-9&\'\-\.]+)*\s+(?i:Ltd\.?|Limited|LLC|L\.L\.C\.?|LLD\.?|LLP|L\.L\.P\.?|Inc\.?|Incorporated|Corp\.?|Corporation|PLC|P\.L\.C\.?|Public\s+Ltd\.?|Public\s+Limited)(?=\s|[,;\.]|$)',
    # European legal forms
    r'[A-Z\u00c0-\u00d60-9][A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff0-9]+(?:\s+[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff0-9]+)*\s+(?:(?i:GmbH|AG|KG|OHG|GbR|S\.?A\.?R?\.?L?\.?|SAS|S\.?A\.?S\.?|S\.?L\.?|S\.?p\.?A\.?|S\.?r\.?l\.?|N\.?V\.?|B\.?V\.?|Pty\.?\s*Ltd\.?|Oy|ApS)|A\.?S\.?|A/S|AB)(?=\s|[,;\.]|$)',
    # Russian/Cyrillic legal forms (ООО, АО, ЗАО, ПАО, ОАО, etc.)
    r'(?:ООО|АО|ЗАО|ПАО|ОАО)\s*«[^»]+»',
]


# ===============================
# REGEX PATTERNS FOR DATES
# ===============================
_EN_MONTHS = (
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December'
    r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
)

DATE_PATTERNS = [
    r'\b\d{4}-\d{2}-\d{2}\b',
    r'\b\d{1,2}[.\/\-]\d{1,2}[.\/\-]\d{4}\b',
    r'\b\d{1,2}[.\/\-]\d{1,2}[.\/\-]\d{2}\b',
    r'\b\d{1,2}\s+' + _EN_MONTHS + r'\.?\s+\d{2,4}\b',
    r'\b' + _EN_MONTHS + r'\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
    r'\b' + _EN_MONTHS + r'\.?\s+\d{4}\b',
    r'\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}\b',
    # "15th day of March, 2022" / "1st day of January, 2020"
    r'\b\d{1,2}(?:st|nd|rd|th)\s+day\s+of\s+' + _EN_MONTHS + r',?\s+\d{4}\b',
    # "24th of July, 2015" / "1st of January 2020"
    r'\b\d{1,2}(?:st|nd|rd|th)\s+of\s+' + _EN_MONTHS + r',?\s+\d{4}\b',
]


# ===============================
# REGEX PATTERNS FOR ADDRESSES
# ===============================
ADDRESS_PATTERNS = [
    # "123 Main Street, City" / "191 ATHALASSIS AVE., P.O.Box 25525, LEFKOSIA-CYPRUS"
    r'\b\d+[A-Za-z]?(?:\s*[-/]\s*\d+[A-Za-z]?)?\s+[A-Za-z][A-Za-z \-]{2,40}(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Place|Pl\.?|Square|Sq\.?|Way|Crescent|Cres\.?|Close|Terrace|Ter\.?|Parkway|Pkwy\.?)(?:\s*,\s*(?:P\.O\.?\s*Box\s*\d+|[A-Za-z][A-Za-z \-]*[A-Za-z]))*',
    # "str. Dourleion 16904/53" / "ul. Name 123" with optional next line for zip code and city
    r'(?i:\b(?:str|ul)\.\s+[A-Za-z][A-Za-z \-]+\d[\d/A-Za-z]*(?:\s*[\n\r]+\s*\d{4,5}\s+[A-Za-z \-]+)?)',
    # UK postcode
    r'\b[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2}\b',
    # Multiline Address fallback (e.g. "82 Akropoleos, 2nd floor\n1012 Acropolis, Cyprus")
    r'\b\d+\s+[A-Za-z][A-Za-z \-]+(?:,\s*\d+(?:st|nd|rd|th)?\s+floor)?\s*[\n\r]+\s*\d{4,5}\s+[A-Za-z][A-Za-z \-]+(?:,\s*[A-Za-z]+)?',
]

# ===============================
# REGEX PATTERNS FOR REGISTRATION IDS
# ===============================
REG_ID_PATTERNS = [
    # Cyprus: H.E.107777, HE317807
    r'H\.?E\.?\s*\d{4,10}',
    # Germany: HRB 12345
    r'HRB\s*\d{4,10}',
    # UK: Company No. 12345678
    r'(?i:Company\s+No\.?\s*\d{4,10})',
    # Generic: Reg. No. 12345, Registration No. 12345
    r'(?i:Reg(?:istration)?\.?\s*No\.?\s*\d{4,10})',
]


# ===============================
# REGEX PATTERNS FOR BANK ACCOUNTS
# ===============================
BANK_ACCOUNT_PATTERNS = [
    # IBAN: 2 letter country code + 2 check digits + up to 30 alphanumeric
    r'\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}(?:\s?[A-Z0-9]{4}){0,5}(?:\s?[A-Z0-9]{1,4})?\b',
    # SWIFT/BIC codes: 8 or 11 alphanumeric characters (e.g. BARCGB22XXX)
    r'(?i:(?:SWIFT|BIC)\s*(?:code)?\s*[:.]?\s*)([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)',
    # Standalone SWIFT/BIC pattern (8 or 11 chars, typical format)
    r'\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b',
    # Account numbers near keywords: "Account No. 1234567890" / "A/C 1234567890"
    r'(?i:(?:account|acct|a/c)\s*(?:no\.?|number|#)?\s*[:.]?\s*)(\d[\d\s\-]{5,25}\d)',
    # Sort code: 12-34-56
    r'(?i:sort\s*code\s*[:.]?\s*)(\d{2}-\d{2}-\d{2})',
]


# ===============================
# REGEX PATTERNS FOR PHONES
# ===============================
PHONE_PATTERNS = [
    r'(?i:\b(?:tel|fax|phone|mobile|mob)\.?\s*(?:\+[\d\s\-\.()]{7,20}|[\d\s\-\.()]{7,20})\b)',
    r'\b\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
]


# ===============================
# ENTITY VALIDATION
# ===============================
_FRAGMENT_INDICATORS = [
    'the company', 'of the', 'in the', 'by the', 'to the', 'for the',
    'at the', 'on the', 'from the', 'with the', 'and the', 'or the',
    'shall', 'may', 'must', 'will', 'would', 'should', 'could',
    'provided that', 'subject to', 'pursuant to', 'in accordance',
    'herein', 'hereof', 'hereto', 'hereby', 'hereunder',
    'including', 'excluding', 'except',
    '\n',
]


def validate_entity(name):
    name = name.strip()
    if len(name) < 3:
        return False
    if name.lower() in GENERIC_TERMS_LOWER:
        return False
    cleaned = re.sub(r'[\[\]\.\(\)\s_\-]', '', name)
    if not cleaned or not any(c.isalpha() for c in cleaned):
        return False
    name_lower = name.lower()
    for indicator in _FRAGMENT_INDICATORS:
        if indicator in name_lower:
            return False
    words = name.split()
    if len(words) == 1:
        if not (name.isupper() and len(name) >= 2):
            return False
    return True


def validate_date(date_str):
    date_str = date_str.strip()
    if len(date_str) < 4:
        return False
    duration_words = {'days', 'day', 'weeks', 'week', 'months', 'month', 'years', 'year'}
    words = date_str.lower().split()
    if any(w in duration_words for w in words):
        return False
    if re.fullmatch(r'\d{4}', date_str):
        return False
    if not any(c.isdigit() for c in date_str):
        return False

    # Reject section/article numbers like "2.2.11", "1.3.5", "10.2.1"
    # These are dot-separated numbering that the date regex can match
    section_match = re.fullmatch(r'(\d{1,2})\.(\d{1,2})\.(\d{1,4})', date_str)
    if section_match:
        a, b, c = int(section_match.group(1)), int(section_match.group(2)), int(section_match.group(3))
        # Real dates: DD.MM.YYYY (day 1-31, month 1-12, year >=1900)
        # or DD.MM.YY  (day 1-31, month 1-12, year 0-99)
        is_plausible_date = (
            1 <= a <= 31 and 1 <= b <= 12 and (
                c >= 1900 or  # DD.MM.YYYY
                (c <= 99 and a <= 31 and b <= 12)  # DD.MM.YY
            )
        )
        # Also check MM.DD.YYYY (US format)
        is_plausible_us = (
            1 <= a <= 12 and 1 <= b <= 31 and c >= 1900
        )
        if not (is_plausible_date or is_plausible_us):
            return False

    return True


def validate_address(addr_str):
    addr_str = addr_str.strip()
    if len(addr_str) < 5:
        return False
    if not any(c.isalpha() for c in addr_str):
        return False

    addr_lower = addr_str.lower()

    # Reject obvious sentence fragments (legal/contract language)
    reject_phrases = [
        'shall', 'will be', 'would', 'should', 'could', 'must',
        'hours after', 'days after', 'time it has been',
        'provided that', 'subject to', 'pursuant to', 'in accordance',
        'in accord', 'notwithstanding', 'herein', 'hereof', 'hereto', 'hereby',
        'the tenant', 'the landlord', 'the company', 'the parties',
        'terminate', 'terminated', 'agreement', 'obligation',
        'or any law', 'unless',
        'be read', 'regulation', 'article', 'section', 'clause',
        'cap.', 'amending', 'substitut', 'earlier',
        'whichever', 'forthwith', 'reasonable', 'written notice',
        'liability', 'indemnity', 'warranty', 'covenant',
        'whereas', 'witnesseth', 'stipulat',
        'extraordinary', 'ordinary', 'resolution', 'meeting',
        'general meeting', 'shareholder', 'dividend', 'quorum',
        'registered office', 'memorandum', 'constitution',
        'paragraph', 'sub-clause', 'schedule', 'appendix', 'annex',
        'approval', 'consent', 'notice of', 'right to',
    ]
    for phrase in reject_phrases:
        if phrase in addr_lower:
            return False

    # Reject if it starts with a bare year (e.g. "2025 unless terminated...")
    if re.match(r'^\d{4}\s', addr_str):
        return False

    # Reject if it looks like a date fragment
    if re.match(r'^\d{1,2}/\d{2,4}\b', addr_str):
        return False

    return True


def validate_phone(phone_str, full_text=None):
    phone_str = phone_str.strip()
    digits = re.sub(r'\D', '', phone_str)
    if len(digits) < 6:
        return False
    # If we have the full text, check context around this number
    # Reject if it appears near account/bank keywords
    if full_text:
        # Find the number in the full text and check surrounding words
        pos = full_text.find(phone_str)
        if pos == -1:
            pos = full_text.lower().find(phone_str.lower())
        if pos != -1:
            # Look at 60 chars before and after for context
            start = max(0, pos - 60)
            end = min(len(full_text), pos + len(phone_str) + 60)
            context = full_text[start:end].lower()
            account_keywords = [
                'account', 'acct', 'a/c', 'iban', 'swift', 'bic',
                'bank', 'deposit', 'routing', 'sort code',
            ]
            if any(kw in context for kw in account_keywords):
                return False
    return True


# ===============================
# ENTITY GROUPING
# ===============================
_LEGAL_SUFFIX_NORMALIZE = [
    (r'\bLtd\.?\b', 'LIMITED'),
    (r'\bInc\.?\b', 'INCORPORATED'),
    (r'\bCorp\.?\b', 'CORPORATION'),
    (r'\bL\.?L\.?C\.?\b', 'LLC'),
    (r'\bL\.?L\.?P\.?\b', 'LLP'),
    (r'\bP\.?L\.?C\.?\b', 'PLC'),
    (r'\bPty\.?\b', 'PTY'),
    (r'\bGmbH\b', 'GMBH'),
]


def normalize_org_name(name):
    """Normalize company name by expanding legal suffix abbreviations.
    E.g. 'DEMETRA INVESTMENTS PUBLIC LTD' -> 'DEMETRA INVESTMENTS PUBLIC LIMITED'"""
    n = name.strip().rstrip('.')
    for pattern, replacement in _LEGAL_SUFFIX_NORMALIZE:
        n = re.sub(pattern, replacement, n, flags=re.IGNORECASE)
    return n.upper()


def group_org_variants(organizations):
    """Group orgs with same suffix-normalized form. Keep longest as canonical."""
    groups = []
    group_norms = []

    for org in organizations:
        matched_idx = None
        org_norm = normalize_org_name(org)
        for i, norm in enumerate(group_norms):
            if org_norm == norm:
                matched_idx = i
                break

        if matched_idx is not None:
            groups[matched_idx][1].append(org)
            if len(org) > len(groups[matched_idx][0]):
                groups[matched_idx] = (org, groups[matched_idx][1])
                group_norms[matched_idx] = normalize_org_name(org)
        else:
            groups.append((org, [org]))
            group_norms.append(org_norm)

    return groups


def group_person_variants(persons):
    """Simple dedup for persons. Keep longest variant."""
    groups = []
    seen_lower = {}
    for person in persons:
        key = person.lower().strip()
        if key in seen_lower:
            idx = seen_lower[key]
            groups[idx][1].append(person)
            if len(person) > len(groups[idx][0]):
                groups[idx] = (person, groups[idx][1])
        else:
            seen_lower[key] = len(groups)
            groups.append((person, [person]))
    return groups


# ===============================
# NAME PATTERN MATCHING
# ===============================
def build_name_pattern(name):
    """Build a regex that matches a person name (case-insensitive, exact tokens)."""
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    regex_parts = []
    for part in parts:
        if not part:
            continue
        regex_parts.append(re.escape(part))
    return r'\s+'.join(regex_parts) if regex_parts else None


def build_person_patterns(person_groups, mapping):
    """Build list of (compiled_regex, placeholder) for name matching."""
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
                full_pat = r'(?<![A-Za-z])' + pat + r'(?![A-Za-z])'
                try:
                    patterns.append((re.compile(full_pat, re.IGNORECASE), placeholder))
                except re.error:
                    pass
    return patterns


# ===============================
# CORE FUNCTIONS
# ===============================
def combine_pages(pages):
    sorted_pages = sorted(pages, key=lambda p: p["page"])
    return "\n\n".join(p["text"] for p in sorted_pages)


def chunk_text_with_overlap(text, max_tokens=4500, overlap_tokens=200):
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
    log(f"Split into {len(chunks)} chunks ({len(tokens)} tokens total)")
    return chunks


MAX_CHUNKS = 80


SYSTEM_PROMPT = """You are a multilingual named entity recognition (NER) assistant for legal and business documents.

Extract ALL of the following from the text:
1. Person names (actual human names only)
2. Organisation / company names (actual registered business names only)
3. Dates (specific calendar dates only)
4. Addresses (physical street/postal addresses in any language)
5. Phone numbers (phone and fax numbers)
6. Registration IDs (company registration numbers, tax IDs)
7. Bank accounts (IBAN numbers, bank account numbers, SWIFT/BIC codes)

CRITICAL RULES - what to extract:
- PERSONS: Only real human names, like "John Smith", "Andreas Menelaou"
  - Extract person names EVEN when they appear in an official capacity
  - Extract person names from witnesses, signatories, advocates
  - If a person's name is used as a business/firm name, extract it as BOTH a person AND an organisation
- ORGANISATIONS: Only actual named companies/firms
  - Extract ALL language variants of the same company
  - Include companies in any language: English, Greek, Russian (e.g., ООО, АО), French, German, etc.
- DATES: Only specific calendar dates, like "01/09/2015", "24th of July, 2015"
  - Do NOT extract section or article numbers as dates (e.g. "2.2.11", "3.1.5" are section numbers, NOT dates)
- ADDRESSES: Physical street/postal addresses in ANY language
  - Extract addresses from EVERYWHERE: signature pages, witness sections, headers, body text
  - Make sure to extract multiple lines if the address spans multiple lines (including floor numbers, postcodes, and cities)
  - Addresses can be in any format and any language
  - Do NOT extract sentence fragments that happen to mention numbers
- PHONES: Phone and fax numbers, like "+357 22 315161", "22314641"
- REGISTRATION IDS: Any company or entity identification numbers
  - Examples: "H.E.107777", "HE317807", "HRB 12345", "Company No. 12345678", "Reg. No. 123456", "Tax ID 123456"
- BANK ACCOUNTS: Bank account numbers, IBAN codes, SWIFT/BIC codes
  - Examples: "CY17 0020 0128 0000 0012 0052 7600", "BCYPCY2N", "Account No. 0120052760"
  - Include ANY numbers explicitly labeled as bank accounts, deposit accounts, or payment accounts

CRITICAL RULES - what NOT to extract:
- Do NOT extract role titles as persons: Chairman, Director, Secretary, Landlord, Tenant
- Do NOT extract generic legal terms as organisations: "the Company", "Board of Directors"
- Do NOT extract countries alone as organisations
- Do NOT extract time durations as dates: "fourteen days", "six months"
- Do NOT extract bare years as dates: "2014" alone is NOT a date
- Do NOT extract section/article numbers as dates: "2.2.11", "3.1.5" are NOT dates
- Do NOT extract sentence fragments as addresses
- Do NOT extract bank account numbers, IBAN codes, or reference numbers as phone numbers

Output ONLY valid JSON with no explanation or thinking. Do not wrap in markdown.

{
  "persons": ["name1", "name2"],
  "organizations": ["org1", "org2"],
  "dates": ["date1", "date2"],
  "addresses": ["addr1", "addr2"],
  "phones": ["phone1", "phone2"],
  "registration_ids": ["H.E.107777"],
  "bank_accounts": ["CY17 0020 0128 0000 0012 0052 7600"]
}"""


def strip_thinking(text):
    """Remove <think>...</think> blocks from model output (safety net)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def extract_entities_llm(text_chunk):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract all persons, organizations, dates, addresses, phones, registration IDs, and bank accounts:\n\n{text_chunk}"}
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
    log(f"LLM raw (first 500): {raw_output[:500]}")

    # Strip any <think> blocks (safety net for thinking models)
    cleaned_output = strip_thinking(raw_output)
    if cleaned_output != raw_output:
        log(f"Stripped think tags, cleaned (first 500): {cleaned_output[:500]}")

    try:
        json_match = re.search(r'\{[^{}]*"persons"\s*:.*\}', cleaned_output, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
        result = json.loads(json_match.group()) if json_match else json.loads(cleaned_output)
        persons = [p.strip() for p in result.get("persons", []) if p and p.strip()]
        organizations = [o.strip() for o in result.get("organizations", []) if o and o.strip()]
        dates = [d.strip() for d in result.get("dates", []) if d and d.strip()]
        addresses = [a.strip() for a in result.get("addresses", []) if a and a.strip()]
        phones = [p.strip() for p in result.get("phones", []) if p and p.strip()]
        reg_ids = [r.strip() for r in result.get("registration_ids", []) if r and r.strip()]
        bank_accounts = [b.strip() for b in result.get("bank_accounts", []) if b and b.strip()]
        log(f"LLM extracted: {len(persons)}p {len(organizations)}o {len(dates)}d {len(addresses)}a {len(phones)}ph {len(reg_ids)}reg {len(bank_accounts)}bank")
        return persons, organizations, dates, addresses, phones, reg_ids, bank_accounts
    except (json.JSONDecodeError, AttributeError) as e:
        log(f"Parse failed: {e}")
        log(f"Cleaned output was: {cleaned_output[:1000]}")
        return [], [], [], [], [], [], []


# ===============================
# REGEX DETECTION FUNCTIONS
# ===============================
def detect_companies_regex(text):
    organizations = []
    for pattern in COMPANY_FULL_PATTERNS:
        for match in re.finditer(pattern, text):
            clean = match.group().strip()
            if clean and clean not in organizations:
                organizations.append(clean)
    log(f"Regex found {len(organizations)} orgs")
    return organizations


def detect_dates_regex(text):
    dates = []
    seen = set()
    for pattern in DATE_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            token = m.group().strip()
            if token and token.lower() not in seen:
                seen.add(token.lower())
                dates.append(token)
    return dates


def detect_addresses_regex(text):
    addresses = []
    seen = set()
    for pattern in ADDRESS_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            token = m.group().strip()
            if token and token.lower() not in seen:
                seen.add(token.lower())
                addresses.append(token)
    return addresses


def detect_phones_regex(text):
    phones = []
    seen = set()
    for pattern in PHONE_PATTERNS:
        for m in re.finditer(pattern, text):
            token = m.group().strip()
            if token and token.lower() not in seen and len(re.sub(r'\D', '', token)) >= 6:
                seen.add(token.lower())
                phones.append(token)
    return phones


def detect_bank_accounts_regex(text):
    """Detect bank account numbers, IBANs, SWIFT/BIC codes via regex."""
    accounts = []
    seen = set()
    for pattern in BANK_ACCOUNT_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            # Some patterns have capture groups for the actual value
            token = (m.group(1) if m.lastindex and m.group(1) else m.group()).strip()
            if token and token.lower() not in seen and len(token) >= 6:
                seen.add(token.lower())
                accounts.append(token)
    return accounts


def validate_bank_account(acct_str):
    """Validate a bank account string."""
    acct_str = acct_str.strip()
    if len(acct_str) < 6:
        return False
    # Must contain at least some digits
    if not any(c.isdigit() for c in acct_str):
        return False
    return True


# ===============================
# REPLACEMENT FUNCTIONS
# ===============================
def _flexible_pattern(text_str):
    """Build a regex that matches text_str with flexible whitespace,
    but NEVER matches in the middle of a word."""
    escaped = re.escape(text_str)
    flexible = escaped.replace(r'\ ', r'\s+')
    # Word boundary guards: prevent matching inside words
    # Left:  must not be preceded by a word character (letter/digit/underscore)
    # Right: must not be followed by a word character
    return r'(?<!\w)' + flexible + r'(?!\w)'


def replace_dates(text, date_map):
    for date_str in sorted(date_map.keys(), key=len, reverse=True):
        pattern = _flexible_pattern(date_str)
        text = re.sub(pattern, date_map[date_str], text, flags=re.IGNORECASE)
    return text


def replace_addresses(text, addr_map):
    for addr_str in sorted(addr_map.keys(), key=len, reverse=True):
        pattern = _flexible_pattern(addr_str)
        text = re.sub(pattern, addr_map[addr_str], text, flags=re.IGNORECASE)
    return text


def replace_phones(text, phone_map):
    for phone_str in sorted(phone_map.keys(), key=len, reverse=True):
        pattern = _flexible_pattern(phone_str)
        text = re.sub(pattern, phone_map[phone_str], text, flags=re.IGNORECASE)
    return text


def safe_replace(text, mapping, name_patterns=None):
    sorted_entities = sorted(mapping.keys(), key=len, reverse=True)
    for entity in sorted_entities:
        placeholder = mapping[entity]
        escaped = re.escape(entity)
        pattern = r'(?<![A-Za-z])' + escaped + r'(?![A-Za-z])'
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    if name_patterns:
        for compiled_re, placeholder in name_patterns:
            text = compiled_re.sub(placeholder, text)
    return text


def dedup_substrings(items):
    if not items:
        return items
    sorted_items = sorted(items, key=len, reverse=True)
    result = []
    for item in sorted_items:
        item_lower = item.lower()
        is_sub = False
        for accepted in result:
            if item_lower in accepted.lower():
                is_sub = True
                break
        if not is_sub:
            result.append(item)
    return result


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
        norm = n.rstrip('. ').lower()
        if n and norm not in seen_o and validate_entity(n):
            seen_o.add(norm)
            organizations.append(n)

    log(f"After validation: {len(persons)} persons, {len(organizations)} orgs")
    return persons, organizations


def build_ordered_mapping(full_text, person_groups, org_groups):
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
        placeholder = f"[COMPANY{idx}]"
        for v in variants:
            mapping[v] = placeholder

    for idx, (canonical, variants, _) in enumerate(person_positions, 1):
        placeholder = f"[PERSON{idx}]"
        for v in variants:
            mapping[v] = placeholder

    log(f"Mapping ({len(mapping)} entries):")
    shown = set()
    for entity, ph in mapping.items():
        if ph not in shown:
            shown.add(ph)
            log(f"  {ph} <- {entity}")

    return mapping


# ===============================
# MAIN ANONYMIZATION PIPELINE
# ===============================
def anonymize_document(pages):
    total_start = time.time()
    log(f"Processing {len(pages)} pages")

    full_text = combine_pages(pages)
    log(f"Combined: {len(full_text)} chars")

    chunks = chunk_text_with_overlap(full_text)

    if len(chunks) > MAX_CHUNKS:
        return {
            "error": (
                f"Document too large: {len(chunks)} chunks required but the maximum is {MAX_CHUNKS}. "
                "Please split the document and process it in parts."
            )
        }

    all_persons, all_orgs, all_dates, all_addresses, all_phones, all_reg_ids, all_bank_accounts = [], [], [], [], [], [], []
    ner_start = time.time()
    for i, chunk in enumerate(chunks):
        t = time.time()
        log(f"Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        persons, orgs, dates, addresses, phones, reg_ids, bank_accounts = extract_entities_llm(chunk)
        all_persons.extend(persons)
        all_orgs.extend(orgs)
        all_dates.extend(dates)
        all_addresses.extend(addresses)
        all_phones.extend(phones)
        all_reg_ids.extend(reg_ids)
        all_bank_accounts.extend(bank_accounts)
        log(f"Chunk {i+1} done in {time.time()-t:.1f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    log(f"NER done in {time.time()-ner_start:.1f}s — {len(all_persons)}p {len(all_orgs)}o {len(all_dates)}d {len(all_addresses)}a {len(all_phones)}ph {len(all_reg_ids)}reg {len(all_bank_accounts)}bank (raw)")

    # Regex fallback: catch anything the LLM missed
    regex_orgs = detect_companies_regex(full_text)
    regex_dates = detect_dates_regex(full_text)
    regex_addresses = detect_addresses_regex(full_text)
    regex_phones = detect_phones_regex(full_text)
    regex_bank_accounts = detect_bank_accounts_regex(full_text)

    persons, organizations = merge_entities(all_persons, all_orgs, regex_orgs)

    # Deduplicate dates (LLM + regex)
    seen_d = set()
    unique_dates = []
    for d in all_dates + regex_dates:
        d = d.strip()
        if d and d.lower() not in seen_d and validate_date(d):
            seen_d.add(d.lower())
            unique_dates.append(d)
    unique_dates = dedup_substrings(unique_dates)
    log(f"Dates: {len(unique_dates)} unique")

    # Deduplicate addresses (LLM + regex)
    seen_a = set()
    unique_addresses = []
    for a in all_addresses + regex_addresses:
        a = a.strip()
        if a and a.lower() not in seen_a and validate_address(a):
            seen_a.add(a.lower())
            unique_addresses.append(a)
    unique_addresses = dedup_substrings(unique_addresses)
    log(f"Addresses: {len(unique_addresses)} unique")

    # Deduplicate phones (LLM + regex)
    seen_ph = set()
    unique_phones = []
    for ph in all_phones + regex_phones:
        ph = ph.strip()
        if ph and ph.lower() not in seen_ph and validate_phone(ph, full_text):
            seen_ph.add(ph.lower())
            unique_phones.append(ph)
    unique_phones = dedup_substrings(unique_phones)
    log(f"Phones: {len(unique_phones)} unique")

    # Group variants
    person_groups = group_person_variants(persons)
    org_groups = group_org_variants(organizations)
    log(f"Grouped: {len(person_groups)} person groups, {len(org_groups)} org groups")

    mapping = build_ordered_mapping(full_text, person_groups, org_groups)

    # Build date/address/phone mappings ordered by first appearance
    def find_first_pos(token):
        pos = full_text.find(token)
        if pos == -1:
            pos = full_text.lower().find(token.lower())
        return pos if pos != -1 else float('inf')

    date_map = {}
    for i, d in enumerate(sorted(unique_dates, key=find_first_pos), 1):
        date_map[d] = f"[DATE{i}]"

    addr_map = {}
    for i, a in enumerate(sorted(unique_addresses, key=find_first_pos), 1):
        addr_map[a] = f"[ADDRESS{i}]"

    phone_map = {}
    for i, p in enumerate(sorted(unique_phones, key=find_first_pos), 1):
        phone_map[p] = f"[PHONE{i}]"

    # Detect H.E. registration IDs via regex
    reg_ids = []
    seen_reg = set()
    for pattern in REG_ID_PATTERNS:
        for m in re.finditer(pattern, full_text, flags=re.IGNORECASE):
            token = m.group().strip()
            if token and token.lower() not in seen_reg:
                seen_reg.add(token.lower())
                reg_ids.append(token)
    # Also add any from LLM extraction
    for chunk_result in all_reg_ids:
        token = chunk_result.strip()
        if token and token.lower() not in seen_reg:
            seen_reg.add(token.lower())
            reg_ids.append(token)

    reg_id_map = {}
    for i, r in enumerate(sorted(reg_ids, key=find_first_pos), 1):
        reg_id_map[r] = f"[REG_ID{i}]"

    # Deduplicate bank accounts (LLM + regex)
    seen_bank = set()
    unique_bank_accounts = []
    for ba in all_bank_accounts + regex_bank_accounts:
        ba = ba.strip()
        if ba and ba.lower() not in seen_bank and validate_bank_account(ba):
            seen_bank.add(ba.lower())
            unique_bank_accounts.append(ba)
    unique_bank_accounts = dedup_substrings(unique_bank_accounts)
    log(f"Bank accounts: {len(unique_bank_accounts)} unique")

    bank_account_map = {}
    for i, ba in enumerate(sorted(unique_bank_accounts, key=find_first_pos), 1):
        bank_account_map[ba] = f"[BANK_ACCOUNT{i}]"

    log(f"Date mapping: {len(date_map)} entries")
    for orig, ph in date_map.items():
        log(f"  {ph} <- {orig}")
    log(f"Address mapping: {len(addr_map)} entries")
    for orig, ph in addr_map.items():
        log(f"  {ph} <- {orig}")
    log(f"Phone mapping: {len(phone_map)} entries")
    for orig, ph in phone_map.items():
        log(f"  {ph} <- {orig}")
    log(f"Reg ID mapping: {len(reg_id_map)} entries")
    for orig, ph in reg_id_map.items():
        log(f"  {ph} <- {orig}")
    log(f"Bank account mapping: {len(bank_account_map)} entries")
    for orig, ph in bank_account_map.items():
        log(f"  {ph} <- {orig}")

    # Build person name patterns
    name_patterns = build_person_patterns(person_groups, mapping) if mapping else []

    # Replace all entities page by page
    anonymized_pages = []
    for page in sorted(pages, key=lambda p: p["page"]):
        anon_text = page["text"]
        if mapping:
            anon_text = safe_replace(anon_text, mapping, name_patterns)
        anon_text = replace_addresses(anon_text, addr_map)
        anon_text = replace_dates(anon_text, date_map)
        anon_text = replace_phones(anon_text, phone_map)
        # Replace H.E. registration IDs
        for reg_str in sorted(reg_id_map.keys(), key=len, reverse=True):
            pattern = _flexible_pattern(reg_str)
            anon_text = re.sub(pattern, reg_id_map[reg_str], anon_text, flags=re.IGNORECASE)
        # Replace bank account numbers
        for ba_str in sorted(bank_account_map.keys(), key=len, reverse=True):
            pattern = _flexible_pattern(ba_str)
            anon_text = re.sub(pattern, bank_account_map[ba_str], anon_text, flags=re.IGNORECASE)
        anonymized_pages.append({"page": page["page"], "text": anon_text})

    # Build display mapping
    display_mapping = {}
    for canonical, variants in org_groups:
        ph = mapping.get(canonical) or mapping.get(variants[0])
        if ph:
            display_mapping[canonical] = ph
    for canonical, variants in person_groups:
        ph = mapping.get(canonical) or mapping.get(variants[0])
        if ph:
            display_mapping[canonical] = ph
    display_mapping.update(addr_map)
    display_mapping.update(date_map)
    display_mapping.update(phone_map)
    display_mapping.update(reg_id_map)
    display_mapping.update(bank_account_map)

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
