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
# (Russian + English + common multilingual terms)
# ===============================
GENERIC_TERMS_LOWER = {
    # Russian generic terms
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
    # English generic terms
    'company', 'companies', 'the company', 'the companies',
    'party', 'parties', 'the party', 'the parties',
    'borrower', 'lender', 'lessor', 'lessee', 'tenant', 'landlord',
    'client', 'contractor', 'counterparty', 'agent', 'principal',
    'employer', 'employee', 'vendor', 'buyer', 'seller', 'purchaser',
    'assignor', 'assignee', 'guarantor', 'beneficiary',
    'director', 'general director', 'ceo', 'shareholder',
    'authorized person', 'representative', 'signatory',
    'group company', 'group companies', 'subsidiary', 'subsidiaries',
}


# ===============================
# REGEX PATTERNS FOR COMPANIES (Russian + International)
# ===============================
COMPANY_FULL_PATTERNS = [
    # ---- Russian legal forms ----
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
    # ---- English / Anglo-Saxon legal forms ----
    # "Acme Ltd", "Acme Limited", "Acme LLC", "Acme Inc.", "Acme Corp.", "Acme Co."
    r'[A-Z][A-Za-z0-9&,\.\s\-]{1,60}?\s+(?:Ltd\.?|Limited|LLC|L\.L\.C\.?|LLP|L\.L\.P\.?|Inc\.?|Incorporated|Corp\.?|Corporation|Co\.?|Company|PLC|P\.L\.C\.?)(?=\s|[,;\.]|$)',
    # "The Acme Corporation" / "Acme International" (capitalised multi-word names before comma/period)
    r'(?:The\s+)?[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){1,5}\s+(?:Group|Holdings|Enterprises|Industries|Services|Solutions|Technologies|Partners|Associates|Consulting|Capital|Finance|Investments)',
    # ---- European / other legal forms ----
    # German: GmbH, AG, KG, OHG; French: S.A., S.A.R.L., S.A.S.; Spanish/Italian: S.L., S.p.A., S.r.l.
    r'[A-ZÀ-Ö][A-Za-zÀ-ÖØ-öø-ÿ0-9&,\.\s\-]{1,60}?\s+(?:GmbH|AG|KG|OHG|GbR|S\.?A\.?R?\.?L?\.?|SAS|S\.?A\.?S\.?|S\.?L\.?|S\.?p\.?A\.?|S\.?r\.?l\.?|N\.?V\.?|B\.?V\.?|A\.?S\.?|A/S|Pty\.?\s*Ltd\.?|AB|Oy|ApS)(?=\s|[,;\.]|$)',
    # ---- Quoted organisation names (any language) ----
    # Catches: Organisation «Name», Organization "Name", Firma 'Name'
    r'[A-ZА-ЯЁ\u4E00-\u9FFF][\w\s\-\.]{0,40}?\s*[«"\u201C][^»"\u201D\n]{2,60}[»"\u201D]',
]


# ===============================
# REGEX PATTERNS FOR DATES (multilingual)
# ===============================
_RU_MONTHS = r'(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)'
_RU_MONTHS_NOM = r'(?:январе?|феврале?|марте?|апреле?|мае?|июне?|июле?|августе?|сентябре?|октябре?|ноябре?|декабре?)'
_EN_MONTHS = (
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December'
    r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
)
_DE_MONTHS = r'(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)'
_FR_MONTHS = r'(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)'
_ES_MONTHS = r'(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)'
_ALL_MONTHS = f'(?:{_EN_MONTHS}|{_RU_MONTHS}|{_RU_MONTHS_NOM}|{_DE_MONTHS}|{_FR_MONTHS}|{_ES_MONTHS})'

DATE_PATTERNS = [
    # ISO: YYYY-MM-DD
    r'\b\d{4}-\d{2}-\d{2}\b',
    # Numeric with separators: DD.MM.YYYY / DD/MM/YYYY / MM-DD-YYYY etc.
    r'\b\d{1,2}[.\/-]\d{1,2}[.\/-]\d{2,4}\b',
    # "12 January 2023" / "12 января 2023 г." / "12 janvier 2023" / "12 März 2023"
    r'\b\d{1,2}\s+' + _ALL_MONTHS + r'\.?\s+\d{2,4}(?:\s+г(?:ода)?\.?)?\b',
    # "January 12, 2023" / "January 12th, 2023"
    r'\b' + _EN_MONTHS + r'\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
    # "January 2023" / "январь 2023" / "März 2023"
    r'\b' + _ALL_MONTHS + r'\.?\s+\d{4}\s*(?:г(?:ода)?\.?)?\b',
    # Russian: "в 2023 году"
    r'\bв\s+\d{4}\s+году?\b',
    # "«__» ________ 20__ г." blank-date template lines
    r'«__»\s*_{3,}\s*\d{2,4}\s*(?:г\.?|года)?',
    # Plain 4-digit year with word-boundary (last resort, after above patterns)
    r'\b(?:19|20)\d{2}\s+(?:г(?:ода)?|year|año|Jahr|an)\b',
]


# ===============================
# REGEX PATTERNS FOR ADDRESSES (Russian + International)
# ===============================
ADDRESS_PATTERNS = [
    # ---- Russian addresses ----
    # Full address with 6-digit postal code: "123456, г. Москва, ул. Ленина, д. 5, кв. 10"
    r'\b\d{6},?\s*(?:[гГ](?:ород|\.)\s*[А-ЯЁа-яё\-]+,?\s*)?(?:[а-яёА-ЯЁ][а-яёА-ЯЁ\s\.\-]+,?\s*(?:[дД][.\/]?\s*\d+[а-яёА-ЯЁ]?(?:/\d+)?(?:,?\s*(?:[кК](?:орп)?|[сСтТ](?:тр)?)[.\/]?\s*\d+)?(?:,?\s*(?:[кКqQ]в|офис|оф\.?)\s*\.?\s*\d+)?)?)\b',
    # Street keyword: ул., пр., пер., бульвар, набережная ...
    r'(?:(?:[уУ]л(?:ица|\.)?|[пП]р(?:оспект|[\.\-])?|[пП]ер(?:еулок|\.)?|[шШ]оссе|[бБ]ульвар|[нН]аб(?:ережная|\.)?|[пП]лощадь|[пП]л\.?|[пП]роезд)\s+[А-ЯЁа-яё][А-ЯЁа-яё\s\-]+,?\s*(?:[дД][.\/]?\s*\d+[а-яё]?(?:/\d+)?)(?:,?\s*(?:[кК](?:орп)?|[сСтТ](?:тр)?)[.\/]?\s*\d+)?(?:,?\s*(?:[кКqQ]в|офис|оф\.?)\s*\.?\s*\d+)?)',
    # Russian city reference: "г. Москва", "г. Санкт-Петербург"
    r'\b[гГ]\.\s*[А-ЯЁ][а-яёА-ЯЁ\-]+(?:\s*,\s*[а-яёА-ЯЁ][а-яёА-ЯЁ\s\-]+(?:обл(?:асть|\.)?|кра[йя]|респ(?:ублик[аи]|\.)|округ))?\b',
    # ---- English / international addresses ----
    # "123 Main Street, London, UK" / "Suite 4B, 10 Downing Street"
    r'\b\d+[A-Za-z]?(?:\s*[-/]\s*\d+[A-Za-z]?)?\s+[A-Z][A-Za-z\s\.\-]{2,40},?\s*(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Place|Pl\.?|Square|Sq\.?|Way|Crescent|Cres\.?|Close|Terrace|Ter\.?|Parkway|Pkwy\.?)(?:[,\s]+[A-Z][A-Za-z\s\-]+){0,3}',
    # UK postcode: "SW1A 2AA", "EC1A 1BB"
    r'\b[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2}\b',
    # US ZIP: "90210" or "90210-1234"
    r'\b\d{5}(?:-\d{4})?\b(?=[,\s])',
    # Generic: "City, State/Country" pattern with a known keyword
    r'\b[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?,?\s+(?:USA|United States|UK|United Kingdom|Germany|France|Spain|Italy|Netherlands|Switzerland|Austria|Belgium|Poland|Czech Republic|China|Japan|UAE|Canada|Australia)\b',
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


def is_org_abbreviation(inner_a, inner_b):
    """Check if one inner name is an abbreviation of the other.
    E.g., 'УК ГРП' is abbreviation of 'Управляющая компания ГРП'."""
    words_a = inner_a.split()
    words_b = inner_b.split()

    # Determine which is shorter
    if len(words_a) >= len(words_b):
        short_words, long_words = words_b, words_a
    else:
        short_words, long_words = words_a, words_b

    if len(short_words) == 0 or len(long_words) == 0:
        return False
    if len(short_words) >= len(long_words):
        return False

    # Count matching suffix words (e.g., "ГРП" matches in both)
    shared_suffix = 0
    for i in range(1, min(len(short_words), len(long_words)) + 1):
        if short_words[-i].lower() == long_words[-i].lower():
            shared_suffix += 1
        else:
            break

    if shared_suffix == 0:
        return False

    # Remaining prefix: short should be initials of long
    remaining_short = short_words[:len(short_words) - shared_suffix]
    remaining_long = long_words[:len(long_words) - shared_suffix]

    if not remaining_short and not remaining_long:
        return True
    if not remaining_short or not remaining_long:
        return shared_suffix > 0

    # "УК" should match initials of "Управляющая Компания"
    abbrev = ''.join(remaining_short).upper()
    long_initials = ''.join(w[0] for w in remaining_long).upper()

    return abbrev == long_initials


def group_org_variants(organizations):
    """Group orgs sharing the same or abbreviated inner «name». Keep longest as canonical."""
    groups = []  # [(canonical, [all_variants])]

    for org in organizations:
        inner = extract_quoted_name(org)
        matched_idx = None

        if inner:
            for i, (canonical, variants) in enumerate(groups):
                existing_inner = extract_quoted_name(canonical)
                if existing_inner and (
                    inner == existing_inner or
                    is_org_abbreviation(inner, existing_inner)
                ):
                    matched_idx = i
                    break

        if matched_idx is not None:
            groups[matched_idx][1].append(org)
            if len(org) > len(groups[matched_idx][0]):
                groups[matched_idx] = (org, groups[matched_idx][1])
        else:
            groups.append((org, [org]))

    return groups


def split_name_parts(name):
    """Split name into parts, expanding 'Е.Б.' into separate initials ['Е', 'Б']."""
    # Insert space between adjacent initials: "Е.Б." -> "Е. Б."
    expanded = re.sub(r'([А-ЯЁ])\.([А-ЯЁ])', r'\1. \2', name.strip())
    parts = [p.strip().rstrip('.') for p in expanded.split() if p.strip()]
    return parts


def is_initial_match(name_a, name_b):
    """Check if one name is abbreviated form of the other (e.g. 'Сурова Е.Б.' vs 'Сурова Елена Борисовна')."""
    parts_a = split_name_parts(name_a)
    parts_b = split_name_parts(name_b)

    has_init_a = any(len(p) <= 1 for p in parts_a[1:]) if len(parts_a) > 1 else False
    has_init_b = any(len(p) <= 1 for p in parts_b[1:]) if len(parts_b) > 1 else False

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
        init = short_parts[j] if len(short_parts[j]) <= 1 else long_parts[j]
        full = long_parts[j] if len(short_parts[j]) <= 1 else short_parts[j]
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
# NAME DECLENSION / FUZZY MATCHING (Cyrillic + Latin)
# ===============================
def _is_cyrillic_name(name):
    """Return True if the name is primarily Cyrillic script."""
    cyrillic = sum(1 for c in name if '\u0400' <= c <= '\u04FF')
    latin = sum(1 for c in name if c.isalpha() and c.isascii())
    return cyrillic >= latin


def build_name_pattern(name):
    """Build a regex that matches a person name in declined/variant forms.

    - For Cyrillic names: uses stem-based matching to handle Russian grammatical cases.
    - For Latin names: matches exact tokens (case-insensitive), since most Latin-script
      languages don't inflect personal names.
    """
    parts = name.strip().split()
    if len(parts) < 2:
        return None

    if _is_cyrillic_name(name):
        # --- Cyrillic / Russian declension logic ---
        has_initials = bool(re.search(r'[А-ЯЁ]\.', name))
        if has_initials:
            surname_match = re.match(r'([А-ЯЁа-яё]+)', name)
            initials = re.findall(r'([А-ЯЁ])\.', name)
            if not surname_match or not initials:
                return None
            surname = surname_match.group(1)
            stem_len = max(3, len(surname) - 2)
            pat = re.escape(surname[:stem_len]) + '[а-яёА-ЯЁ]{0,4}'
            initials_pat = r'\.?\s*'.join(re.escape(i) for i in initials) + r'\.?'
            pat += r'\s+' + initials_pat
            return pat
        else:
            regex_parts = []
            for part in parts:
                if len(part) < 2:
                    continue
                stem_len = max(3, len(part) - 2)
                regex_parts.append(re.escape(part[:stem_len]) + '[а-яёА-ЯЁ]{0,4}')
            return r'\s+'.join(regex_parts) if regex_parts else None
    else:
        # --- Latin / other scripts: exact token matching (case-insensitive) ---
        # Handles initials like "J." or "J"
        regex_parts = []
        for part in parts:
            if not part:
                continue
            if re.match(r'^[A-ZА-ЯЁ]\.$', part):  # single initial with dot
                regex_parts.append(re.escape(part[0]) + r'\.?')
            elif re.match(r'^[A-ZА-ЯЁ]$', part):   # single initial without dot
                regex_parts.append(re.escape(part) + r'\.?')
            else:
                regex_parts.append(re.escape(part))
        return r'\s+'.join(regex_parts) if regex_parts else None


def build_person_patterns(person_groups, mapping):
    """Build list of (compiled_regex, placeholder) for declined/variant name matching."""
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
                # Universal word-boundary that works for both Cyrillic and Latin
                full_pat = r'(?<![\w\u0400-\u04FF])' + pat + r'(?![\w\u0400-\u04FF])'
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


SYSTEM_PROMPT = """You are a multilingual named entity recognition (NER) assistant for legal and business documents.

The document may be in ANY language (Russian, English, German, French, Arabic, Chinese, etc.) or a mix of languages.
You must detect and extract entities regardless of the language they appear in.

Extract ALL person names and ALL organisation names from the text.

Rules:
- Extract FULL person names in any language and script (e.g. "John Smith", "Иванов Иван Иванович", "张伟", "محمد علي").
- Extract abbreviated person names too (e.g. "J. Smith", "Шатрова Ю.И.").
- Extract FULL organisation names including legal form in any language
  (e.g. "ООО «Ромашка»", "Acme Ltd", "GmbH Müller", "Société Générale S.A.").
- Do NOT extract generic role words like: party, parties, company, borrower, lender, lessor, lessee,
  contractor, client, agent, director, общество, стороны, компания, заёмщик, or similar terms.
- Do NOT extract positions, titles, addresses, postal codes, or dates.
- If a name appears in multiple grammatical forms (e.g. Russian declension), extract every form you see.
- Output ONLY valid JSON, no explanation text.

Output format:
{
  "persons": ["Full Name 1", "Full Name 2"],
  "organizations": ["Org Name 1", "Org Name 2"]
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


def replace_dates(text):
    """Replace all date expressions with [DATE1], [DATE2], ... sequentially."""
    date_map = {}  # normalized_match -> placeholder
    counter = [0]

    def _replace(m):
        token = m.group().strip()
        if token not in date_map:
            counter[0] += 1
            date_map[token] = f"[DATE{counter[0]}]"
        return date_map[token]

    for pattern in DATE_PATTERNS:
        text = re.sub(pattern, _replace, text, flags=re.IGNORECASE)
    return text


def replace_addresses(text):
    """Replace all address expressions with [ADDRESS1], [ADDRESS2], ... sequentially."""
    addr_map = {}
    counter = [0]

    def _replace(m):
        token = m.group().strip()
        if token not in addr_map:
            counter[0] += 1
            addr_map[token] = f"[ADDRESS{counter[0]}]"
        return addr_map[token]

    for pattern in ADDRESS_PATTERNS:
        text = re.sub(pattern, _replace, text)
    return text


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
        anon_text = replace_addresses(anon_text)
        anon_text = replace_dates(anon_text)
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
