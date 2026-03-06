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
# LOAD QWEN 3 14B
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
log("OpenPipe/Qwen3-14B-Instruct loaded")


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
    # English generic terms — roles, titles, legal terms
    'company', 'companies', 'the company', 'the companies',
    'party', 'parties', 'the party', 'the parties',
    'borrower', 'lender', 'lessor', 'lessee', 'tenant', 'landlord',
    'client', 'contractor', 'counterparty', 'agent', 'principal',
    'employer', 'employee', 'vendor', 'buyer', 'seller', 'purchaser',
    'assignor', 'assignee', 'guarantor', 'beneficiary',
    'director', 'directors', 'general director', 'ceo', 'shareholder', 'shareholders',
    'authorized person', 'representative', 'signatory',
    'group company', 'group companies', 'subsidiary', 'subsidiaries',
    # Titles & positions
    'chairman', 'secretary', 'treasurer', 'president', 'vice president',
    'manager', 'administrator', 'auditor', 'inspector', 'officer',
    'member', 'members', 'board', 'board of directors', 'committee',
    'trustee', 'trustees', 'receiver', 'liquidator', 'executor',
    'personal representative', 'personal representatives',
    'curator bonis', 'requisitionists', 'proxy',
    # Family / relationship words
    'parent', 'child', 'children', 'spouse', 'husband', 'wife',
    'brother', 'sister', 'mother', 'father', 'son', 'daughter',
    'widow', 'widower', 'deceased', 'heir', 'heirs',
    'remote issue', 'issue',
    # Government / institutional generic references
    'registrar of companies', 'ministry', 'court', 'tribunal',
    'government', 'state', 'republic', 'authority',
    'board of directors',
    # Countries and standalone geographic terms (not real entity names)
    'cyprus', 'belize', 'republic of cyprus', 'united kingdom',
    'united states', 'germany', 'france', 'spain', 'italy',
}


# ===============================
# REGEX PATTERNS FOR COMPANIES (Russian + International)
# ===============================
COMPANY_FULL_PATTERNS = [
    # ---- Russian legal forms ----
    r'(?i:ООО)\s*[«""][^»""]+[»""]',
    r'(?i:МКАО)\s*[«""][^»""]+[»""]',
    r'(?i:АО)\s*[«""][^»""]+[»""]',
    r'(?i:ПАО)\s*[«""][^»""]+[»""]',
    r'(?i:ЗАО)\s*[«""][^»""]+[»""]',
    r'(?i:ИП)\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.',
    r'(?i:ИП)\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+',
    r'(?i:Акционерное\s+общество)\s*[«""][^»""]+[»""]',
    r'(?i:Общество\s+с\s+ограниченной\s+ответственностью)\s*[«""][^»""]+[»""]',
    r'(?i:Публичное\s+акционерное\s+общество)\s*[«""][^»""]+[»""]',
    r'(?i:Международн\w+\s+компани\w+\s+акционерн\w+\s+общества?)\s*[«""][^»""]+[»""]',
    # ---- English / international legal forms ----
    r'[A-Z0-9][A-Za-z0-9&\'\-\.]+(?:\s+[A-Za-z0-9&\'\-\.]+)*\s+(?i:Ltd\.?|Limited|LLC|L\.L\.C\.?|LLP|L\.L\.P\.?|Inc\.?|Incorporated|Corp\.?|Corporation|PLC|P\.L\.C\.?|Public\s+Ltd\.?|Public\s+Limited)(?=\s|[,;\.]|$)',
    # ---- European legal forms ----
    r'[A-ZÀ-Ö0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*\s+(?:(?i:GmbH|AG|KG|OHG|GbR|S\.?A\.?R?\.?L?\.?|SAS|S\.?A\.?S\.?|S\.?L\.?|S\.?p\.?A\.?|S\.?r\.?l\.?|N\.?V\.?|B\.?V\.?|Pty\.?\s*Ltd\.?|Oy|ApS)|A\.?S\.?|A/S|AB)(?=\s|[,;\.]|$)',
    # ---- Greek legal forms ----
    r'[\u0391-\u03a9][\u0370-\u03ff]+(?:\s+[\u0370-\u03ff]+)*\s+(?i:ΛΤΔ\.?|ΕΠΕ\.?|Α\.?Ε\.?|ΔΗΜΟΣΙΑ\s+ΛΤΔ\.?)(?=\s|[,;\.]|$)',
    # ---- Quoted organisation names (any language) ----
    r'(?i:[ОООАОПАОЗАОМКАО]\w*)\s*[«"“][^»"”\n]{2,60}[»"”]',
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
    # Numeric with separators: DD.MM.YYYY / DD/MM/YYYY — must have 4-digit year
    r'\b\d{1,2}[.\/\-]\d{1,2}[.\/\-]\d{4}\b',
    # Numeric with 2-digit year: DD.MM.YY / DD/MM/YY
    r'\b\d{1,2}[.\/\-]\d{1,2}[.\/\-]\d{2}\b',
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
    # "02 JUL 2010" — DD MON YYYY (3-letter uppercase month)
    r'\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}\b',
    # "5 octobre 1961" (French months with day & year)
    r'\b\d{1,2}\s+' + _FR_MONTHS + r'\s+\d{4}\b',
]


# ===============================
# REGEX PATTERNS FOR ADDRESSES (Russian + International)
# ===============================
ADDRESS_PATTERNS = [
    # ---- Russian addresses ----
    r'\b\d{6},?\s*(?:[гГ](?:ород|\.)\s*[А-ЯЁа-яё\-]+,?\s*)?(?:[а-яёА-ЯЁ][а-яёА-ЯЁ\s\.\-]+,?\s*(?:[дД][.\/]?\s*\d+[а-яёА-ЯЁ]?(?:/\d+)?(?:,?\s*(?:[кК](?:орп)?|[сСтТ](?:тр)?)[.\/]?\s*\d+)?(?:,?\s*(?:[кКqQ]в|офис|оф\.?)\s*\.?\s*\d+)?)?)\b',
    r'(?:(?:[уУ]л(?:ица|\.)?|[пП]р(?:оспект|[\.\-])?|[пП]ер(?:еулок|\.)?|[шШ]оссе|[бБ]ульвар|[нН]аб(?:ережная|\.)?|[пП]лощадь|[пП]л\.?|[пП]роезд)\s+[А-ЯЁа-яё][А-ЯЁа-яё\s\-]+,?\s*(?:[дД][.\/]?\s*\d+[а-яё]?(?:/\d+)?)(?:,?\s*(?:[кК](?:орп)?|[сСтТ](?:тр)?)[.\/]?\s*\d+)?(?:,?\s*(?:[кКqQ]в|офис|оф\.?)\s*\.?\s*\d+)?)',
    r'\b[гГ]\.\s*[А-ЯЁ][а-яёА-ЯЁ\-]+(?:\s*,\s*[а-яёА-ЯЁ][а-яёА-ЯЁ\s\-]+(?:обл(?:асть|\.)?|кра[йя]|респ(?:ублик[аи]|\.)?|округ))?\b',
    # ---- English / international addresses (case-insensitive via re.IGNORECASE) ----
    # "123 Main Street, London, UK" / "191 ATHALASSIS AVE., P.O.Box 25525"
    r'\b\d+[A-Za-z]?(?:\s*[-/]\s*\d+[A-Za-z]?)?\s+[A-Za-z][A-Za-z\s\.\-]{2,40},?\s*\b(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Place|Pl\.?|Square|Sq\.?|Way|Crescent|Cres\.?|Close|Terrace|Ter\.?|Parkway|Pkwy\.?)\b(?:\s*[,\s]\s*(?:P\.O\.Box\s*\d+|[A-Za-z][A-Za-z\s\-\.]+)){0,4}(?:[\s\n]+TEL\.?\s*[\d\s\-\.]+(?:,?\s*FAX\.?\s*[\d\s\-\.]+)?)?',
    # UK postcode: "SW1A 2AA", "EC1A 1BB"
    r'\b[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2}\b',
    # US ZIP: "90210" or "90210-1234"
    r'\b\d{5}(?:-\d{4})?\b(?=[,\s])',
    # Generic: "City, State/Country" pattern
    r'\b[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?,?\s+(?:USA|United States|UK|United Kingdom|Germany|France|Spain|Italy|Netherlands|Switzerland|Austria|Belgium|Poland|Czech Republic|China|Japan|UAE|Canada|Australia)\b',
]


# ===============================
# ENTITY VALIDATION
# ===============================

# Sentence fragment indicators — if an entity contains these, it's not a real name
_FRAGMENT_INDICATORS = [
    'the company', 'of the', 'in the', 'by the', 'to the', 'for the',
    'at the', 'on the', 'from the', 'with the', 'and the', 'or the',
    'shall', 'may', 'must', 'will', 'would', 'should', 'could',
    'provided that', 'subject to', 'pursuant to', 'in accordance',
    'at all times', 'from time to time', 'at any time',
    'herein', 'hereof', 'hereto', 'hereby', 'hereunder',
    'aforesaid', 'aforementioned', 'notwithstanding',
    'including', 'excluding', 'except',
    '\n',  # newlines in entity = sentence fragment
]


def validate_entity(name):
    """Filter out generic terms, garbage, sentence fragments, and invalid entities."""
    name = name.strip()
    if len(name) < 3:
        return False
    if name.lower() in GENERIC_TERMS_LOWER:
        return False
    if ',' in name and '«' in name:
        return False
    if name.count('«') > 1:
        return False
    # Reject entries that are only brackets/dots/punctuation/placeholders
    cleaned = re.sub(r'[\[\]\.\.\(\)\s_\-]', '', name)
    if not cleaned or not any(c.isalpha() for c in cleaned):
        return False
    # Reject sentence fragments (contain legal boilerplate phrases)
    name_lower = name.lower()
    for indicator in _FRAGMENT_INDICATORS:
        if indicator in name_lower:
            return False
    # Reject single-word entries that are common English words (not proper names)
    # A real person name has at least 2 words; a single-word org must be distinctive
    words = name.split()
    if len(words) == 1:
        # Single word: only accept if it looks like an acronym (all caps, 2+ chars)
        if not (name.isupper() and len(name) >= 2):
            return False
    return True


def validate_date(date_str):
    """Filter out non-date strings that regex/LLM might have wrongly extracted."""
    date_str = date_str.strip()
    if len(date_str) < 4:
        return False
    # Reject duration phrases: "fourteen days", "six months", "eighteen months"
    duration_words = {
        'days', 'day', 'weeks', 'week', 'months', 'month', 'years', 'year',
        'hours', 'hour', 'minutes', 'minute',
        'дней', 'недель', 'месяцев', 'лет',
    }
    words = date_str.lower().split()
    if any(w in duration_words for w in words):
        return False
    # Reject bare 4-digit year without month context ("2014" alone)
    if re.fullmatch(r'\d{4}', date_str):
        return False
    # Reject bracket-only placeholders
    if re.fullmatch(r'[\[\]\.\s_]+', date_str):
        return False
    # Must contain at least one digit (a real date always has numbers)
    if not any(c.isdigit() for c in date_str):
        return False
    return True


def validate_address(addr_str):
    """Filter out non-address strings."""
    addr_str = addr_str.strip()
    if len(addr_str) < 5:
        return False
    # Must contain at least one letter
    if not any(c.isalpha() for c in addr_str):
        return False
    # Reject bracket-only placeholders
    if re.fullmatch(r'[\[\]\.\s_]+', addr_str):
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


# Maximum number of LLM NER chunks we allow (safeguard against runpod timeout).
# 80 chunks × ~4 500 tokens = ~360 000 tokens ≈ 700+ pages.
MAX_CHUNKS = 80


SYSTEM_PROMPT = """You are a multilingual named entity recognition (NER) assistant for legal and business documents.

The document may be in ANY language (Russian, English, German, French, Arabic, Chinese, etc.) or a mix of languages.

Extract ALL of the following from the text:
1. Person names (actual human names only)
2. Organisation / company names (actual registered business names only)
3. Dates (specific calendar dates only)
4. Addresses (physical street/postal addresses only)

CRITICAL RULES — what to extract:
- PERSONS: Only real human names, like "John Smith", "Иванов Иван Иванович", "Andreas Menelaou"
  • Extract person names EVEN when they appear in an official capacity (e.g. "Irene Athanasiadou for Registrar of Companies" → extract "Irene Athanasiadou")
  • Extract person names that appear as witnesses, signatories, advocates, translators, notaries
- ORGANISATIONS: Only actual named companies/firms, like "UFG Capital Investment Management Ltd", "ООО «Ромашка»"
- DATES: Only specific calendar dates, like "02.01.1983", "October 31, 2024", "5 octobre 1961", "02 JUL 2010"
- ADDRESSES: Only physical addresses, like "24A, Parnithos street, Strovolos, 2007, Nicosia, Cyprus"

CRITICAL RULES — what NOT to extract:
- Do NOT extract role titles as persons: Chairman, Director, Secretary, Administrator, Treasurer, Trustee, Receiver, Liquidator, Proxy, Committee
- Do NOT extract family words as persons: Parent, Child, Brother, Sister, Spouse, Husband, Wife, Widow, Widower, Deceased, Heir
- Do NOT extract generic legal terms as organisations: "the Company", "Board of Directors", "Members of the Company", "Registrar of Companies", "General Meeting"
- Do NOT extract sentence fragments containing "the Company" or similar — only extract the actual registered name
- Do NOT extract countries or cities alone as organisations: "Cyprus", "Belize", "Republic of Cyprus"
- Do NOT extract government bodies as organisations: "Ministry of Justice", "Registrar of Companies"
- Do NOT extract time durations as dates: "fourteen days", "six months", "eighteen months"
- Do NOT extract bare years as dates: "2014" alone is NOT a date
- Do NOT extract template placeholders: "[........]" is NOT an entity

Output ONLY valid JSON, no explanation text.

Output format:
{
  "persons": ["Andreas Menelaou", "Louiza Georgiou"],
  "organizations": ["UFG Capital Investment Management Ltd"],
  "dates": ["02.01.1983", "October 31, 2024"],
  "addresses": ["24A, Parnithos street, Strovolos, 2007, Nicosia, Cyprus"]
}"""


def extract_entities_llm(text_chunk):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract all persons, organizations, dates, and addresses:\n\n{text_chunk}"}
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
    log(f"LLM raw: {raw_output[:500]}")

    try:
        # Try to find JSON block in the output
        json_match = re.search(r'\{[^{}]*"persons"\s*:.*\}', raw_output, re.DOTALL)
        result = json.loads(json_match.group()) if json_match else json.loads(raw_output)
        persons = [p.strip() for p in result.get("persons", []) if p and p.strip()]
        organizations = [o.strip() for o in result.get("organizations", []) if o and o.strip()]
        dates = [d.strip() for d in result.get("dates", []) if d and d.strip()]
        addresses = [a.strip() for a in result.get("addresses", []) if a and a.strip()]
        return persons, organizations, dates, addresses
    except (json.JSONDecodeError, AttributeError) as e:
        log(f"Parse failed: {e}")
        return [], [], [], []


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
        # Normalize: strip trailing dots/whitespace for dedup comparison
        norm = n.rstrip('. ').lower()
        if n and norm not in seen_o and validate_entity(n):
            seen_o.add(norm)
            # If we already have a shorter variant, replace with the longer one
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


def detect_dates_regex(text):
    """Extract all date strings from text using regex patterns."""
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
    """Extract all address strings from text using regex patterns."""
    addresses = []
    seen = set()
    for pattern in ADDRESS_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            token = m.group().strip()
            if token and token.lower() not in seen:
                seen.add(token.lower())
                addresses.append(token)
    return addresses


def _flexible_pattern(text_str):
    """Build a regex pattern from a string where spaces match any whitespace (incl. newlines).
    This handles entries that appear split across lines in the document."""
    escaped = re.escape(text_str)
    # Replace escaped spaces with \s+ so "Strovolos, 2049" matches "Strovolos,\n2049"
    return escaped.replace(r'\ ', r'\s+')


def replace_dates(text, date_map):
    """Replace dates in text using the pre-built date_map.
    Replaces longest entries first to prevent partial matches.
    Whitespace-flexible and case-insensitive."""
    for date_str in sorted(date_map.keys(), key=len, reverse=True):
        pattern = _flexible_pattern(date_str)
        text = re.sub(pattern, date_map[date_str], text, flags=re.IGNORECASE)
    return text


def replace_addresses(text, addr_map):
    """Replace addresses in text using the pre-built addr_map.
    Replaces longest entries first to prevent partial matches.
    Whitespace-flexible to handle multi-line addresses."""
    for addr_str in sorted(addr_map.keys(), key=len, reverse=True):
        pattern = _flexible_pattern(addr_str)
        text = re.sub(pattern, addr_map[addr_str], text)
    return text


def safe_replace(text, mapping, name_patterns=None):
    """Replace entities with placeholders using case-insensitive matching
    and Cyrillic-aware word boundaries."""
    sorted_entities = sorted(mapping.keys(), key=len, reverse=True)

    for entity in sorted_entities:
        placeholder = mapping[entity]
        escaped = re.escape(entity)
        pattern = r'(?<![\u0410-\u042f\u0401\u0430-\u044f\u0451A-Za-z])' + escaped + r'(?![\u0410-\u042f\u0401\u0430-\u044f\u0451A-Za-z])'
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

    if name_patterns:
        for compiled_re, placeholder in name_patterns:
            text = compiled_re.sub(placeholder, text)

    return text


def dedup_substrings(items):
    """Remove items that are substrings of a longer item in the list.
    E.g. ['octobre 1961', '5 octobre 1961'] -> ['5 octobre 1961']
    """
    if not items:
        return items
    # Sort longest first so we check shorter items against longer ones
    sorted_items = sorted(items, key=len, reverse=True)
    result = []
    for item in sorted_items:
        item_lower = item.lower()
        # Check if this item is a substring of any already-accepted (longer) item
        is_sub = False
        for accepted in result:
            if item_lower in accepted.lower():
                is_sub = True
                break
        if not is_sub:
            result.append(item)
    return result


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
                f"This corresponds to roughly {MAX_CHUNKS * 4500 // 500} pages. "
                "Please split the document and process it in parts."
            )
        }

    all_persons, all_orgs, all_dates, all_addresses = [], [], [], []
    ner_start = time.time()
    for i, chunk in enumerate(chunks):
        t = time.time()
        log(f"Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        persons, orgs, dates, addresses = extract_entities_llm(chunk)
        all_persons.extend(persons)
        all_orgs.extend(orgs)
        all_dates.extend(dates)
        all_addresses.extend(addresses)
        log(f"Chunk {i+1} done in {time.time()-t:.1f}s — {len(persons)}p {len(orgs)}o {len(dates)}d {len(addresses)}a")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    log(f"NER done in {time.time()-ner_start:.1f}s — {len(all_persons)}p {len(all_orgs)}o {len(all_dates)}d {len(all_addresses)}a (raw)")

    # ---- Regex fallback: catch anything the LLM missed ----
    regex_orgs = detect_companies_regex(full_text)
    regex_dates = detect_dates_regex(full_text)
    regex_addresses = detect_addresses_regex(full_text)

    persons, organizations = merge_entities(all_persons, all_orgs, regex_orgs)

    # Deduplicate dates (LLM + regex) with validation
    seen_d = set()
    unique_dates = []
    for d in all_dates + regex_dates:
        d = d.strip()
        if d and d.lower() not in seen_d and validate_date(d):
            seen_d.add(d.lower())
            unique_dates.append(d)
    # Remove partial dates that are substrings of longer dates
    unique_dates = dedup_substrings(unique_dates)
    log(f"Dates: {len(unique_dates)} unique (LLM {len(all_dates)} + regex {len(regex_dates)})")

    # Deduplicate addresses (LLM + regex) with validation
    seen_a = set()
    unique_addresses = []
    for a in all_addresses + regex_addresses:
        a = a.strip()
        if a and a.lower() not in seen_a and validate_address(a):
            seen_a.add(a.lower())
            unique_addresses.append(a)
    # Remove partial addresses that are substrings of longer addresses
    unique_addresses = dedup_substrings(unique_addresses)
    log(f"Addresses: {len(unique_addresses)} unique (LLM {len(all_addresses)} + regex {len(regex_addresses)})")

    # Group variants for persons/orgs
    person_groups = group_person_variants(persons)
    org_groups = group_org_variants(organizations)
    log(f"Grouped: {len(person_groups)} person groups, {len(org_groups)} org groups")

    mapping = build_ordered_mapping(full_text, person_groups, org_groups)

    # ---- Build global date & address mappings (numbered by first appearance) ----
    def find_first_pos(token):
        pos = full_text.find(token)
        if pos == -1:
            pos = full_text.lower().find(token.lower())
        return pos if pos != -1 else float('inf')

    # Sort dates/addresses by first appearance in text, assign sequential IDs
    date_map = {}
    sorted_dates = sorted(unique_dates, key=find_first_pos)
    for i, d in enumerate(sorted_dates, 1):
        date_map[d] = f"[DATE{i}]"

    addr_map = {}
    sorted_addrs = sorted(unique_addresses, key=find_first_pos)
    for i, a in enumerate(sorted_addrs, 1):
        addr_map[a] = f"[ADDRESS{i}]"

    log(f"Date mapping: {len(date_map)} entries")
    for orig, ph in date_map.items():
        log(f"  {ph} <- {orig}")
    log(f"Address mapping: {len(addr_map)} entries")
    for orig, ph in addr_map.items():
        log(f"  {ph} <- {orig}")


    # ---- Build declension patterns for persons ----
    name_patterns = build_person_patterns(person_groups, mapping) if mapping else []
    log(f"Built {len(name_patterns)} declension patterns")

    # ---- Replace all entities page by page ----
    anonymized_pages = []
    for page in sorted(pages, key=lambda p: p["page"]):
        anon_text = page["text"]
        # Replace persons & orgs (LLM-detected)
        if mapping:
            anon_text = safe_replace(anon_text, mapping, name_patterns)
        # Replace addresses (longest-first from pre-built map)
        anon_text = replace_addresses(anon_text, addr_map)
        # Replace dates (longest-first from pre-built map)
        anon_text = replace_dates(anon_text, date_map)
        anonymized_pages.append({"page": page["page"], "text": anon_text})

    # ---- Build clean display mapping ----
    display_mapping = {}
    # Persons & organisations
    for canonical, variants in org_groups:
        ph = mapping.get(canonical) or mapping.get(variants[0])
        if ph:
            display_mapping[canonical] = ph
    for canonical, variants in person_groups:
        ph = mapping.get(canonical) or mapping.get(variants[0])
        if ph:
            display_mapping[canonical] = ph
    # Addresses & dates
    display_mapping.update(addr_map)
    display_mapping.update(date_map)

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
