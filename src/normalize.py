"""Text normalisation for business names and addresses.

Everything here is hand-written rules + small dictionaries of common
abbreviations; no external data or services are used.
"""
import re
import unicodedata

# ---------------------------------------------------------------- Devanagari
_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
    "ळ": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "rh", "फ़": "f",
}
_INDEP = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
_MATRA = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e",
    "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}
_OTHER = {"ं": "n", "ँ": "n", "ः": "h", "़": "", "्": "", "।": " ", "॥": " "}
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
_DEVA_RE = re.compile("[\u0900-\u097F]")


def has_devanagari(s: str) -> bool:
    return bool(_DEVA_RE.search(s))


def translit_devanagari(s: str) -> str:
    """Rough Hindi -> Latin transliteration with word-final schwa deletion."""
    if not _DEVA_RE.search(s):
        return s
    s = unicodedata.normalize("NFC", s)
    out = []
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if i + 1 < n and s[i + 1] == "़" and ch + "़" in _CONS:
            ch = ch + "़"
            i += 1
        if ch in _CONS:
            out.append(_CONS[ch])
            nxt = s[i + 1] if i + 1 < n else " "
            # inherent vowel unless followed by matra / virama / end of word
            if nxt not in _MATRA and nxt != "्" and _DEVA_RE.match(nxt):
                out.append("a")
        elif ch in _MATRA:
            out.append(_MATRA[ch])
        elif ch in _INDEP:
            out.append(_INDEP[ch])
        elif ch in _OTHER:
            out.append(_OTHER[ch])
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------- generic
_NULLS = {"null", "none", "nan", "na", "n/a", "<null>"}


def to_ascii(s: str) -> str:
    s = translit_devanagari(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

LEGAL = {
    # English / India
    "ltd", "limited", "pvt", "private", "llc", "l", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llp", "lp", "pllc", "pc", "plc", "public",
    "the", "and", "of", "m", "s", "ms", "pra", "li", "pa", "dba", "formerly",
    "prop", "proprietor", "praivet", "privet", "elaelapi", "limitid", "lemited",
    # French
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scp", "selarl", "et", "de",
    "du", "des", "la", "le", "les",
}
# generic business words: kept for full-name similarity, down-weighted in keys
GENERIC = {
    "services", "service", "group", "center", "centre", "holdings", "enterprises",
    "enterprise", "solutions", "india", "international", "global", "industries",
    "associates", "partners", "ventures", "trading", "traders", "company",
    "technologies", "technology", "tech", "consultants", "consultancy", "overseas",
    "exports", "shri", "sri", "smt", "mr", "dr", "and", "systems", "store",
}


def name_tokens(name: str):
    s = to_ascii(name).replace("&", " and ")
    s = re.sub(r"\.(com|net|org|in|co|fr|us|io|biz)\b", " ", s)
    toks = []
    for t in _TOKEN_SPLIT.split(s):
        if not t:
            continue
        if t.isdigit():
            continue  # '#75357' style noise
        if any(c.isdigit() for c in t):
            t = t.translate(_LEET)
        toks.append(t)
    return toks


def clean_name(name: str) -> str:
    return " ".join(name_tokens(name))


def core_name(name: str) -> str:
    toks = [t for t in name_tokens(name) if t not in LEGAL]
    return " ".join(toks)


# ------------------------------------------------------------ address
_ADDR_ABBR = {
    "st": "street", "str": "street", "saint": "street", "rd": "road", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place",
    "hwy": "highway", "pkwy": "parkway", "trl": "trail", "sq": "square",
    "ter": "terrace", "cres": "crescent", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor",
    "no": "", "number": "", "h": "house", "hno": "house", "opp": "opposite",
    "nr": "near", "r": "rue", "imp": "impasse", "all": "allee", "ch": "chemin",
    "rte": "route", "fg": "faubourg", "unit": "", "po": "po", "box": "box",
    "marg": "marg", "clony": "colony",
}
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "newhampshire",
    "nj": "newjersey", "nm": "newmexico", "ny": "newyork", "nc": "northcarolina",
    "nd": "northdakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhodeisland", "sc": "southcarolina", "sd": "southdakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "westvirginia", "wi": "wisconsin", "wy": "wyoming", "dc": "districtofcolumbia",
}
IN_STATES = {
    "ap": "andhrapradesh", "ar": "arunachalpradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachalpradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhyapradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya",
    "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha", "orissa": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamilnadu", "tg": "telangana",
    "ts": "telangana", "tr": "tripura", "up": "uttarpradesh", "uk": "uttarakhand",
    "ut": "uttarakhand", "wb": "westbengal", "dl": "delhi", "jk": "jammukashmir",
    "ch": "chandigarh", "py": "puducherry", "la": "ladakh",
}
_MULTI_WORD = {}
for _d in (US_STATES, IN_STATES):
    for _full in _d.values():
        _MULTI_WORD[_full] = _full
_TWO_WORD_RE = re.compile(
    r"\b(new|north|south|west|rhode|district|andhra|arunachal|himachal|madhya|tamil|uttar|"
    r"jammu|of)\s+(hampshire|jersey|mexico|york|carolina|dakota|virginia|island|of|columbia|"
    r"pradesh|nadu|bengal|kashmir|and)\b"
)
_NUM_ORD = re.compile(r"^(\d+)(st|nd|rd|th)$")


def address_tokens(addr: str, country: str = ""):
    """Returns (word_tokens, number_tokens)."""
    s = to_ascii(addr)
    for _ in range(2):
        s = _TWO_WORD_RE.sub(lambda m: m.group(1) + m.group(2), s)
    s = s.replace("jammuandkashmir", "jammukashmir").replace("districtofcolumbia", "districtofcolumbia")
    words, nums = [], []
    states = US_STATES if country == "US" else IN_STATES if country == "India" else {}
    for comp in s.split(","):
        raw = [t for t in _TOKEN_SPLIT.split(comp) if t and t not in _NULLS]
        if len(raw) == 1 and raw[0] in states:
            words.append(states[raw[0]])
            continue
        for t in raw:
            m = _NUM_ORD.match(t)
            if m:
                t = m.group(1)
            if t.isdigit():
                nums.append(t.lstrip("0") or "0")
                continue
            if any(c.isdigit() for c in t):
                d = re.sub(r"\D", "", t).lstrip("0")
                if d:
                    nums.append(d)
                t = re.sub(r"\d", "", t)
                if not t:
                    continue
            t = _ADDR_ABBR.get(t, t)
            if t:
                words.append(t)
    return words, nums


# ------------------------------------------------------------ skeleton
_SKEL_SUBS = [("ph", "f"), ("sh", "s"), ("chh", "k"), ("ch", "k"), ("kh", "k"),
              ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("jh", "j"),
              ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "j"),
              ("w", "v"), ("m", "n")]
_VOWELS_RE = re.compile(r"[aeiouyh]")
_REPEAT_RE = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    """Consonant skeleton: robust to vowel typos & Hindi transliteration."""
    if not tok:
        return ""
    t = tok
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    first = t[0]
    rest = _VOWELS_RE.sub("", t[1:])
    t = first + rest
    t = _REPEAT_RE.sub(r"\1", t)
    if len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    return t
