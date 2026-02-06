"""Text processing utilities ported from the existing Chatterbox Gradio app.

Only used when a model's config has needs_text_preprocessing: true.
Chunking is shared across all models.
"""

import re


# ── Number-to-words ──────────────────────────────────────────────

ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = [
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + number_to_words(-n)
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    if n < 100:
        return TENS[n // 10] + (" " + ONES[n % 10] if n % 10 != 0 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (
            " and " + number_to_words(n % 100) if n % 100 != 0 else ""
        )
    return str(n)


def normalize_numbers(text: str) -> str:
    return re.sub(r'\b\d+\b', lambda m: number_to_words(int(m.group())), text)


# ── Abbreviation expansion ──────────────────────────────────────

ABBREVIATIONS = {
    "mr": "mister", "mrs": "misses", "ms": "miss", "dr": "doctor",
    "prof": "professor", "sr": "senior", "jr": "junior",
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "apt": "apartment", "dept": "department", "etc": "et cetera",
    "vs": "versus", "approx": "approximately", "esp": "especially",
    "inc": "incorporated", "ltd": "limited", "co": "company",
}


def expand_abbreviations(text: str) -> str:
    words = text.split()
    expanded = []
    for word in words:
        base_word = word.rstrip(".,!?;:")
        punct = word[len(base_word):]
        lower_word = base_word.lower()
        if lower_word in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[lower_word] + punct)
        else:
            expanded.append(word)
    return " ".join(expanded)


# ── Text cleaning ────────────────────────────────────────────────

def clean_text_fn(
    text: str,
    clean_whitespace: bool = True,
    move_punctuation: bool = True,
    replace_dashes: bool = True,
    normalize_nums: bool = False,
    expand_abbrev: bool = False,
) -> str:
    if not text:
        return text
    if normalize_nums:
        text = normalize_numbers(text)
    if expand_abbrev:
        text = expand_abbreviations(text)
    if replace_dashes:
        text = text.replace("\u2014", ", ").replace("\u2013", ", ").replace("--", ", ")
        text = re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', ' ', text)
    if move_punctuation:
        text = re.sub(r'(["\'])([.,!?;])', r'\2\1', text)
    if clean_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.,!?;:])(?=\S)', r'\1 ', text)
    return text


# ── Chunking ─────────────────────────────────────────────────────

def split_text_into_chunks(text: str, strategy: str = "Sentence Batching (<300 chars)") -> list[str]:
    if strategy == "No Split":
        return [text]
    if strategy == "Paragraph Split":
        chunks = [c.strip() for c in re.split(r'\n\n+', text) if c.strip()]
        return chunks if chunks else [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if strategy == "Sentence Split":
        return sentences

    # Sentence batching
    max_chars = 300 if "300" in strategy else 500
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# ── Pause tag parsing ────────────────────────────────────────────

def parse_pause_tags(text: str) -> list[dict]:
    """Parse [pause:Xs] tags into text and pause segments.

    Returns a list of dicts:
      {"type": "text", "content": "..."}
      {"type": "pause", "duration": 1.5}
    """
    pause_pattern = r'\[pause:([\d.]+)s\]'
    segments = []
    last_end = 0
    for match in re.finditer(pause_pattern, text):
        pre_text = text[last_end:match.start()].strip()
        if pre_text:
            segments.append({"type": "text", "content": pre_text})
        segments.append({"type": "pause", "duration": float(match.group(1))})
        last_end = match.end()
    remaining = text[last_end:].strip()
    if remaining:
        segments.append({"type": "text", "content": remaining})
    return segments
