"""Shared Persian text, structural chunking, and sparse BM25 helpers."""

from collections import Counter
from functools import lru_cache
import math
import re

try:
    from hazm import Normalizer, SentenceTokenizer, WordTokenizer
except ImportError:  # The application still starts before optional install.
    Normalizer = SentenceTokenizer = WordTokenizer = None


_normalizer = Normalizer() if Normalizer else None
_sentence_tokenizer = SentenceTokenizer() if SentenceTokenizer else None
_word_tokenizer = WordTokenizer() if WordTokenizer else None
_PERSIAN_WORD_PATTERN = r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff\u200c]+"


def normalize_persian(text: str) -> str:
    text = text or ""

    if _normalizer:
        text = _normalizer.normalize(text)

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "هٔ",
        "ة": "ه",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    text = normalize_persian(text).lower()
    if _word_tokenizer:
        words = _word_tokenizer.tokenize(text)
    else:
        words = re.findall(r"[\w\u200c]+", text, flags=re.UNICODE)
    return [word for word in words if re.search(r"[\w\u0600-\u06ff]", word)]


def _edit_distance(left: str, right: str, limit: int) -> int:
    """Bounded Damerau-Levenshtein distance for short Persian words."""
    if abs(len(left) - len(right)) > limit:
        return limit + 1

    previous_previous = None
    previous = list(range(len(right) + 1))

    for row, left_char in enumerate(left, start=1):
        current = [row]
        row_minimum = row

        for column, right_char in enumerate(right, start=1):
            value = min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            )
            if (
                previous_previous is not None
                and row > 1
                and column > 1
                and left_char == right[column - 2]
                and left[row - 2] == right_char
            ):
                value = min(value, previous_previous[column - 2] + 1)

            current.append(value)
            row_minimum = min(row_minimum, value)

        if row_minimum > limit:
            return limit + 1
        previous_previous, previous = previous, current

    return previous[-1]


def build_query_corrector(bm25: dict | None):
    """Return a cached, conservative typo corrector built from corpus terms."""
    if not bm25:
        return normalize_persian

    vocabulary = set(bm25.get("vocabulary", {}))
    by_length: dict[int, tuple[str, ...]] = {}
    for length in {len(word) for word in vocabulary}:
        by_length[length] = tuple(
            word
            for word in vocabulary
            if len(word) == length and re.fullmatch(_PERSIAN_WORD_PATTERN, word)
        )

    @lru_cache(maxsize=4096)
    def correct_word(word: str) -> str:
        if word in vocabulary or len(word) < 4:
            return word

        # Do not "correct" a valid inflected form merely because that exact
        # form did not occur in the small document collection.
        suffixes = (
            "هایی", "های", "ترین", "تر", "ی",
            "مان", "تان", "شان", "ام", "ات", "اش",
        )
        if any(
            len(word) > len(suffix) + 2
            and word[:-len(suffix)] in vocabulary
            for suffix in suffixes
            if word.endswith(suffix)
        ):
            return word

        limit = 1 if len(word) < 7 else 2
        candidates = []
        candidate_lengths = (
            (len(word),)
            if len(word) < 7
            else range(len(word) - limit, len(word) + limit + 1)
        )
        for length in candidate_lengths:
            for candidate in by_length.get(length, ()):
                distance = _edit_distance(word, candidate, limit)
                if distance <= limit:
                    candidates.append((distance, candidate))

        if not candidates:
            return word

        candidates.sort()
        best_distance = candidates[0][0]
        best = [candidate for distance, candidate in candidates if distance == best_distance]

        # Ambiguous corrections are more dangerous than an unchanged typo.
        return best[0] if len(best) == 1 else word

    def correct_query(text: str) -> str:
        text = normalize_persian(text)
        return re.sub(
            _PERSIAN_WORD_PATTERN,
            lambda match: correct_word(match.group(0).lower()),
            text,
        )

    return correct_query


def _sentences(text: str) -> list[str]:
    if _sentence_tokenizer:
        return _sentence_tokenizer.tokenize(text)
    return [
        part.strip()
        for part in re.split(r"(?<=[.!؟!])\s+|\n+", text)
        if part.strip()
    ]


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if line.endswith((".", "؟", "!", "،", ";", "؛")):
        return False
    words = line.split()
    numbered = bool(
        re.match(r"^\s*\d+(?:\s*-\s*\d+)*\s*-", line)
    )
    known_heading = line in {
        "چکیده",
        "مقدمه",
        "نتیجه گیری",
        "نتیجه‌گیری",
        "منابع",
        "فهرست مطالب",
    }
    return numbered or known_heading or 1 <= len(words) <= 4


def structural_chunks(
    text: str,
    target_chars: int = 850,
    overlap_sentences: int = 2,
) -> list[dict]:
    """Split at headings/paragraphs/sentences and retain section context."""
    text = normalize_persian(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []

    for line in lines:
        if _looks_like_heading(line):
            if body:
                sections.append((heading, " ".join(body)))
                body = []
            heading = line
        else:
            body.append(line)
    if body or heading:
        sections.append((heading, " ".join(body)))

    chunks: list[dict] = []
    for heading, body_text in sections:
        sentences = _sentences(body_text or heading)
        current: list[str] = []

        for sentence in sentences:
            candidate = " ".join(current + [sentence])
            prefix_size = len(heading) + 2 if heading else 0
            if current and len(candidate) + prefix_size > target_chars:
                chunk_body = " ".join(current).strip()
                chunk_text = f"عنوان بخش: {heading}\n{chunk_body}" if heading else chunk_body
                chunks.append({"text": chunk_text, "section": heading})
                current = current[-overlap_sentences:] if overlap_sentences else []
            current.append(sentence)

        if current:
            chunk_body = " ".join(current).strip()
            chunk_text = f"عنوان بخش: {heading}\n{chunk_body}" if heading else chunk_body
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "section": heading})

    return chunks


def build_bm25(chunks: list[str]) -> dict:
    tokenized = [tokenize_words(text) for text in chunks]
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    vocabulary = {
        token: index
        for index, token in enumerate(sorted(document_frequency))
    }
    count = len(tokenized)
    idf = {
        token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
    }
    average_length = sum(map(len, tokenized)) / max(count, 1)
    return {
        "vocabulary": vocabulary,
        "idf": idf,
        "average_length": average_length,
        "document_count": count,
    }


def sparse_document_vector(text: str, bm25: dict) -> tuple[list[int], list[float]]:
    tokens = tokenize_words(text)
    frequencies = Counter(tokens)
    length = len(tokens)
    average_length = bm25["average_length"] or 1.0
    k1, b = 1.5, 0.75
    weighted = []

    for token, frequency in frequencies.items():
        if token not in bm25["vocabulary"]:
            continue
        denominator = frequency + k1 * (1 - b + b * length / average_length)
        value = bm25["idf"][token] * frequency * (k1 + 1) / denominator
        weighted.append((bm25["vocabulary"][token], value))

    weighted.sort()
    return [item[0] for item in weighted], [item[1] for item in weighted]


def sparse_query_vector(text: str, bm25: dict) -> tuple[list[int], list[float]]:
    frequencies = Counter(tokenize_words(text))
    weighted = [
        (
            bm25["vocabulary"][token],
            bm25["idf"][token] * (1 + math.log(frequency)),
        )
        for token, frequency in frequencies.items()
        if token in bm25["vocabulary"]
    ]
    weighted.sort()
    return [item[0] for item in weighted], [item[1] for item in weighted]
