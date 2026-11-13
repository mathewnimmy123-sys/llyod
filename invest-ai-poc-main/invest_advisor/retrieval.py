"""Small, dependency-free retriever over the policy document (pure Python, testable).

Why not the original whole-section word-overlap?  It split on blank lines, did not strip
punctuation ('pension,' != 'pension'), always returned 2 sections even for irrelevant
queries, and cited only a file name. This version:
  * chunks at bullet level (each rule is a citable unit, e.g. S1-B2),
  * ranks with BM25 (+ light stemming and a tiny synonym map),
  * abstains ("no_match") below a score threshold so the agent says "I don't know"
    instead of improvising.
Swap `Retriever.search` for Vertex AI Search / RAG Engine in production; the return shape
is the contract the agents rely on.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_STOP = {
    "a", "an", "the", "is", "are", "am", "was", "be", "to", "of", "in", "on", "for", "and",
    "or", "my", "me", "i", "we", "you", "your", "it", "this", "that", "do", "does", "can",
    "should", "would", "could", "what", "which", "how", "much", "many", "with", "about",
    "if", "at", "by", "as", "from", "will", "get", "there", "any", "so", "than", "then",
}
# Query words that don't literally appear in the document, mapped to words that do.
_SYNONYMS = {
    "pay": ["contribution"], "put": ["contribution"], "save": ["contribution"],
    "invest": ["contribution"], "cap": ["limit", "allowance"], "max": ["limit"],
    "maximum": ["limit"], "limit": ["allowance"], "relief": ["relief"],
    "bonus": ["bonus"], "young": ["lifetime"], "house": ["lifetime"],
}


def _stem(tok: str) -> str:
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("xes", "x")):
        if tok.endswith(suf) and len(tok) > len(suf) + 1:
            return tok[: -len(suf)] + rep
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def tokenize(text: str, expand: bool = False) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())  # 60,000 -> 60000
    toks = [_stem(t) for t in re.findall(r"[a-z0-9£%]+", text) if t not in _STOP]
    if expand:
        for t in list(toks):
            toks.extend(_SYNONYMS.get(t, []))
    return toks


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section_id: str
    section_title: str
    label: str
    text: str

    @property
    def index_text(self) -> str:
        return f"{self.section_title} {self.label} {self.text}"


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    sec_no, sec_title, bullet_no = 0, "", 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("DOCUMENT:"):
            continue
        m = re.match(r"SECTION\s+(\d+):\s*(.+)", line, re.I)
        if m:
            sec_no, sec_title, bullet_no = int(m.group(1)), m.group(2).strip(), 0
            continue
        if line.startswith("-"):
            bullet_no += 1
            body = line.lstrip("- ").strip()
            label, _, rest = body.partition(":")
            if not rest:  # bullet without "Label:" prefix
                label, rest = "", body
            chunks.append(
                Chunk(f"S{sec_no}-B{bullet_no}", f"S{sec_no}", sec_title, label.strip(), rest.strip())
            )
        elif chunks:  # wrapped continuation line
            last = chunks.pop()
            chunks.append(Chunk(last.chunk_id, last.section_id, last.section_title, last.label,
                                f"{last.text} {line}"))
    return chunks


class Retriever:
    K1, B = 1.5, 0.75

    def __init__(self, chunks: list[Chunk], source_name: str, min_score: float = 1.0):
        self.chunks, self.source_name, self.min_score = chunks, source_name, min_score
        self._docs = [tokenize(c.index_text) for c in chunks]
        self._avgdl = (sum(map(len, self._docs)) / len(self._docs)) if self._docs else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, q_terms: list[str], doc: list[str]) -> float:
        tf, dl, s = Counter(doc), len(doc), 0.0
        for t in set(q_terms):
            if t in tf:
                f = tf[t]
                s += self._idf[t] * f * (self.K1 + 1) / (f + self.K1 * (1 - self.B + self.B * dl / self._avgdl))
        return s

    def search(self, query: str, top_k: int = 3) -> dict:
        q_terms = tokenize(query, expand=True)
        scored = sorted(
            ((self._score(q_terms, d), c) for d, c in zip(self._docs, self.chunks)),
            key=lambda x: x[0], reverse=True,
        )
        hits = [(s, c) for s, c in scored[:top_k] if s >= self.min_score]
        if not hits:
            return {
                "status": "no_match",
                "source_document": self.source_name,
                "message": "No relevant passage found in the policy document. Do not answer "
                           "from general knowledge; say this is outside the documented scope.",
                "passages": [],
            }
        return {
            "status": "success",
            "source_document": self.source_name,
            "passages": [
                {"chunk_id": c.chunk_id, "section": f"{c.section_id}: {c.section_title}",
                 "text": f"{c.label}: {c.text}" if c.label else c.text, "score": round(s, 2)}
                for s, c in hits
            ],
        }
