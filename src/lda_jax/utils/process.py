from __future__ import annotations

"""jax_topics.data_tools
=================================

A small self‑contained helper module with responsibility:

1. **Corpus Pre‑processing** – convert raw strings into the `(word_ids,
   doc_ids)` representation required by our JAX Gibbs kernels.

The code is designed to be jit‑friendly
"""
from typing import List, Sequence, NamedTuple, Optional
import itertools
from collections import Counter

import numpy as np
import jax.numpy as jnp

################################################################################
# 1. Corpus container ###########################################################
################################################################################

class Corpus(NamedTuple):
    """Minimal container for a bag‑of‑words corpus."""

    word_ids: jnp.ndarray              # shape (N,)
    doc_ids:  jnp.ndarray              # shape (N,)
    vocab:    List[str]
    doc_ptrs: jnp.ndarray              # shape (D + 1,)

    # Convenience helpers ----------------------------------------------------
    @property
    def num_tokens(self) -> int:  # noqa: D401 – short docstring is fine here
        """Total tokens N."""
        return int(self.word_ids.size)

    @property
    def num_docs(self) -> int:  # noqa: D401
        """Number of documents D."""
        return int(self.doc_ptrs.size - 1)

    @property
    def vocab_size(self) -> int:  # noqa: D401
        """Vocabulary size V."""
        return len(self.vocab)

################################################################################
# 2. Pre‑processing utilities ###################################################
################################################################################

def _tokenize_texts(
    texts: Sequence[str],
    *,
    lowercase: bool = True,
    allowed_pos: Optional[Sequence[str]] = None,
) -> List[List[str]]:
    """Light‑weight tokeniser using a rules‑only spaCy pipeline."""
    import spacy  # Local import keeps spaCy optional until actually used.

    nlp = spacy.blank("en")
    out: List[List[str]] = []
    for doc in nlp.pipe(texts, disable=["parser", "ner", "tagger"]):
        toks = [
            t.text.lower() if lowercase else t.text
            for t in doc
            if t.is_alpha and not t.is_stop and (allowed_pos is None or t.pos_ in allowed_pos)
        ]
        out.append(toks)
    return out


def build_vocab(
    token_lists: Sequence[Sequence[str]],
    *,
    min_count: int = 5,
    max_vocab: int = 50_000,
) -> List[str]:
    """Frequency‑filtered vocabulary sorted by descending count."""
    counts = Counter(itertools.chain.from_iterable(token_lists))
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort(key=lambda w: (-counts[w], w))
    return vocab[:max_vocab]


def corpus_from_texts(
    texts: Sequence[str],
    *,
    min_count: int = 5,
    max_vocab: int = 50_000,
    lowercase: bool = True,
    allowed_pos: Optional[Sequence[str]] = None,
) -> Corpus:
    """Convert raw text documents into a :class:`Corpus`."""
    token_lists = _tokenize_texts(texts, lowercase=lowercase, allowed_pos=allowed_pos)

    vocab = build_vocab(token_lists, min_count=min_count, max_vocab=max_vocab)
    word2id = {w: i for i, w in enumerate(vocab)}

    # Map tokens to ids -------------------------------------------------------
    word_ids, doc_ids = [], []
    for d, toks in enumerate(token_lists):
        for w in toks:
            if w in word2id:
                word_ids.append(word2id[w])
                doc_ids.append(d)

    word_ids = jnp.asarray(word_ids, dtype=jnp.int32)
    doc_ids  = jnp.asarray(doc_ids,  dtype=jnp.int32)

    # Document pointers for fast slicing -------------------------------------
    doc_lengths_np = np.bincount(np.asarray(doc_ids), minlength=len(texts))
    doc_lengths    = jnp.asarray(doc_lengths_np, dtype=jnp.int32)
    doc_ptrs       = jnp.concatenate([
        jnp.array([0], dtype=jnp.int32),
        jnp.cumsum(doc_lengths, dtype=jnp.int32),
    ])

    return Corpus(word_ids, doc_ids, list(vocab), doc_ptrs)