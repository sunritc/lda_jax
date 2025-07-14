from __future__ import annotations

"""jax_topics.data_tools
=================================

A small self‑contained helper module with responsibility:

1. **Synthetic Data Generation** – draw corpora from the LDA generative
   model for unit tests and benchmarking.

The code is designed to be jit‑friendly
"""
from typing import  Sequence, NamedTuple, Tuple, Union
from .process import Corpus
import jax
from jax import Array
import jax.numpy as jnp

################################################################################
# 1. Synthetic LDA generator ####################################################
################################################################################

class LDASynthetic(NamedTuple):
    """Return object for :func:`generate_lda_corpus`."""

    corpus: Corpus
    z:      Array  # (N,) topic assignment per token
    theta:  Array  # (D, K) document–topic dists
    beta:   Array  # (K, V) topic–word dists

# -----------------------------------------------------------------------------
# Utility: Dirichlet draw that returns (next_key, sample)
# -----------------------------------------------------------------------------

def _draw_dirichlet(key: Array, alpha: Array, shape: Tuple[int, ...]) -> Tuple[Array, Array]:
    key, sub = jax.random.split(key)
    return key, jax.random.dirichlet(sub, alpha, shape=shape)


def generate_lda_corpus(
    key: Array,
    *,
    num_docs: int,
    num_topics: int,
    vocab_size: int,
    doc_length: Union[int, Sequence[int]],
    alpha: float = 0.1,
    eta: float = 0.1,
) -> LDASynthetic:
    """Draw a corpus from the standard LDA generative model."""

    # 1. β  ~ Dir_V(eta)
    key, beta = _draw_dirichlet(key, jnp.full((vocab_size,), eta), (num_topics,))

    # 2. θ  ~ Dir_K(alpha)
    key, theta = _draw_dirichlet(key, jnp.full((num_topics,), alpha), (num_docs,))

    # 3. Document lengths
    if isinstance(doc_length, int):
        doc_lengths = jnp.full((num_docs,), doc_length, dtype=jnp.int32)
    else:
        doc_lengths = jnp.asarray(doc_length, dtype=jnp.int32)
        assert doc_lengths.shape == (num_docs,)

    # 4. Sample individual documents ----------------------------------------
    def _sample_doc(rng: Array, d_idx: int):
        L_d       = int(doc_lengths[d_idx])  # cast to python int for type checker
        rng, sub1 = jax.random.split(rng)
        z_dn      = jax.random.categorical(sub1, jnp.log(theta[d_idx]), shape=(L_d,))
        rng, sub2 = jax.random.split(rng)
        w_dn      = jax.random.categorical(sub2, jnp.log(beta[z_dn]), axis=-1)
        return rng, (z_dn, w_dn)

    keys = jax.random.split(key, num_docs + 1)
    key  = keys[0]

    z_all, w_all, d_all = [], [], []
    for d in range(num_docs):
        _, (z_dn, w_dn) = _sample_doc(keys[d + 1], d)
        z_all.append(z_dn)
        w_all.append(w_dn)
        d_all.append(jnp.full_like(z_dn, d))

    z        = jnp.concatenate(z_all)
    word_ids = jnp.concatenate(w_all)
    doc_ids  = jnp.concatenate(d_all)

    doc_ptrs = jnp.concatenate([
        jnp.array([0], dtype=jnp.int32),
        jnp.cumsum(doc_lengths, dtype=jnp.int32),
    ])

    vocab  = [f"w{i}" for i in range(vocab_size)]
    corpus = Corpus(word_ids, doc_ids, vocab, doc_ptrs)
    return LDASynthetic(corpus, z, theta, beta)
