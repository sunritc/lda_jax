from __future__ import annotations

"""lda_jax.models.lda
======================

A *minimal but complete* Latent Dirichlet Allocation (LDA) implementation
built for JAX.  The goals are

1. **Pure-JAX kernels** so everything can be `jit`-ed / `vmap`-ed.
2. **Clear data structures** that match the Gibbs-sampling math.
3. **Separation of model vs inference**, so future HMC / SVI back-ends can
   reuse the same `LDAState` container.

----------------------------------------------------------------------
Data structures
----------------------------------------------------------------------

``Corpus``  (from :pymod:`lda_jax.utils.process`)
    Bag-of-words representation produced by the pre-processing utils.  Only
    the four arrays are required by the sampler:

    ============= ============================= =========================
    field         shape / dtype                description
    ============= ============================= =========================
    ``word_ids``  ``(N,)  int32``              token → vocab id
    ``doc_ids``   ``(N,)  int32``              token → document id
    ``vocab``     ``List[str]``                id → token string
    ``doc_ptrs``  ``(D+1,)  int32``            prefix sums for docs
    ============= ============================= =========================

``LDAState``
    The *latent* variables tracked by collapsed Gibbs:

    ========== =========================================
    field       shape / dtype        notes
    ========== =========================================
    ``z``       ``(N,)  int32``      topic assignment per token
    ``n_dk``    ``(D,K) int32``      doc–topic counts
    ``n_kw``    ``(K,V) int32``      topic–word counts
    ``n_k``     ``(K,)  int32``      topic totals (row-sum of ``n_kw``)
    ========== =========================================

----------------------------------------------------------------------
Key API
----------------------------------------------------------------------

* :func:`init_state` – random initialisation given a corpus.
* :func:`collapsed_gibbs_step` – **one sweep** over all tokens (word-level).
* :func:`run_gibbs` – driver to collect an arbitrary number of iterations.
* :func:`log_likelihood` – joint log-prob under collapsed model (for debugging).

The implementation favours clarity over ultra-high throughput.  For real-world
corpora you’ll likely switch to a *blocked* version that updates an entire
**document** at once; this file keeps the simpler word-level kernel so the
mathematics is transparent.
"""

from dataclasses import dataclass
from typing import Tuple, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from lda_jax.utils.process import Corpus

__all__ = [
    "LDAState",
    "LDAModel",
    "collapsed_gibbs_step",
    "run_gibbs",
    "log_likelihood",
]

################################################################################
# Containers ###################################################################
################################################################################

class LDAState(NamedTuple):
    """Latent variables tracked during collapsed Gibbs inference."""

    z: Array      # (N,)  topic per token
    n_dk: Array   # (D,K) doc-topic counts
    n_kw: Array   # (K,V) topic-word counts
    n_k: Array    # (K,)   topic totals


@dataclass(frozen=True)
class LDAModel:
    """Fixed hyper-parameters of the LDA generative model."""

    num_topics: int  # K
    vocab_size: int  # V
    alpha: float = 0.1  # symmetric Dir_K(alpha)
    eta: float = 0.1    # symmetric Dir_V(eta)

    def init_state(self, corpus: Corpus, *, key: Array) -> LDAState:
        """Randomly assign each token to a topic and build count tables."""
        D, K, V = corpus.num_docs, self.num_topics, self.vocab_size

        key, sub = jax.random.split(key)
        z = jax.random.randint(sub, shape=(corpus.num_tokens,), minval=0, maxval=K, dtype=jnp.int32)

        n_dk = jnp.zeros((D, K), dtype=jnp.int32)
        n_kw = jnp.zeros((K, V), dtype=jnp.int32)
        n_k  = jnp.zeros((K,),   dtype=jnp.int32)

        # Increment counts using vectorised segment-ops ----------------------
        n_dk = n_dk.at[corpus.doc_ids, z].add(1)
        n_kw = n_kw.at[z, corpus.word_ids].add(1)
        n_k  = n_k.at[z].add(1)

        return LDAState(z, n_dk, n_kw, n_k)

################################################################################
# Collapsed Gibbs kernel ########################################################
################################################################################

def _token_conditional_probs(state: LDAState, word_id: int, doc_id: int, model: LDAModel) -> Array:
    """Compute unnormalised P(topic | rest) for a *single* token (collapsed)."""
    K, V = model.num_topics, model.vocab_size
    alpha, eta = model.alpha, model.eta

    theta = state.n_dk[doc_id] + alpha             # (K,)
    phi   = (state.n_kw[:, word_id] + eta) / (state.n_k + V * eta)  # (K,)
    probs = theta * phi
    return probs / probs.sum()


def collapsed_gibbs_step(state: LDAState, corpus: Corpus, model: LDAModel, *, key: Array) -> Tuple[LDAState, Array]:
    """One full sweep over **all tokens** (word-level Gibbs)."""

    def _update_token(carry, idx):
        st, k = carry
        k_old  = st.z[idx]
        d      = corpus.doc_ids[idx]
        w      = corpus.word_ids[idx]

        # Remove token -------------------------------------------------------
        st = st._replace(
            z    = st.z.at[idx].set(-1),
            n_dk = st.n_dk.at[d, k_old].add(-1),
            n_kw = st.n_kw.at[k_old, w].add(-1),
            n_k  = st.n_k.at[k_old].add(-1),
        )

        # Sample new topic ---------------------------------------------------
        probs      = _token_conditional_probs(st, w, d, model)
        k, subkey  = jax.random.split(k)
        k_new      = jax.random.categorical(subkey, jnp.log(probs))

        # Add token back -----------------------------------------------------
        st = st._replace(
            z    = st.z.at[idx].set(k_new),
            n_dk = st.n_dk.at[d, k_new].add(1),
            n_kw = st.n_kw.at[k_new, w].add(1),
            n_k  = st.n_k.at[k_new].add(1),
        )
        return (st, k), None

    (state, key), _ = jax.lax.scan(_update_token, (state, key), jnp.arange(corpus.num_tokens))
    return state, key

################################################################################
# Driver #######################################################################
################################################################################

def run_gibbs(
    corpus: Corpus,
    model: LDAModel,
    *,
    num_iters: int,
    key: Array,
    keep_history: bool = False,
) -> Tuple[LDAState, Sequence[LDAState]]:
    """Run collapsed Gibbs for a specified number of iterations.

    Parameters
    ----------
    corpus
        Bag-of-words data.
    model
        ``LDAModel`` with hyper-parameters.
    num_iters
        Number of full sweeps.
    key
        JAX PRNG key.
    keep_history
        If *True* return a list with state for each iteration.
    """
    """Fully-jitted Gibbs driver that optionally returns a trace."""
    # -------------------------
    # (1) initial state
    # -------------------------
    init_state = model.init_state(corpus, key=key)

    # -------------------------
    # (2) single sweep kernel
    # -------------------------
    def _one_sweep(carry, _):
        state, rng = carry
        rng, sub = jax.random.split(rng)
        state, rng = collapsed_gibbs_step(state, corpus, model, key=sub)  # word- or doc-blocked
        return (state, rng), state  # 2nd value is what scan will stack

    # -------------------------
    # (3) JIT-compiled scan
    # -------------------------
    @jax.jit
    def _scan_gibbs(state, rng):
        (state, rng), states = jax.lax.scan(_one_sweep, (state, rng), None, length=num_iters)
        return state, states  # final state + stack [num_iters, ...]

    final_state, all_states = _scan_gibbs(init_state, key)

    # -------------------------
    # (4) Return according to flag
    # -------------------------
    if keep_history:
        # `all_states` is a PyTree whose leaves have shape (num_iters, …)
        # Convert to a Python list of LDAState – or keep the stacked version.
        history: Sequence = jax.tree_util.tree_map(  # split along axis 0
            lambda x: list(jnp.split(x, num_iters, axis=0)), all_states
        )
        # tree_map produced a tree of lists; zip it into a list of LDAState
        history = [model.state_class(*leaf) for leaf in zip(*history.values())]  # type: ignore
        return final_state, history
    else:
        return final_state, None

################################################################################
# Likelihood / Perplexity ######################################################
################################################################################

def log_likelihood(state: LDAState, model: LDAModel, *, with_const: bool = False) -> Array:
    """Compute the *joint* log-prob P(z, w | α, η) in collapsed form.

    For debugging / convergence diagnostics.
    """
    K, V = model.num_topics, model.vocab_size
    alpha, eta = model.alpha, model.eta

    # Dirichlet-multinomial normalisers -------------------------------------
    def _dirichlet_multinomial(counts_row, alpha_vec):
        return (
            jax.scipy.special.gammaln(alpha_vec.sum())
            - jax.scipy.special.gammaln(alpha_vec.sum() + counts_row.sum())
            + jax.scipy.special.gammaln(alpha_vec + counts_row).sum()
            - jax.scipy.special.gammaln(alpha_vec).sum()
        )

    # Document component ----------------------------------------------------
    alpha_vec = jnp.full((K,), alpha)
    ll_docs = jax.vmap(_dirichlet_multinomial, in_axes=(0, None))(state.n_dk, alpha_vec).sum()

    # Topic component -------------------------------------------------------
    eta_vec = jnp.full((V,), eta)
    ll_topics = jax.vmap(_dirichlet_multinomial, in_axes=(0, None))(state.n_kw, eta_vec).sum()

    const = 0.0
    if with_const:
        const = -jax.scipy.special.gammaln(alpha_vec).sum() * state.n_dk.shape[0]

    return ll_docs + ll_topics + const

################################################################################
# End of file ##################################################################
################################################################################