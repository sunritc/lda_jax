from __future__ import annotations

"""Quick unit test: can Gibbs sampler recover topics on synthetic corpus."""

import numpy as np
import jax

from scipy.optimize import linear_sum_assignment

from lda_jax.utils.generator import generate_lda_corpus
from lda_jax.models.lda import LDAModel
from lda_jax.inference.sampler import GibbsSampler, SamplerConfig


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between two discrete distributions."""
    return 0.5 * np.abs(p - q).sum()


def test_lda_recovery():
    # ------------------------------------------------------------------
    # 1. Generate synthetic data
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(0)
    synth = generate_lda_corpus(
        key,
        num_docs=100,
        num_topics=5,
        vocab_size=100,
        doc_length=35,
    )

    corpus     = synth.corpus
    true_beta  = np.asarray(synth.beta)  # (K, V)

    # ------------------------------------------------------------------
    # 2. Fit model with Gibbs sampler
    # ------------------------------------------------------------------
    model   = LDAModel(num_topics=5, vocab_size=corpus.vocab_size)
    config  = SamplerConfig(num_iters=1000, burn_in=500, thin=20, rng_key=jax.random.PRNGKey(1), show_progress=False)
    sampler = GibbsSampler(corpus, model, config)
    sampler.run()
    est_phi = np.asarray(sampler.posterior_phi())  # (K, V)

    # ------------------------------------------------------------------
    # 3. Align topics using Hungarian matching & compute average TV distance
    # ------------------------------------------------------------------
    K = true_beta.shape[0]
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = _tv_distance(true_beta[i], est_phi[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    avg_tv = cost_matrix[row_ind, col_ind].mean()

    # We expect reasonable recovery (< 0.25 TV distance on average)
    assert avg_tv < 0.25, f"Average topic TV distance too high: {avg_tv:.3f}"
