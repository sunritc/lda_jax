from __future__ import annotations

"""lda_jax.inference.sampler
================================

High-level driver that **orchestrates MCMC** for topic-models implemented
inside :pymod:`lda_jax.models`.  The idea is:

* Keep model-specific kernels (e.g. collapsed Gibbs for LDA) inside the
  model module itself.
* Provide a *uniform* interface to run sampling loops, handle burn-in,
  thinning, progress bars, and compact storage of draws.

This initial version focuses on the **collapsed-Gibbs LDA** but the
scaffolding is generic enough to plug in HDP or partially-collapsed
kernels later.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Optional, Protocol, Tuple

import jax
from jax import Array
from tqdm import tqdm
import jax.numpy as jnp

from lda_jax.utils.process import Corpus
from lda_jax.models.lda import LDAModel, LDAState, collapsed_gibbs_step

__all__ = [
    "SamplerConfig",
    "GibbsSampler",
]

################################################################################
# Config dataclass #############################################################
################################################################################

@dataclass(slots=True)
class SamplerConfig:
    """Hyper-parameters controlling an MCMC run."""

    num_iters: int                   # total sweeps (including burn-in)
    burn_in: int = 0                 # number of initial sweeps to discard
    thin: int = 1                    # record every `thin`-th sweep
    rng_key: Array | None = None     # seed for reproducibility
    show_progress: bool = True       # tqdm progress bar

    def __post_init__(self):
        assert self.num_iters > 0 and self.burn_in < self.num_iters, (
            "`num_iters` must be positive and greater than `burn_in`."
        )

################################################################################
# Protocol for kernels #########################################################
################################################################################

class GibbsKernelFn(Protocol):
    """Signature any collapsed/uncollapsed Gibbs kernel must satisfy."""

    def __call__(
        self, state: LDAState, corpus: Corpus, model: LDAModel, *, key: Array
    ) -> Tuple[LDAState, Array]: ...

################################################################################
# Sampler driver ###############################################################
################################################################################

@dataclass
class GibbsSampler:
    """Thin wrapper that executes a Gibbs kernel repeatedly.

    Parameters
    ----------
    corpus
        Tokenised corpus.
    model
        `LDAModel` instance holding hyper-parameters.
    config
        Run-time configuration (iterations, burn-in, rng).
    kernel
        A callable implementing the Gibbs sweep.  Defaults to
        `lda_jax.models.lda.collapsed_gibbs_step`.
    store_state
        *If provided*, a function ``fn(state) -> Any`` that extracts the
        quantities you want to keep from each post-burn-in draw.  By
        default, we keep individual **n_kw** and **n_dk** count matrices,
        which are sufficient for computing φ/θ estimates.
    """

    corpus: Corpus
    model: LDAModel
    config: SamplerConfig
    kernel: GibbsKernelFn = collapsed_gibbs_step
    # Function that extracts what we store from each draw
    store_state: Optional[Callable[[LDAState], dict[str, Array]]] = None

    # State managed internally (initialised in run())
    _rng_key: Array = field(init=False, repr=False)
    _state: LDAState = field(init=False, repr=False)
    _trace: List = field(init=False, repr=False, default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Sequence[dict[str, Array]]:
        """Execute Gibbs sampling according to ``SamplerConfig``."""
        self._rng_key = (
            self.config.rng_key if self.config.rng_key is not None else jax.random.PRNGKey(0)
        )
        self._state = self.model.init_state(self.corpus, key=self._rng_key)

        if self.store_state is None:
            self.store_state = lambda s: {
                "n_kw": s.n_kw.copy(),
                "n_dk": s.n_dk.copy(),
            }

        iterator = range(self.config.num_iters)
        if self.config.show_progress:
            iterator = tqdm(iterator, desc="Gibbs")

        for it in iterator:
            self._rng_key, sub = jax.random.split(self._rng_key)
            self._state, self._rng_key = self.kernel(self._state, self.corpus, self.model, key=sub)

            if it >= self.config.burn_in and (it - self.config.burn_in) % self.config.thin == 0:
                self._trace.append(self.store_state(self._state))

        return self._trace

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def posterior_phi(self) -> Array:
        """Posterior mean of φ (topic‑word distribution)."""
        if not self._trace:
            raise ValueError("Sampler trace is empty; run() first.")

        K, V, eta = self.model.num_topics, self.model.vocab_size, self.model.eta
        n_kw_stack = jnp.stack([d["n_kw"] for d in self._trace])  # (T, K, V)
        n_k_stack = n_kw_stack.sum(axis=2, keepdims=True)  # (T, K, 1)
        phi_stack = (n_kw_stack + eta) / (n_k_stack + V * eta)  # (T, K, V)
        return phi_stack.mean(axis=0)  # (K, V)

    def posterior_theta(self) -> Array:
        """Posterior mean of θ (doc‑topic distribution)."""
        if not self._trace:
            raise ValueError("Sampler trace is empty; run() first.")

        D, K, alpha = self.corpus.num_docs, self.model.num_topics, self.model.alpha
        n_dk_stack = jnp.stack([d["n_dk"] for d in self._trace])  # (T, D, K)
        n_d_stack = n_dk_stack.sum(axis=2, keepdims=True)  # (T, D, 1)
        theta_stack = (n_dk_stack + alpha) / (n_d_stack + K * alpha)  # (T, D, K)
        return theta_stack.mean(axis=0)  # (D, K)

