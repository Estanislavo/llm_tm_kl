"""Semantic initialization and frequency-scaled pseudo-counts for AARTM.

Uses the same sufficient statistics and local attention as aartm.py. A positive
prior adds word-topic evidence directly, including entries previously at zero.
The legacy KL option is retained for controlled comparisons.
"""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from .core import EPSILON, norm, calc_attn, calc_attn_transposed, get_context_weights_1d


def _kl(phi, q):
    r = phi / (phi.sum(axis=1, keepdims=True) + EPSILON)
    return jnp.sum(q * (jnp.log(q + EPSILON) - jnp.log(r + EPSILON)))


@jax.jit
def document_average(matrix, bounds):
    """Uniform attention to other tokens in the same document (no self token).

    Each document block is symmetric, so this is also its transpose action.
    """
    ids = jnp.searchsorted(bounds, jnp.arange(len(matrix)), side='right') - 1
    sums = jax.ops.segment_sum(matrix, ids, len(bounds) - 1)
    lengths = jnp.diff(bounds)[ids, None]
    return (sums[ids] - matrix) / jnp.maximum(lengths - 1, 1)


def mixed_attention(matrix, bounds, weights, document_weight):
    return (1 - document_weight) * calc_attn(matrix, bounds, weights) + document_weight * document_average(matrix, bounds)


@partial(jax.jit, static_argnames=("steps",))
def _fit(data, bounds, phi, q, weights, kl_strength, prior_strength, final_ratio, document_weight, *, steps):
    n_w = jnp.bincount(data, length=phi.shape[0])[:, None]

    def step(state, iteration):
        phi, n_t = state
        theta = mixed_attention(phi[data], bounds, weights, document_weight)
        post = norm(phi[data] * theta / (n_t + EPSILON), axis=1)
        n_wt = jax.ops.segment_sum(post, data, phi.shape[0])
        ratio = post / (theta + EPSILON)
        adjoint = (1 - document_weight) * calc_attn_transposed(ratio, bounds, weights) + document_weight * document_average(ratio, bounds)
        N_wt = jax.ops.segment_sum(adjoint, data, phi.shape[0])
        mass = n_wt.sum(axis=1, keepdims=True)
        coeff = n_wt / jnp.where(mass > EPSILON, mass, 1.)
        schedule = 1. + (final_ratio - 1.) * iteration / max(steps - 1, 1)
        grad = jax.grad(_kl)(phi, q)
        evidence = n_wt + coeff * N_wt - kl_strength * schedule * coeff * grad
        evidence += prior_strength * schedule * n_w * q
        next_phi = norm(evidence, axis=1)
        return (next_phi, post.sum(axis=0)), None

    (phi, n_t), _ = jax.lax.scan(step, (phi, jnp.ones(phi.shape[1])), jnp.arange(steps))
    return phi, n_t


def fit_semantic_aartm(data, bounds, q, *, seed=0, initialization=0.,
                       kl_strength=0., prior_strength=0., final_ratio=1.,
                       steps=100, ctx_len=5, document_weight=0.):
    """Return p(topic|word), n_t. q has shape (vocabulary, topics).

    initialization mixes q with seeded random rows. prior_strength is the
    pseudo-count mass per observed word occurrence, not the legacy KL lambda.
    Training uses full document bounds, avoiding artificial batch boundaries.
    """
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2 or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("q must be a finite nonnegative word-topic matrix")
    if not np.allclose(q.sum(axis=1), 1., atol=1e-5):
        raise ValueError("Each q row must sum to one")
    if not 0 <= initialization <= 1 or min(kl_strength, prior_strength, final_ratio) < 0:
        raise ValueError("Invalid initialization or regularization strength")
    if steps < 1 or ctx_len < 1:
        raise ValueError("steps and ctx_len must be positive")
    if not 0 <= document_weight <= 1:
        raise ValueError("document_weight must be between zero and one")
    q = jnp.asarray(q)
    random_phi = norm(jax.random.uniform(jax.random.key(seed), q.shape), axis=1)
    # Keep support at initialization; later pseudo-counts can also revive zeros.
    phi = norm((1 - initialization) * random_phi + initialization * q + 1e-8, axis=1)
    weights = get_context_weights_1d(ctx_len, .6, False)
    return _fit(jnp.asarray(data), jnp.asarray(bounds), phi, q, weights,
                jnp.asarray(kl_strength), jnp.asarray(prior_strength),
                jnp.asarray(final_ratio), jnp.asarray(document_weight), steps=steps)


def word_topic_probabilities(phi, data):
    phi = np.asarray(phi, dtype=np.float64)
    counts = np.bincount(np.asarray(data), minlength=len(phi))
    result = phi * counts[:, None]
    return result / np.maximum(result.sum(axis=0, keepdims=True), 1e-30)
