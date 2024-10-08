import jax.numpy as jnp
import jax

from tdmpc2_jax.common.util import two_hot, breakpoint_if_nan

def soft_crossentropy(pred_logits: jax.Array, target: jax.Array,
                      low: float, high: float, num_bins: int) -> jax.Array:
  pred = jax.nn.log_softmax(pred_logits, axis=-1)
  target = two_hot(target, low, high, num_bins)
  r = -(pred * target).sum(axis=-1)
  #breakpoint_if_nan(r)
  return r
