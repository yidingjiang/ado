import math
import typing as tp
import numpy as np
import equinox as eqx
import jax

jnp = jax.numpy
KeyArray = tp.Any
Array = jax.numpy.ndarray
jrandom = jax.random


class Embedding(eqx.Module):
    """For some reason, default Embedding impl is slow under vmap+JIT."""
    V: int = eqx.field(static=True)  # num embeddings
    D: int = eqx.field(static=True)  # embedding size
    weight_VxD: Array

    def __init__(
            self, num_embeddings: int, embedding_size: int, weight: tp.Optional[Array]=None,
            *, key: tp.Optional[KeyArray]=None
    ):
        super().__init__()
        self.V, self.D = num_embeddings, embedding_size
        if weight is not None:
            self.weight_VxD = weight
        elif key is not None:
            self.weight_VxD = jrandom.normal(key, (self.V, self.D))
        else:
            raise ValueError("need weight or key to be not None")

    @jax.named_scope("Embedding")
    def __call__(self, x_T, *, key=None):
        return jnp.take(self.weight_VxD, x_T, axis=0)


class Linear(eqx.Module):
    """Linear with trunc normal init."""
    weight_MxN: Array

    def __init__(
        self, in_features: int, out_features: int, weight: tp.Optional[Array]=None, 
        *, key: tp.Optional[KeyArray]=None
    ):
        super().__init__()
        if weight is not None:
            self.weight_MxN = weight
        elif key is not None:
            self.weight_MxN = (1 / math.sqrt(in_features)) * jrandom.truncated_normal(
                key, lower=-2, upper=2, shape=(out_features, in_features))
        else:
            raise ValueError("need weight or key to be not None")

    @jax.named_scope("Linear")
    def __call__(self, x_N: Array, *, key: KeyArray=None) -> Array:
        x_M = self.weight_MxN @ x_N
        return x_M


class RMSNorm(eqx.Module):
    weight_M: tp.Optional[Array]
    eps: float

    def __init__(self, dim: int, use_weight=False, eps=1e-6):
        super().__init__()
        self.eps, self.weight_M = eps, None
        if use_weight:
            self.weight_M = jnp.ones((dim,))

    @jax.named_scope("RMSNorm")
    def __call__(self, x_M: Array) -> Array:
        out_M = x_M * jax.lax.rsqrt(jnp.mean(jnp.square(x_M), keepdims=True) + self.eps)
        if self.weight_M is not None:
            out_M = out_M * self.weight_M
        return out_M


## RoPE functions
def fixed_pos_embedding(C: int, T: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    inv_freq_D = 1.0 / (10000 ** (np.arange(0, C, 2) / C))  # D = C // 2
    sinusoid_inp_TxD = np.einsum("i,j -> i j", np.arange(T), inv_freq_D)
    return np.sin(sinusoid_inp_TxD), np.cos(sinusoid_inp_TxD)


def rotate_every_two(x: Array) -> Array:  # [a b c d] -> [-b a -d c]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return jnp.reshape(x, x.shape[:-2] + (-1,))


def apply_rotary_pos_emb(x_HxTxC: Array, sin_TxD_np: np.ndarray, cos_TxD_np: np.ndarray) -> Array:
    sin_TxD = jnp.asarray(sin_TxD_np, dtype=x_HxTxC.dtype)
    cos_TxD = jnp.asarray(cos_TxD_np, dtype=x_HxTxC.dtype)
    sin_1xTxC = jnp.stack((sin_TxD, sin_TxD), axis=-1)
    sin_1xTxC = jnp.reshape(sin_1xTxC, sin_1xTxC.shape[:-2] + (-1,))
    cos_1xTxC = jnp.stack((cos_TxD, cos_TxD), axis=-1)
    cos_1xTxC = jnp.reshape(cos_1xTxC, cos_1xTxC.shape[:-2] + (-1,))
    return (x_HxTxC * cos_1xTxC) + (rotate_every_two(x_HxTxC) * sin_1xTxC)
