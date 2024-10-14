import equinox as eqx
import jax
import jax.numpy as jnp

from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)


def pad_and_concat(
    max_length: int,
    tensors: List,
    padding_side: Literal["right", "left"] = "right",
):
    """
    Method for padding a list of tensors given the maximum tensor
    length in the batch. Used for batching inputs and continuations in
    seq2seq models.
    """
    assert (
        padding_side == "left" or padding_side == "right"
    ), f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"

    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)  # squeeze, in case passed [1, seq] size
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == "right":
                # right-pad
                tensors[i] = jnp.concatenate(
                    [
                        tensor,  # [seq]
                        jnp.zeros(
                            max_length - tensor_len,
                        ).astype(jnp.int32),  # [padding_length - seq]
                    ],
                    axis=0,
                )[jnp.newaxis, :]
            else:
                # left-pad
                tensors[i] = jnp.concatenate(
                    [
                        jnp.zeros(
                            max_length - tensor_len,
                        ).astype(jnp.int32),  # [padding_length - seq]
                        tensor,  # [seq]
                    ],
                    axis=0,
                )[jnp.newaxis, :]
        else:
            tensors[i] = tensor[jnp.newaxis, :]

    return jnp.concatenate(tensors, axis=0)
