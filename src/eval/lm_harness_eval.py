import typing as tp
import copy

import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp

from tqdm import tqdm
from transformers import AutoTokenizer

import lm_eval

# from lm_eval.base import BaseLM
from lm_eval import utils
from lm_eval.models.utils import Collator, stop_sequences_criteria

from llm_harness_jax_utils import pad_and_concat

# Add the following imports
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

jnp, jrandom, vmap, scan, jtu = (
    jax.numpy,
    jax.random,
    jax.vmap,
    jax.lax.scan,
    jax.tree_util,
)


def cast_pytree(pytree: tp.Any, dtype: jnp.dtype) -> tp.Any:
    """Cast a pytree of arrays to a given dtype, ignore non-arrays."""

    def cast(x):
        if eqx.is_array(x):
            return x.astype(dtype)
        return x

    return jtu.tree_map(cast, pytree)


def get_shard_fn(mesh, sharding):
    """Shard fn for data parallelism.
    Different from training because we don't have a microbatch dimension.
    Handles non-divisible data by simple padding."""
    n_procs = jax.process_count()
    n_local_devices = len(mesh.local_devices)

    def shard(x):
        total_devices = n_procs * n_local_devices
        items_per_device = -(-x.shape[0] // total_devices)  # Ceiling division
        pad_size = items_per_device * total_devices - x.shape[0]

        # Pad the input if necessary
        if pad_size > 0:
            pad_width = [(0, pad_size)] + [(0, 0)] * (x.ndim - 1)
            x_padded = jnp.pad(x, pad_width, mode='constant', constant_values=0)
        else:
            x_padded = x

        local_shard_size = items_per_device * n_local_devices
        xs = jax.device_put(jnp.array_split(x_padded[:local_shard_size], n_local_devices), mesh.local_devices)

        global_shape = (x_padded.shape[0], *x_padded.shape[1:])
        sharded_array = jax.make_array_from_single_device_arrays(global_shape, sharding, xs)

        return sharded_array

    return shard


class LMEvaluator(lm_eval.api.model.LM):
    _DEFAULT_MAX_LENGTH = 2048

    """Adapting LLM harness for our GPT model."""
    def __init__(self, model, mesh, batch_size=2, max_length=None, truncation=False):
        super().__init__()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        self.truncation = truncation
        self.batch_size = batch_size
        self._max_length = max_length
        self.mesh = mesh
        data_sharding = NamedSharding(self.mesh, P(('replica', 'data'), None))  # (B, D)
        self.data_shard_fn = get_shard_fn(self.mesh, data_sharding)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        # seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        # for attr in seqlen_config_attrs:
        #     if hasattr(self.model.config, attr):
        #         return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    def loglikelihood(self, requests):
        print("Calling loglikelihood")
        new_reqs = []
        print(f"Processing {len(requests)} requests.")
        for context, continuation in [req.args for req in requests]:
            if context == "":
                raise ValueError("Context cannot be empty.")
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)
    
    def generate_until(self, requests):
        raise NotImplementedError("This function is not implemented yet.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("This function is not implemented yet.")

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):

        encoding = self.tokenizer.encode(string)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings,
        padding_side="left",
        left_truncate_len=None,
        truncation=False,
    ):
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = False

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        res = []

        def _collate(x):
            """Defines the key for the sorted method"""
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs if override_bs is not None else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)))
        for chunk in chunks:
            inps = []
            inplens = []  # length of context + continuation
            max_padded_len = None  # max length of context + continuation
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying
            ctxlens = []
            cntlens = []

            for _, context_enc, continuation_enc in chunk:
                # if len(context_enc):
                #     context_enc, continuation_enc
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                inp = jnp.array(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :],
                ).astype(jnp.int32)
                (inplen,) = inp.shape

                max_padded_len = (
                    max(max_padded_len, inplen)
                    if max_padded_len is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                inplens.append(inplen)
                ctxlens.append(len(context_enc))
                cntlens.append(len(continuation_enc))

            call_kwargs = {}
            batched_inps = pad_and_concat(
                max_padded_len, inps, padding_side="right"
            )  # [batch, padding_len_inp]
            batched_inps = self.data_shard_fn(batched_inps)
            multi_logits = jax.nn.log_softmax(
                self._model_call(batched_inps, self.model, **call_kwargs), axis=-1
            )[:len(inps)]  # [batch, padding_length (inp or cont), vocab]

            for inp, inplen, ctxlen, cntlen, logits in zip(inps, inplens, ctxlens, cntlens, multi_logits):
                logprobs = logits[None, :]
                token_logprob = jnp.take_along_axis(
                    logprobs[:, ctxlen-1:inplen-1], inp[:, ctxlen:inplen, None], axis=2)  # [1, seq, 1]
                greedy_tokens = logprobs[:, ctxlen-1:inplen-1].argmax(axis=-1)
                cont_toks = jnp.array(inp[:, ctxlen:inplen], dtype=jnp.int32)[None, :]  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                answer = (float(token_logprob.sum()),  bool(max_equal))
                res.append(answer)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def _select_cont_toks(self, logits, contlen=None, inplen=None):
        assert (
            contlen and inplen
        ), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]

        return logits

    def _encode_pair(self, context: str, continuation: str):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation, add_special_tokens=False)
        context_enc = self.tok_encode(context, add_special_tokens=False)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    @eqx.filter_jit
    def _model_call(self, x_BxT, model, attn_mask=None, labels=None):
        logits_BxTxV = vmap(model)(x_BxT, key=None).astype(jnp.float32)
        return logits_BxTxV

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
