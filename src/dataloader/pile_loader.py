import collections
from typing import Dict
import numpy as np
import jax
from jax.experimental import multihost_utils
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoTokenizer
from .adaptive_loader import TasksSampler
from src.tfds import the_pile_grouped
from src.sharding import get_shard_fn, get_data_sharding
from .abstract import Loader
Mesh = jax.sharding.Mesh


DOREMI_WEIGHTS = {
    "Pile-CC": 0.1379,
    "YoutubeSubtitles": 0.0117,
    "PhilPapers": 0.0093,
    "HackerNews": 0.0084,
    "Enron Emails": 0.004,
    "EuroParl": 0.012,
    "Ubuntu IRC": 0.0083,
    "BookCorpus2": 0.0037,
    "NIH ExPorter": 0.0084,
    "OpenSubtitles": 0.0032,
    "Gutenberg (PG-19)": 0.0292,
    "DM Mathematics": 0.0019,
    "Wikipedia (en)": 0.1068,
    "OpenWebText2": 0.1905,
    "Github": 0.0325,
    "FreeLaw": 0.038,
    "USPTO Backgrounds": 0.0327,
    "Books3": 0.0757,
    "PubMed Abstracts": 0.097,
    "StackExchange": 0.0746,
    "ArXiv": 0.0535,
    "PubMed Central": 0.0608,
}

DEFAULT_WEIGHTS = {
    'FreeLaw': 0.04493927695030662,
    'Enron Emails': 0.000998021865918546,
    'Github': 0.12267758913758665,
    'OpenSubtitles': 0.015835745965429738,
    'PubMed Central': 0.12148621531516873,
    'OpenWebText2': 0.10960682218906206,
    'StackExchange': 0.049107965728456646,
    'Pile-CC': 0.1824984780261193,
    'ArXiv': 0.08862621733009907,
    'USPTO Backgrounds': 0.02616577419097875,
    'Books3': 0.10458626728299704,
    'Wikipedia (en)': 0.04016661238580172,
    'PubMed Abstracts': 0.02212837481440004,
    'NIH ExPorter': 0.0018685647881937016,
    'BookCorpus2': 0.006327357399975309,
    'EuroParl': 0.008072738376112661,
    'HackerNews': 0.004731183407655429,
    'DM Mathematics': 0.019084626704901235,
    'YoutubeSubtitles': 0.004027438721554198,
    'PhilPapers': 0.0026731438901686708,
    'Ubuntu IRC': 0.004850316881507234,
    'Gutenberg (PG-19)': 0.0195412686476066
}


def make_tfds_dataset(task_name, split, block_size, n_proc, proc_idx, shuffle=False, data_dir=None, repeat=True):
    """Returns a dset that yields block_size tokens at a time."""
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    EOS_TOKEN_ID = tokenizer.eos_token_id
    valid_name = the_pile_grouped.TASK_TO_VALID_NAME[task_name]
    read_config = tfds.ReadConfig(skip_prefetch=True)
    builder = tfds.builder(f"the_pile_grouped/{valid_name}", data_dir=data_dir)
    sharded_split = split
    shard_files = builder.info.splits[split].num_examples >= n_proc
    if shard_files:
        # Don't shard unless there's enough examples to shard across all workers
        sharded_split = tfds.even_splits(split, n_proc, drop_remainder=False)[proc_idx]
    dset = tfds.load(
        f"the_pile_grouped/{valid_name}", split=sharded_split,
        shuffle_files=shuffle, read_config=read_config, data_dir=data_dir)
    if repeat:
        dset = dset.repeat()
    dset = dset.map(lambda x: tf.concat(
        [x['tokens'], [EOS_TOKEN_ID]], axis=0), num_parallel_calls=tf.data.AUTOTUNE)
    dset = dset.rebatch(block_size, drop_remainder=True)
    if not shard_files:
        assert not shuffle
        dset = dset.shard(n_proc, proc_idx)
    if shuffle:
        shuffle_size = 1_000
        if the_pile_grouped.TASK_TO_AVG_LENGTH[task_name] > 10_000:
            shuffle_size = 10_000
        dset = dset.shuffle(shuffle_size)
    return dset


def make_eval_dset(batch_size, block_size, data_dir):
    n_proc = jax.process_count()
    proc_idx = jax.process_index()
    proc_bs = batch_size // n_proc
    dsets = []
    for i, task in enumerate(the_pile_grouped.TASKS):
        dset = make_tfds_dataset(
            task, "val", block_size + 1, n_proc, proc_idx,
            shuffle=False, data_dir=data_dir, repeat=False)
        dset = dset.map(lambda x: (i, x))
        dsets.append(dset)
    return tf.data.Dataset.sample_from_datasets(
        dsets, stop_on_empty_dataset=False, rerandomize_each_iteration=False, seed=0,
    ).batch(proc_bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


class ThePileBalanced(Loader):
    """Task balanced sampling."""
    def __init__(
            self,
            batch_size: int,
            split: str,
            mesh,
            block_size: int,
            g_accum_iters: int,
            data_dir: str,
            balancing_strategy: str = "task_balanced",  # | "doremi" | "empirical"
    ):
        self.n_proc = jax.process_count()
        self.proc_idx = jax.process_index()
        self.proc_bs = batch_size // self.n_proc
        self.g_accum_iters = g_accum_iters
        dsets = []
        for i, task in enumerate(the_pile_grouped.TASKS):
            dset = make_tfds_dataset(
                task, split, block_size + 1, self.n_proc, self.proc_idx,
                shuffle=split=="train", data_dir=data_dir)
            dset = dset.map(lambda x: (i, x))
            dsets.append(dset)
        weights = None  # no weights => task_balanced
        if balancing_strategy == "doremi":
            weights = np.array([DOREMI_WEIGHTS[task] for task in the_pile_grouped.TASKS])
            weights = (weights / weights.sum()).tolist()
        elif balancing_strategy == "empirical":
            weights = np.array([DEFAULT_WEIGHTS[task] for task in the_pile_grouped.TASKS])
            weights = (weights / weights.sum()).tolist()
        self.dset = tf.data.Dataset.sample_from_datasets(
            dsets, stop_on_empty_dataset=True, rerandomize_each_iteration=True,
            weights=weights,
        ).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        self.sharding_fn = get_shard_fn(mesh, get_data_sharding(mesh))
        self.cumulative_cnts = collections.Counter()

    def get_batch(self, itr):
        task_idcs, samples = [], []
        for _ in range(self.g_accum_iters * self.proc_bs):
            task_idx, x = next(self.dset)
            samples.append(x)
            task_idcs.append(task_idx.item())
        samples_BGxTp1 = np.stack(samples)
        samples_GxBxTp1 = samples_BGxTp1.reshape(self.g_accum_iters, self.proc_bs, -1)
        x_GxBxT = samples_GxBxTp1[..., :-1]
        y_GxBxT = samples_GxBxTp1[..., 1:]
        task_id_GxB = np.array(task_idcs).reshape(self.g_accum_iters, self.proc_bs)
        info = {"task_ids": task_id_GxB}
        x_GxBxT, y_GxBxT, info = jax.tree_util.tree_map(self.sharding_fn, (x_GxBxT, y_GxBxT, info))
        return x_GxBxT, y_GxBxT, info

    def update(self, loss_GxB, info, itr, **kwargs):
        loss_GxB, task_id_GxB = multihost_utils.process_allgather((loss_GxB, info["task_ids"]))
        # count number of examples with each task id
        self.cumulative_cnts.update(task_id_GxB.reshape(-1).tolist())
        metrics = {}
        for task_id in range(len(the_pile_grouped.TASKS)):
            metrics[f"loss_cnts_cum/task{task_id}"] = self.cumulative_cnts[task_id]
            mask = task_id_GxB == task_id
            if mask.sum() > 0:
                metrics[f"loss/task{task_id}"] = loss_GxB[mask].mean().item()
        return metrics


class ThePileTasksSampler(TasksSampler):

    def __init__(self, split: str, block_size: int, data_dir: str):
        self.n_proc = jax.process_count()
        self.proc_idx = jax.process_index()
        self.block_size = block_size
        dsets = []
        self.total_token_n = []
        for i, task in enumerate(the_pile_grouped.TASKS):
            self.total_token_n.append(DEFAULT_WEIGHTS[task])
            dset = make_tfds_dataset(
                task, split, block_size + 1, self.n_proc, self.proc_idx,
                shuffle=split=="train", data_dir=data_dir)
            dset = dset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
            dsets.append(dset)
        self.total_token_n = np.array(self.total_token_n)
        self.dsets = dsets
        self._k = len(dsets)
        self.tasks = the_pile_grouped.TASKS
        print(f"{self.k} tasks. Order: {self.tasks}.")

        self._init_dist = self.total_token_n / self.total_token_n.sum()

    def get_batch(self, num_per_task, B):
        tokens, task_ids = [], []
        for task_idx in range(self.k):
            num_samples = num_per_task[task_idx].item()
            if num_samples == 0:
                continue
            for _ in range(num_samples):
                tokens.append(next(self.dsets[task_idx]))
            task_ids.append(np.ones(num_samples, dtype=np.int32) * task_idx)
        tokens = np.stack(tokens)
        x_BxT, y_BxT = tokens[..., :-1], tokens[..., 1:]
        task_id_B = np.concatenate(task_ids, axis=0)
        return x_BxT, y_BxT, task_id_B

    @property
    def init_dist(self):
        return self._init_dist

    @property
    def k(self):
        return self._k
