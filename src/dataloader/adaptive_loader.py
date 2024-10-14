from abc import ABC, abstractmethod
import typing as tp
from collections import Counter
import numpy as np
import hydra
import jax
from jax.experimental import multihost_utils
from src.data_selectors.ado import AdoSelector
from src.dataloader.abstract import Array
from src.sharding import get_shard_fn, get_data_sharding
from .abstract import Loader


class TasksSampler(ABC):
    """Helper class for sampling from a dataset that has been grouped into tasks."""
    @abstractmethod
    def __init__(self, split, g_accum_iters, block_size, **kwargs):
        pass

    @abstractmethod
    def get_batch(self, num_per_task, B) -> tp.Tuple[Array, Array, Array]:
        pass

    @property
    @abstractmethod
    def k(self) -> int:
        pass

    @property
    @abstractmethod
    def init_dist(self) -> np.ndarray:
        pass


class AdaptiveLoader(Loader):
    def __init__(
            self,
            batch_size: int,
            split: str,
            mesh,
            block_size: int, 
            g_accum_iters: int,
            # params below configured by hydra
            data_dir: str,
            tasks_sampler_cfg,  # hydra config for ByTask sampler
            ignore_steps: int=2400,  # use init distribution for this long
            start_step: int=5000,  # when to switch off of using the init distribution
            mode: str="empirical",  # or uniform. What loss to target for the selector.
            policy_type: str="ado",  # which type of selector is used
            y_axis: str="train_loss",  # train_loss | val_loss
    ):
        self.n_procs = jax.process_count()
        self.sharding_fn = get_shard_fn(mesh, get_data_sharding(mesh))
        self.block_size = block_size
        self.g_accum_iters = g_accum_iters
        self.batch_size = batch_size

        self.tasks_sampler: TasksSampler = hydra.utils.instantiate(
            tasks_sampler_cfg, split, block_size, data_dir
        )
        self.k = self.tasks_sampler.k
        if mode == "empirical":
            task_token_count = self.tasks_sampler.total_token_n
        else:
            task_token_count = np.ones(self.k)

        if policy_type == "ado":
            self.data_policy = AdoSelector(
                n_task=self.k,
                task_token_count=task_token_count,
                block_size=block_size,
                ignore_steps=ignore_steps,
                start_step=start_step,
                y_axis=y_axis,
            )
        else:
            raise ValueError(f"Unrecognized")

        self._curr_dist0 = None  # curr_dist0 is the selector on host0's curr_dist
        self.cumulative_cnts = Counter()
        print(f"Init dist: {self.curr_dist0}.")

    @property
    def curr_dist0(self):
        if self._curr_dist0 is None:
            return self.data_policy.empirical_dist
        return self._curr_dist0

    def get_batch(self, itr: int):
        B = (self.batch_size // self.n_procs)  # per proc batch size
        BG = B * self.g_accum_iters

        task_probs_K = self.curr_dist0
        task_probs_K = np.float64(task_probs_K)
        task_probs_K /= task_probs_K.sum()
        idx = np.random.choice(np.arange(self.k), BG, p=task_probs_K, replace=True)
        cnt = Counter(idx)
        num_per_task = np.array([cnt[i] for i in range(self.k)])
        x_BGxT, y_BGxT, task_ids_BG = self.tasks_sampler.get_batch(num_per_task, BG)
        x_GxBxT = x_BGxT.reshape(self.g_accum_iters, B, self.block_size)
        y_GxBxT = y_BGxT.reshape(self.g_accum_iters, B, self.block_size)
        task_ids_GxB = task_ids_BG.reshape(self.g_accum_iters, B)
        task_probs_GxB = task_probs_K[task_ids_GxB]
        info = {"task_ids": task_ids_GxB[..., None], "p": task_probs_GxB[..., None]}
        x_GxBxT, y_GxBxT, info = jax.tree_util.tree_map(self.sharding_fn, (x_GxBxT, y_GxBxT, info))
        return x_GxBxT, y_GxBxT, info

    def update_val_loss(self, loss_np_N, taskid_np_N, step: int):
        self.data_policy.update_val_loss(loss_np_N, taskid_np_N, step)

    def update(self, loss_GxB: jax.Array, info: tp.Dict, itr: int, lr: float):
        metrics = {}
        loss_GxB, info = multihost_utils.process_allgather((loss_GxB, info))
        task_id_GxB = info["task_ids"].squeeze(-1)
        self.cumulative_cnts.update(task_id_GxB.reshape(-1).tolist())
        unique_ids = np.unique(task_id_GxB).tolist()
        for task_id in unique_ids:
            mask = task_id_GxB == task_id
            metrics[f"loss/task{task_id}"] = loss_GxB[mask].mean().item()
            metrics[f"cum_task_cnts/{task_id}"] = self.cumulative_cnts[task_id]
        self.data_policy.update(task_id_GxB, loss_GxB, itr, lr=lr, metrics=metrics)

        for i, v in enumerate(self.curr_dist0):
            metrics[f'task_prob/task{i}'] = v

        curr_dist = self.data_policy.curr_dist(itr)
        if curr_dist is None:
            return metrics

        if self.n_procs > 1:
            global_curr_dist = multihost_utils.process_allgather(curr_dist)
            prob_std = np.std(global_curr_dist, axis=0)
            for i in range(self.k):
                metrics[f'multi_host_std/task{i}'] = prob_std[i]
            self._curr_dist0 = global_curr_dist[0]
        else:
            self._curr_dist0 = curr_dist
        return metrics
