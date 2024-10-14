import functools
import os
import time
import numpy as np
from jaxopt import LBFGS
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils, multihost_utils
from scipy.signal import savgol_filter
import wandb
try:
    import gcsfs
except:
    print("no gcsfs")


class AdoSelector:
    def __init__(
        self,
        n_task,
        task_token_count,
        block_size,
        ignore_steps,  # when fitting scaling laws, ignore this many steps
        start_step,  # first step to start fitting scaling laws
        update_interval=1000,
        y_axis="train_loss",  # train_loss | val_loss. TODO: reset default to train_loss and make it cfgable
        use_same_step_size=True,
        rundir = None,
    ):
        self.k = n_task
        self.block_size = block_size
        self.tokens_remaining = task_token_count
        self.empirical_dist = task_token_count / task_token_count.sum()
        self.remain_dist = self.empirical_dist
        self.ignore_steps = ignore_steps
        self.start_step = start_step
        self.y_axis = y_axis
        print(f"Scaling law selector warmup={ignore_steps}, start_step={start_step}, update_interval={update_interval}.")
        self.update_interval = update_interval
        self.use_same_step_size = use_same_step_size

        # internal state
        self.current_params = None
        self.task_cnt_TxK = np.empty((0, self.k), dtype=np.int32)
        self.task_loss_TxK = np.empty((0, self.k), dtype=np.float32)
        # S = T / eval_interval
        self.val_loss_SxK = np.empty((0, self.k), dtype=np.float32)
        # tracks the corresponding number of steps up to each val loss
        self.val_step_S = np.empty((0,), dtype=np.int32)
        self.logit = np.log(self.empirical_dist)
        self.p = self.empirical_dist  # y
        self.p_avg = self.empirical_dist  # x
        self.p_history = self.empirical_dist

        self.current_lr = 1e-3

        self.compression_so_far = 0

        local_cnt = jax.local_device_count()
        self.mesh = Mesh(mesh_utils.create_device_mesh(
            (jax.device_count() // local_cnt, local_cnt)), axis_names=('replica', 'data'))

        self.fs = None
        self.data_path = None
        if rundir is not None:
            self.fs = gcsfs.GCSFileSystem()
            self.data_path = os.path.join(rundir, "scaling_law_data.npz")
            if self.fs.exists(self.data_path):
                with self.fs.open(self.data_path, "rb") as f:
                    saved_data = np.load(f)
                    self.task_cnt_TxK = saved_data["task_cnt_TxK"]
                    self.task_loss_TxK = saved_data["task_loss_TxK"]
                    self.val_loss_SxK = saved_data["val_loss_SxK"]
                    self.val_step_S = saved_data["val_step_S"]
                    self.p_history = saved_data["p_history"]
                    self.p_avg = saved_data["p_avg"]
                print(f"\nLoaded scaling law data from {self.data_path}.")
                print(f"task_cnt_TxK shape: {self.task_cnt_TxK.shape}.")
                print(f"task_loss_TxK shape: {self.task_loss_TxK.shape}.")

                start_time = time.time()
                print(f"Refitting parameters due to reloading...")
                current_params = self.compute_scaling_law_params()
                print(f"Took {time.time() - start_time} seconds.")
                if self.check_valid(current_params):
                    self.current_params = current_params

    def curr_dist(self, itr):
        # dL/dn_i = b_i * (L_i - d_i) / n_i
        # = (Task i size) * exponent * (reducible loss) / (# of samples from task i so far)
        if itr < self.start_step or self.current_params is None:
            return None
        irred_loss_K = np.exp(self.current_params[:, 1])
        # arbitrarily bound exponent to be >= 0.05
        expnt_K = np.maximum(self.current_params[:, 2], 0.05)
        curr_loss_K = scaling_law(
            (self.current_params[:, 0], self.current_params[:, 1], self.current_params[:, 2]),
            self.task_cnt_TxK.sum(0))
        n_seen_K = np.float32(np.sum(self.task_cnt_TxK, 0))
        slope_K = expnt_K * (curr_loss_K - irred_loss_K) / n_seen_K
        loss_diff = curr_loss_K - irred_loss_K

        weighted_slope_K = slope_K
        weighted_slope_K = weighted_slope_K * self.empirical_dist * (self.p_history * (1-self.p_history))**0.5

        dist_K_instant = weighted_slope_K

        dist_K = dist_K_instant / dist_K_instant.sum()

        dist_K = clip_min_probability(dist_K, 1e-2)

        self.p = 0.9 * self.p_avg + 0.1 * dist_K
        self.p_avg = (1 - 1/(itr + 1)) * self.p_avg + 1/(itr + 1) * dist_K

        dist_K = self.p

        self._cached_curr_dist = dist_K
        self._cached_curr_dist_itr = itr
        return dist_K

    def update(self, task_ids, losses, itr, lr, metrics=None, **kwargs):
        counts_K = np.zeros(self.k, dtype=np.int32)
        unique_values, counts = np.unique(task_ids, return_counts=True)
        counts_K[unique_values] = counts
        self.tokens_remaining = self.tokens_remaining - counts_K * 1024  # TODO: hardcoded context length
        self.tokens_remaining = np.clip(self.tokens_remaining, 1, self.tokens_remaining.max())
        self.remain_dist = self.tokens_remaining / self.tokens_remaining.sum()
        if self.use_same_step_size:
            increment = np.ones_like(counts_K[None]) * counts_K.sum()
            self.task_cnt_TxK = np.append(self.task_cnt_TxK, increment, axis=0)
        else:
            self.task_cnt_TxK = np.append(self.task_cnt_TxK, counts_K[None], axis=0)

        gamma = 0.9
        count_dist = counts_K / counts_K.sum()
        self.p_history = gamma * self.p_history + (1 - gamma) * count_dist
        self.current_lr = lr

        for i in range(self.k):
            metrics[f'count_dist/task{i}'] = count_dist[i]
            metrics[f'p_history/task{i}'] = self.p_history[i]

        avg_loss = np.mean(losses).item()
        task_loss_K = np.zeros((self.k,))
        for i in range(self.k):
            # TODO(ayz): The problem is we can sample batches where some tasks are missing.
            if np.sum(task_ids == i) > 0:  # if task i is in the batch
                task_loss_K[i] = np.mean(losses[np.where(task_ids == i)])
            elif self.task_loss_TxK.shape[0] > 0:  # impute with previous value
                task_loss_K[i] = self.task_loss_TxK[-1, i]
            else:
                task_loss_K[i] = avg_loss  # use average loss

        self.compression_so_far += 1024 * counts_K * task_loss_K
        for i in range(self.k):
            # c_t * l_t + (R_t - c_t) * l_t = R_t l_t
            # d/dm R_t l_t = m_t l_t + R_t dl / dt m_t = (l_t + R_t slope history)
            per_task_compression = self.compression_so_far + self.tokens_remaining * task_loss_K
            metrics[f'compression_size/task{i}'] = per_task_compression.sum()

        self.task_loss_TxK = np.append(self.task_loss_TxK, task_loss_K[None], axis=0)

        if jax.process_index() == 0 and itr % self.update_interval == 0:
            if self.fs is not None:
                saved_data = {
                    "task_cnt_TxK": self.task_cnt_TxK,
                    "task_loss_TxK": self.task_loss_TxK,
                    "val_loss_SxK": self.val_loss_SxK,
                    "val_step_S": self.val_step_S,
                    "p_history": self.p_history,
                    "p_avg": self.p_avg,
                }
                with self.fs.open(self.data_path, "wb") as f:
                    np.savez(f, **saved_data)
                print(f"At itr {itr}, saved scaling law data to {self.data_path}.")

        if itr >= self.start_step and itr % self.update_interval == 0:
            start_time = time.time()
            print(f"Refitting parameters at step {itr}...")
            current_params = self.compute_scaling_law_params()
            print(f"Took {time.time() - start_time} seconds.")
            if self.check_valid(current_params):
                self.current_params = current_params
        if metrics is not None and self.current_params is not None:
            for i in range(self.k):
                metrics[f'param_mult/task{i}'] = np.exp(self.current_params[i, 0])
                metrics[f'param_irred/task{i}'] = np.exp(self.current_params[i, 1])
                metrics[f'param_expon/task{i}'] = self.current_params[i, 2]
                metrics[f'remaining_dist/task{i}'] = self.remain_dist[i]

    def update_val_loss(self, loss_np_N, taskid_np_N, step: int):
        self.val_step_S = np.append(self.val_step_S, step)
        val_loss_K = np.zeros((self.k,))
        for i in range(self.k):
            mask = taskid_np_N == i
            assert np.sum(mask) > 0
            val_loss_K[i] = np.mean(loss_np_N[mask])
        self.val_loss_SxK = np.append(self.val_loss_SxK, val_loss_K[None], axis=0)

    def check_valid(self, params):
        nans = np.isnan(np.sum(params)).item()
        infs = np.isinf(np.sum(params)).item()
        valid = not (nans or infs)
        if not valid:
            print(f"Found: nans={nans}, infs={infs}.")
            print(params)
        return valid

    def _make_global_array(self, arr, sharding):
        global_shape = arr.shape
        arrays = [
        jax.device_put(arr[index], d)
            for d, index in sharding.addressable_devices_indices_map(global_shape).items()]
        return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)

    def compute_scaling_law_params(self, verbose=True):
        cum_cnt_TxK = np.cumsum(self.task_cnt_TxK, 0)
        if self.y_axis == "train_loss":
            print("Fitting scaling law on train loss.")
            cum_cnt_KxT = jnp.asarray(cum_cnt_TxK[self.ignore_steps::10].T)
            # Opt loss is noisy, filter
            task_loss_KxT = jnp.asarray(savgol_filter(
                self.task_loss_TxK.T, 101, 3, axis=-1)[:, self.ignore_steps::10])
        else:
            print(f"Fitting scaling law on val loss up to step {self.val_step_S[-1]}.")
            mask = self.val_step_S >= self.ignore_steps
            val_step_S = self.val_step_S[mask]
            cum_cnt_KxT = jnp.asarray(cum_cnt_TxK[val_step_S, :].T)
            task_loss_KxT = jnp.asarray(self.val_loss_SxK[mask].T)
            print(f"cum_cnt_KxT shape: {cum_cnt_KxT.shape}, task_loss_KxT shape: {task_loss_KxT.shape}.")
        cum_cnt_KxT, task_loss_KxT = multihost_utils.broadcast_one_to_all(
            (cum_cnt_KxT, task_loss_KxT))

        sharding = jax.sharding.NamedSharding(self.mesh, P())
        cum_cnt_KxT = self._make_global_array(cum_cnt_KxT, sharding)
        task_loss_KxT = self._make_global_array(task_loss_KxT, sharding)

        fit_fn = functools.partial(fit_scaling_law, mesh=self.mesh)
        params_Kx3, _state = jax.vmap(fit_fn, in_axes=(0, 0))(cum_cnt_KxT, task_loss_KxT)
        params_np_Kx3 = multihost_utils.process_allgather(params_Kx3)
        return params_np_Kx3


def clip_min_probability(probs, min_prob):
    total_deficit = max(min_prob * len(probs) - probs.sum(), 0)
    scale_factor = (1 - total_deficit) / probs.sum()
    scaled_probs = probs * scale_factor
    clipped_probs = np.maximum(scaled_probs, min_prob)
    return clipped_probs / clipped_probs.sum()


def huber_loss(target, pred, delta=1e-3):
    abs_diff = jnp.abs(target - pred)
    return jnp.where(abs_diff > delta,
                   delta * (abs_diff - .5 * delta),
                   0.5 * abs_diff ** 2)


def logsumexp(a, axis=None):
    a_max = jnp.max(a, axis=axis, keepdims=True)
    a_shifted = a - a_max  # improves numerical stability
    exp_a_shifted = jnp.exp(a_shifted)
    sum_exp_a_shifted = jnp.sum(exp_a_shifted, axis=axis, keepdims=True)
    log_sum_exp = jnp.log(sum_exp_a_shifted) + a_max  # undo the shift

    if axis is not None:
        return jnp.squeeze(log_sum_exp, axis=axis)
    else:
        return log_sum_exp


def loss_fn(params_3, D: float, L: float):
    a, e, alpha = params_3
    arg1 = a - alpha * jnp.log(D)
    arg2 = e + jnp.zeros(arg1.shape)

    pred = logsumexp(jnp.stack((arg1, arg2), axis=0), axis=0)
    target = jnp.log(L)
    alpha_term = jnp.maximum(alpha - 0.8, 0)  # prevent alpha > 0.8
    alpha_term2 = - jnp.minimum(alpha, 0.001)  # prevent alpha < 0
    a_term = jnp.maximum(a - 6.5, 0)
    e_term = -jnp.minimum(e - 0.5, 0)  # prevent e < 0.5
    return jnp.sum(huber_loss(target, pred)) + alpha_term + alpha_term2 + e_term + a_term


def scaling_law(params, D):
    a, e, alpha = params
    return np.exp(e) + np.exp(a)/D**alpha


@functools.partial(jax.jit, static_argnums=(2,))
def fit_scaling_law(D_T, losses_T, mesh):
    init_a = jnp.arange(-2, 6, 1)
    init_e = jnp.arange(-2, 2, 0.5)
    init_alpha = jnp.arange(0, 0.8, 0.1)

    X, Y, Z = jnp.meshgrid(init_a, init_e, init_alpha, indexing='ij')
    init_grid_Gx3 = jnp.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    sharding = NamedSharding(mesh, P(('replica', 'data'), None))
    init_grid_Gx3 = jax.lax.with_sharding_constraint(init_grid_Gx3, sharding)

    solver = LBFGS(loss_fn, tol=1e-5, jit=True, maxiter=200)
    run_fn = jax.vmap(solver.run, in_axes=(0, None, None))
    result_Gx3, state = run_fn(init_grid_Gx3, D_T, losses_T)
    return result_Gx3[jnp.nanargmin(state.value)], state