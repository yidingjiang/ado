import time
import typing as tp
from functools import partial
import numpy as np
import hydra
import equinox as eqx
import jax
from jax.experimental import mesh_utils, multihost_utils
import optax
import orbax.checkpoint as ocp
import wandb
from tqdm import trange
from .model import GPT, shard_gpt, count_params
from .sharding import reshard, get_shard_fn, get_data_sharding
from src.dataloader.abstract import Loader
from src.dataloader import pile_loader 

try:
    import lm_eval
    from src.eval.lm_harness_eval import LMEvaluator
except ImportError:
    lm_eval = None
    LMEvaluator = None
    print("lm_eval not installed, will not be able to evaluate on language model harness.")

jax.config.update("jax_threefry_partitionable", True)

jnp, jrandom, vmap, scan, jtu = jax.numpy, jax.random, jax.vmap, jax.lax.scan, jax.tree_util
Array = jax.Array
KeyArray = tp.Any
Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
P, with_sharding_constraint = jax.sharding.PartitionSpec, jax.lax.with_sharding_constraint


def cast_pytree(pytree: tp.Any, dtype: jnp.dtype) -> tp.Any:
    """Cast a pytree of arrays to a given dtype, ignore non-arrays."""
    def cast(x):
        if eqx.is_array(x):
            return x.astype(dtype)
        return x
    return jtu.tree_map(cast, pytree)


def make_training_fns(
        config, optimizer: optax.GradientTransformationExtraArgs,
        mesh: Mesh) -> tp.Tuple[tp.Callable, tp.Callable]:
    def loss_fn(model_params: GPT, model_static: GPT, x_BxT: Array, y_BxT: Array, key: tp.Optional[KeyArray]) -> Array:
        model = eqx.combine(model_params, model_static)
        if key is not None:
            key = jrandom.split(key, x_BxT.shape[0])
        logits_BxTxV = vmap(model)(x_BxT, key=key).astype(jnp.float32)
        vector_loss_BxT = optax.softmax_cross_entropy_with_integer_labels(logits_BxTxV, y_BxT)
        vector_loss_B = vector_loss_BxT.mean(axis=-1)
        return vector_loss_B.mean(), vector_loss_BxT.mean(axis=-1)

    @partial(eqx.filter_jit, donate='all')
    def step(model: GPT, opt_state, x_GxBxT: Array, y_GxBxT: Array, key: KeyArray):
        G = config.g_accum_iters
        params, static = eqx.partition((model), eqx.is_array)
        params_cpt = cast_pytree(params, jnp.dtype(config.compute_dtype))
        # compute loss and grad on microbatch, then scan over microbatches
        def microstep(grad_so_far, xykey_g: tp.Tuple[Array, Array, Array, KeyArray]):
            grad, loss_vector = jax.grad(loss_fn, has_aux=True)(params_cpt, static, *xykey_g)
            grad = shard_gpt(grad, mesh, config.shard_model)
            grad_so_far = jtu.tree_map(lambda x, y: x + y, grad, grad_so_far)
            return grad_so_far, loss_vector
        all_keys = jrandom.split(key, config.g_accum_iters)
        init_grad = jtu.tree_map(jnp.zeros_like, params)
        grad, loss_GxB = scan(microstep, init_grad, (x_GxBxT, y_GxBxT, all_keys))
        # Grad accumulated (summed) over G, so divide.
        loss, grad = jnp.mean(loss_GxB), jtu.tree_map(lambda x: x / G, grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        model = eqx.combine(optax.apply_updates(params, updates), static)
        return model, opt_state, loss, loss_GxB

    @eqx.filter_jit
    def simple_loss(model: tp.Union[GPT, eqx.Partial], x_1xBxD: Array, y_1xBxD: Array, key: tp.Optional[KeyArray]) -> Array:
        """Same as loss_fn, but doesn't split params into compute/static."""
        x_BxD, y_BxD = x_1xBxD.squeeze(0), y_1xBxD.squeeze(0)
        model_params, model_static = eqx.partition(model, eqx.is_array)
        return loss_fn(model_params, model_static, x_BxD, y_BxD, key)

    sharding_fn = get_shard_fn(mesh, get_data_sharding(mesh))
    def evaluate(model: GPT, dset) -> tp.Dict[int, float]:
        start_time = time.time()
        eval_model = eqx.Partial(cast_pytree(model, jnp.dtype(config.compute_dtype)), inference=True)
        metrics = {}
        loss_np_N = np.zeros((0,), dtype=np.float64)
        taskid_np_N = np.zeros((0,), dtype=np.int32)
        # TODO: In theory, at block size 1024 there are 332621 examples in The Pile's val dataset
        # so we should get 162 batches of 2048. But dataset sharding is slightly uneven, and worse
        # this leads to a hang because some workers' datasets exhaust before others. So we stop at
        # 150 for all workers for now.
        first_seq = None
        for i, (taskid_B, seq_BxDp1) in enumerate(dset.take(150).as_numpy_iterator()):
            if first_seq is None:
                first_seq = seq_BxDp1
            x_1xBxD, y_1xBxD = seq_BxDp1[:, :-1][None], seq_BxDp1[:, 1:][None]
            x_1xBxD, y_1xBxD, taskid_1xB = jax.tree_util.tree_map(
                sharding_fn, (x_1xBxD, y_1xBxD, taskid_B[None]))
            loss, loss_B = simple_loss(eval_model, x_1xBxD, y_1xBxD, None)
            loss_np_B, taskid_np_1xB = multihost_utils.process_allgather(
                (loss_B, taskid_1xB))
            loss_np_N = np.concatenate((loss_np_N, loss_np_B), axis=0)
            taskid_np_N = np.concatenate((taskid_np_N, taskid_np_1xB.squeeze(0)), axis=0)
        metrics["empirical_loss"] = loss_np_N.mean()
        tasks = np.unique(taskid_np_N).tolist()
        losses = []
        for task in tasks:
            mask = taskid_np_N == task
            task_loss = loss_np_N[mask].mean()
            metrics[f"loss/task{task}"] = task_loss
            losses.append(task_loss)
        metrics["balanced_loss"] = np.mean(losses)
        metrics["eval_time"] = time.time() - start_time
        metrics["tot_val_exs"] = loss_np_N.shape[0]
        return loss_np_N, taskid_np_N, metrics

    def eval_llm_harness(model: GPT, limit: tp.Optional[int]=None):
        evaluator = LMEvaluator(model, batch_size=1024)
        task_manager = lm_eval.tasks.TaskManager()
        tasks = ["arc_easy"]
        results = lm_eval.simple_evaluate(
            model=evaluator,
            tasks=tasks,
            num_fewshot=0,
            task_manager=task_manager,
            limit=limit
        )
        final_accuracy, final_std = dict(), dict()
        for task_name in tasks:
            cnt = 0
            total_acc, total_variance = 0, 0
            for res in results:
                if task_name in res:
                    total_acc += results[res]["acc,none"]
                    total_variance += results[res]["acc_stderr,none"]**2
                    cnt += 1
            final_accuracy[task_name] = total_acc / cnt
            final_std[task_name] = jnp.sqrt(total_variance) / cnt
        return {"accuracy": final_accuracy, "std": final_std}

    return step, evaluate, eval_llm_harness


def train(config):
    n_devices = jax.device_count()
    local_cnt = jax.local_device_count()
    assert local_cnt in {4, 8}, f"Unexpected local device count: {local_cnt}."
    mesh = Mesh(mesh_utils.create_device_mesh((n_devices // local_cnt, local_cnt)), axis_names=('replica', 'data'))

    val_dset = pile_loader.make_eval_dset(
        config.batch_size, config.block_size, data_dir=config.loader.data_dir)

    if not config.debug:
        options = ocp.CheckpointManagerOptions(
            max_to_keep=1, save_interval_steps=config.eval_interval)
        mngr = ocp.CheckpointManager(
            config.rundir,
            ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
            options=options)

    scheduler = optax.warmup_cosine_decay_schedule(
        0, config.learning_rate, config.warmup_steps, config.lr_decay_steps,
        end_value=config.min_lr)
    @jax.jit
    def get_lr(_opt_state):
        return scheduler(_opt_state[3].count)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=config.beta2),
        optax.add_decayed_weights(config.weight_decay / config.learning_rate),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )
    step, evaluate, eval_llm_harness = make_training_fns(config, optimizer, mesh)

    key = jrandom.PRNGKey(0)
    def init_model(model_key):
        model = GPT(config.model, model_key)
        model = cast_pytree(model, config.param_dtype)
        model = shard_gpt(model, mesh, config.shard_model)
        return model
    key, key1 = jrandom.split(key)
    # Use jit with sharding constraints to init sharded model+opt.
    model= eqx.filter_jit(init_model)(key1)
    print(f'Model has {count_params(model)} parameters.')
    def repl_opt_scalars(x: Array):
        if x.ndim == 0:
            x = reshard(x, NamedSharding(mesh, P()))
        return x
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = jtu.tree_map(repl_opt_scalars, opt_state)
    first_step = 0
    if not config.debug and mngr.latest_step() is not None:  # Restore existing checkpoint.
        ex_state = (jtu.tree_leaves(model), jtu.tree_leaves(opt_state))
        ex_shardings = jtu.tree_map(lambda x: x.sharding if eqx.is_array(x) else None, ex_state)
        restore_args = ocp.checkpoint_utils.construct_restore_args(ex_state, ex_shardings)
        model_leaves, opt_state_leaves = mngr.restore(
            mngr.latest_step(), restore_kwargs={'restore_args': restore_args})
        model = jtu.tree_unflatten(jtu.tree_structure(model), model_leaves)
        opt_state = jtu.tree_unflatten(jtu.tree_structure(opt_state), opt_state_leaves)
        first_step = mngr.latest_step() + 1
    train_sampler: Loader = hydra.utils.instantiate(
        config.loader, 
        batch_size=config.batch_size, 
        split="train", 
        mesh=mesh, 
        block_size=config.block_size,
        g_accum_iters=config.g_accum_iters)
    pbar = trange(
        first_step, config.max_steps, initial=first_step, total=config.max_steps,
        disable=jax.process_index() != 0)
    for itr in pbar:
        metrics = {}  # values to display in the progress bar
        if itr % config.eval_interval == 0:
            if config.eval_harness:
                eval_out = eval_llm_harness(model, limit=1000)
                eval_acc, eval_std = eval_out["accuracy"], eval_out["std"]
                for task in eval_acc:
                    metrics[f'lm_harness_accuracy/{task}'] = eval_acc[task]
                    metrics[f'lm_harness_std/{task}'] = eval_std[task]
            loss_np_N, taskid_np_N, val_metrics = evaluate(model, val_dset)
            if hasattr(train_sampler, 'data_policy') and hasattr(train_sampler.data_policy, 'update_val_loss'):
                train_sampler.update_val_loss(loss_np_N, taskid_np_N, itr)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        key, key1 = jrandom.split(key)
        # sample data
        x_GxBxD, y_GxBxD, info = train_sampler.get_batch(itr)
        if jax.process_index() == 0 and itr == 1:
            jax.profiler.start_trace(wandb.run.dir)
        # step and update
        model, opt_state, loss, loss_GxB = step(model, opt_state, x_GxBxD, y_GxBxD, key1)
        sampler_metrics = train_sampler.update(loss_GxB, info, itr, lr=get_lr(opt_state).item())
        metrics.update({f"opt_{k}": v for k, v in sampler_metrics.items()})
        if not config.debug:
            mngr.save(itr, (jtu.tree_leaves(model), jtu.tree_leaves(opt_state)))
        metrics['loss/opt'] = loss.item()
        metrics['lr'] = get_lr(opt_state).item()
        if jax.process_index() == 0 and itr % 20 == 0:
            if pbar.format_dict['rate'] is not None:
                tok_per_step = config.batch_size * config.g_accum_iters * config.block_size
                metrics['perf/tps'] = pbar.format_dict['rate'] * tok_per_step
                flops_per_step = (6 * count_params(model) + 12 * config.model.n_layer * config.model.n_embd * config.block_size) * (
                    config.block_size * config.batch_size * config.g_accum_iters)
                metrics['perf/mfu'] = flops_per_step * pbar.format_dict['rate'] / (197e12 * n_devices)  # hardcoded for TPU v4.
            wandb.log(metrics, step=itr)
        pbar.set_postfix(**metrics)
        if jax.process_index() == 0 and itr == 5:
            jax.profiler.stop_trace()
    pbar.close()
    if not config.debug:
        mngr.wait_until_finished()
