import numpy as np
import jax

jtu = jax.tree_util
NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def tree_broadcast(prefix, target):
    def _broadcast(leaf, subtree):
        return jtu.tree_map(lambda _: leaf, subtree)
    return jtu.tree_map(_broadcast, prefix, target)


def reshard(tree, shardings):
    # From https://github.com/google-research/big_vision/blob/1b17abc6b754175dcd92e9db3e13c409e2ccb951/big_vision/utils.py#L1288
    def _make_global_arr(x, shard, shape):
        # Avoid unnecessary copies and transfers:
        if hasattr(x, "sharding") and x.sharding.is_equivalent_to(shard, len(shape)):
            return x
        if not getattr(x, "is_fully_addressable", True):
            raise RuntimeError("Trying to reshard a non-fully-addressable array. See link above.")
        x = jax.device_get(x)  # Might be on local devices.
        xs = [jax.device_put(x[s], device=d)
              for d, s in shard.addressable_devices_indices_map(shape).items()]
        return jax.make_array_from_single_device_arrays(shape, shard, xs)

    shapes = jax.tree_map(np.shape, tree)
    shardings = tree_broadcast(shardings, tree)
    return jax.tree_map(_make_global_arr, tree, shardings, shapes)


def get_shard_fn(mesh, sharding):
    """Shard fn for data parallelism."""
    n_procs = jax.process_count()
    def shard(x):
        local_ds = mesh.local_devices
        xs = jax.device_put(np.split(x, len(local_ds), axis=1), local_ds)
        global_shape = (x.shape[0], x.shape[1] * n_procs, *x.shape[2:])
        # each proc has its own sub-batch--"combine" them together into a jax array.
        return jax.make_array_from_single_device_arrays(global_shape, sharding, xs)
    return shard


def get_data_sharding(mesh):
    return NamedSharding(mesh, P(None, ('replica', 'data'), None))  # (G, B, D)
