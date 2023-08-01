import argparse
import numpy
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.shard_map import shard_map
from jax.experimental.pjit import pjit
from flax.linen import partitioning as nn_partitioning

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_async_collective_permute=true --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_highest_priority_async_stream=true'
#os.environ['XLA_FLAGS'] += ' --xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=/results/hlo/rs_matmul_dp8tp1'

from .barrier import _optimization_barrier

with_sharding_constraint = nn_partitioning.with_sharding_constraint

def rs_matmul(lhs, rhs):
    axis_size = jax.lax.psum(1, axis_name='tp')
    axis_idx = jax.lax.axis_index('tp')
    lhs = lhs.reshape((lhs.shape[0], axis_size, lhs.shape[1]//axis_size, lhs.shape[2]))
    def collective_matmul(i, out):
        out = jax.lax.ppermute(
            out,
            'tp',
            perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
        lhs_idx = (axis_idx - i - 1) % axis_size
        update = lhs[:, lhs_idx, ...] @ rhs
        update = _optimization_barrier(update)
        out = out + update
        return out

    lhs_idx = (axis_idx - 1) % axis_size
    out = lhs[:, lhs_idx, ...] @ rhs
    out = jax.lax.fori_loop(
        1, axis_size, collective_matmul, out)

    return out

def main():
    parser = argparse.ArgumentParser(description='Matmul overlap with all-gather communication')
    parser.add_argument("--dp", dest="dp", type=int, default=8)
    parser.add_argument("--tp", dest="tp", type=int, default=1)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=2)
    parser.add_argument("--seq_len", dest="seq_len", type=int, default=2048)
    parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=12288)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    assert(args.dp * args.tp == len(list(jax.devices())))
    assert(args.seq_len % args.tp == 0)
    assert(args.hidden_size % args.tp == 0)
    args.batch_size = args.batch_size * args.dp

    dtype = jnp.bfloat16
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key, 2)
    input = jax.random.uniform(key2, (args.batch_size, args.seq_len, 4*args.hidden_size), dtype=dtype)
    weight = jax.random.uniform(key2, (4*args.hidden_size, args.hidden_size), dtype=dtype)

    mesh_shape = {'dp': args.dp, 'tp': args.tp}
    in_specs = (PartitionSpec('dp', None, 'tp'), PartitionSpec('tp', None))
    out_specs = PartitionSpec('dp', 'tp', None)

    mesh =  Mesh(numpy.array(jax.devices()).reshape(tuple(mesh_shape.values())), tuple(mesh_shape.keys()))
    logical_axis_rules = (('batch', 'dp'), ('seq_rs', 'tp'), ('seq_ag', None), ('emb', None), ('mlp', 'tp'))
    pjitted_rs_matmul = pjit(shard_map(rs_matmul, mesh, in_specs=in_specs, out_specs=out_specs))

    if args.profile:
        import ctypes
        libcudart = ctypes.cdll.LoadLibrary('libcudart.so')
        with mesh, nn_partitioning.axis_rules(logical_axis_rules):
            input = with_sharding_constraint(input, ('batch', 'seq_ag', 'mlp'))
            weight = with_sharding_constraint(weight, ('mlp', 'emb'))
            for i in range(100):
                if i == 9:
                    libcudart.cudaProfilerStart()
                out = pjitted_rs_matmul(input, weight)
            libcudart.cudaProfilerStop()
    else:
        with mesh, nn_partitioning.axis_rules(logical_axis_rules):
            input = with_sharding_constraint(input, ('batch', 'seq_ag', 'mlp'))
            weight = with_sharding_constraint(weight, ('mlp', 'emb'))
            for i in range(100):
                out = pjitted_rs_matmul(input, weight)

    return out

if __name__ == "__main__":
    main()
