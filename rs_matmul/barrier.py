import jax
jax.default_device = jax.devices("cpu")

from jax import core
from jax.interpreters.xla import apply_primitive
from jax.tree_util import tree_flatten,tree_unflatten
import jaxlib.mlir.dialects.stablehlo as hlo
from jax._src import util
from jax._src.interpreters import mlir
from functools import partial
#print(jax.devices())

def _optimization_barrier_abstract_eval(*args):
  return args

def _optimization_barrier_lowering_rule(ctx, *args):
  barrier_types = map(mlir.aval_to_ir_types, ctx.avals_in)
  flat_args = mlir.flatten_lowering_ir_args(args)
  barrier_op = hlo.OptimizationBarrierOp(flat_args)
  return util.unflatten(barrier_op.results, map(len, barrier_types))

def _optimization_barrier(arg):
  flat_args, treedef = tree_flatten(arg)
  return tree_unflatten(treedef, optimization_barrier_p.bind(*flat_args))

optimization_barrier_p = core.Primitive('optimization_barrier')
optimization_barrier_p.multiple_results = True
optimization_barrier_p.def_impl(
    partial(apply_primitive, optimization_barrier_p))
optimization_barrier_p.def_abstract_eval(_optimization_barrier_abstract_eval)
mlir.register_lowering(optimization_barrier_p,
                       _optimization_barrier_lowering_rule)

import jax.experimental.shard_map as shard_map
shard_map.register_standard(optimization_barrier_p)  # doesn't change replication

#import jax.numpy as jnp
#def f(y, z, a):
#    d = jnp.dot(y, z)
#    d = _optimization_barrier(d)
#    acc = d + a
#    return acc
#
#y = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
#z = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
#a = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
#print(jax.jit(f).lower(y, z, a).as_text())
