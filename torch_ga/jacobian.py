"""Jacobian computation utilities for automatic differentiation.

Provides efficient batch Jacobian computation for geometric algebra functions.
"""
import torch

import einops
from collections import namedtuple
from typing import Iterable


__all__ = [
    "get_jacobian"
]

Jacobian = namedtuple("Jacobian", ["y", "j"])

from functools import partial


def get_jacobian(fun, x, m=None, slice_in=None, slice_out=None):
    """Compute the Jacobian of a function with respect to input variable.
    
    Batch evaluates the Jacobian with cost O(1) vs O(d), but has memory cost O(d).
    Only works for low dimensionality (d < 1000).
    
    Args:
        fun: Callable function for which to compute the Jacobian.
        x: Input tensor coordinates for the Jacobian evaluation.
        m: Optional output dimension.
        slice_in: Optional slice for input dimensions of the Jacobian.
        slice_out: Optional slice for output dimensions of the Jacobian.

    Returns:
        Jacobian: Named tuple containing:
            - y: Function value of fun evaluated at x.
            - j: Jacobian matrix of fun evaluated at x.
    """
    if len(x.shape)==1: x.unsqueeze(0)
    shape = x.shape[:-1]
    d = x.shape[-1]
    x = x.view(-1, d)
    n = x.shape[0]
    z = einops.repeat(x, "n j -> (n i) j", i=d)
    z.requires_grad_(True)
    y = fun(z)    
    if  m is None:
        out_grad = torch.eye(d, device=x.device, dtype=x.dtype).tile(n, 1)
    else:        
        out_grad = torch.zeros(d, m, device=x.device, dtype=x.dtype).tile(n, 1)        
        out_grad[slice_out,slice_in] = 1.
    j = torch.autograd.grad(y, z, out_grad, create_graph=True, retain_graph=True)[0].view(*shape, d, d)

    if not slice_in is None:
        j = j[...,slice_out,slice_in]
    
    return Jacobian(
        y=einops.rearrange(y, "(n i) j -> n i j", i=d)[:, 0, :].view(*shape, -1),
        j=j
    )

