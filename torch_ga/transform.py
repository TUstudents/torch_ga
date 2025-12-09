"""Transformation utilities for geometric algebra operations.

Provides functions for computing rotation planes and Jacobian transformations.
"""
from typing import List, Any, Union, Optional
import numbers
import numpy as np
import torch
# from icecream import ic


from .cayley import get_cayley_tensor, blades_from_bases
from .blades import (
    BladeKind, get_blade_of_kind_indices, get_blade_indices_from_names,
    get_blade_repr, invert_blade_indices
)
from .mv_ops import mv_multiply, mv_reversion, mv_grade_automorphism, mv_conv1d, f_mv_conv1d, mv_multiply_element_wise
from .mv import MultiVector
from .torch_ga import GeometricAlgebra
from .jacobian import get_jacobian



def compute_rotation_plane(points, referencepoint):
    """Compute rotation plane between points and reference point.
    
    Args:
        points: Target points.
        referencepoint: Reference point for rotation.
    
    Returns:
        Rotation plane as multivector.
    """   
    p0, p1 =  referencepoint , points
    lines = p0 & p1
    midpoints = 0.5*(p0 + p1)
    rotationplane = lines | midpoints 
    return rotationplane

def compute_log_det_jacobian_transformation(ga,points,referencepoint):
    """Compute log determinant of Jacobian for transformation.
    
    Args:
        ga: GeometricAlgebra instance.
        points: Points to transform.
        referencepoint: Reference point for transformation.
    
    Returns:
        Log determinant of the Jacobian matrix.
    """
    j2 = get_jacobian(lambda p1: compute_rotation_plane(ga(p1),referencepoint).tensor , points.tensor)
    J = j2.j[...,slice(1,5), slice(11,15)]
    return torch.log(torch.linalg.det(J).abs())
