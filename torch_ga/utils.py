"""Utility functions to convert from vector to GA representation for G(0,1,1,1)."""


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


# Probably the primary encoding
# 

def plane2pga(ga, distance, normal_dir):
    """Convert plane with normal direction n in R3 and origin distance d to PGA."""
    inputs = torch.cat([distance,normal_dir],dim=-1)
    blade_indices = (ga.blade_degrees==1).long()
    blade_indices = torch.where(blade_indices)[0]
    tensor = ga.from_tensor(inputs, blade_indices)
    return tensor

def pga2plane(ga,tensor):
    """Convert PGA tensor to plane representation.
    
    Returns:
        Tuple of (distance, normal_dir).
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    blade_indices = (ga.blade_degrees==1).long()
    blade_indices = torch.where(blade_indices)[0]
    dn = tensor[...,blade_indices]
    distance,normal_dir = dn[...,:1],dn[...,1:]
    return distance,normal_dir

def line2pga(ga, orthogonal_shift, direction):
    """Convert line with direction n in R3 and orthogonal shift s to PGA."""
    inputs = torch.cat([orthogonal_shift,direction],dim=-1)
    blade_indices = (ga.blade_degrees==ga.p-1).long()
    blade_indices = torch.where(blade_indices)[0]
    tensor = ga.from_tensor(inputs, blade_indices)
    return tensor

def pga2line_(ga,tensor):
    """Convert PGA tensor to line representation (internal version).
    
    Returns:
        Tuple of (orthogonal_shift, direction).
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    blade_indices = (ga.blade_degrees==ga.p-1).long()
    blade_indices = torch.where(blade_indices)[0]
    # ic(blade_indices,tensor.shape)
    od = tensor[...,blade_indices]
    orthogonal_shift, direction = od[...,:ga.p],od[...,ga.p:]
    return orthogonal_shift, direction

def pga2line(ga,tensor):
    """Convert PGA tensor to line representation.
    
    Returns:
        Tuple of (points, direction).
    """
    if not isinstance(tensor, MultiVector): 
        lines = ga(tensor)
    else:
        lines = tensor
    
    ORIG = ga(ga.e123 if ga.p==3 else ga.e12)
    d = (lines|ORIG)*lines
    points = pga2point(ga,d)
    # ic(points)
    orthogonal_shift, direction = pga2line_(ga,tensor)
    return points, direction
    
    
    # if isinstance(tensor, MultiVector): tensor = tensor.tensor
    # blade_indices = (ga.blade_degrees==2).long()
    # blade_indices = torch.where(blade_indices)[0]
    # od = tensor[...,blade_indices]
    # orthogonal_shift, direction = od[...,:3],od[...,3:]
    # return orthogonal_shift, direction

def point2pga(ga, point):
    """Convert point p in R3 to PGA representation."""
    inputs = torch.cat([point,torch.ones([*point.shape[:-1],1]).to(point)],dim=-1)
    blade_indices = (ga.blade_degrees==ga.p).long()
    blade_indices = torch.where(blade_indices)[0]
    tensor = ga.from_tensor(inputs, blade_indices)
    # /inputs.tensor[...,-2]
    return tensor

def pga2point(ga,tensor):
    """Convert PGA tensor to point in R3.
    
    Returns:
        Point coordinates.
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    blade_indices = (ga.blade_degrees==ga.p).long()
    blade_indices = torch.where(blade_indices)[0]
    od = tensor[...,blade_indices]
    point, one_ = od[...,:ga.p],od[...,ga.p:]
    if True: point = point/one_
    return point

# CGA

def point2cga(ga, point):
    """Convert point p in R3 to CGA representation."""
    inputs = torch.cat([point,torch.ones([*point.shape[:-1],1]).to(point)],dim=-1)
    blade_indices = torch.tensor([5,12,14,15])
    tensor = ga.from_tensor(inputs, blade_indices)
    # /inputs.tensor[...,-2]
    return tensor

def cga2point(ga,tensor):
    """Convert CGA tensor to point in R3.
    
    Returns:
        Point coordinates.
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    blade_indices = torch.tensor([5,12,14,15])
    od = tensor[...,blade_indices]
    point, one_ = od[...,:3],od[...,3:]
    if True: point = point/one_
    return point



# Direction
# 415,425,435,
# -21,-24,-25

# Moment
# 235,315,125
# 23,-20,18

def line2cga(ga, moment, direction):
    """Convert line with direction n in R3 and moment to CGA representation."""
    scaling = torch.tensor([1,-1,1]).to(moment)
    inputs = torch.cat([moment*scaling,-direction],dim=-1)
    blade_indices = torch.tensor([23,20,18,21,24,25])
    tensor = ga.from_tensor(inputs, blade_indices)
    return tensor

def cga2line_(ga,tensor):
    """Convert CGA tensor to line representation (internal version).
    
    Returns:
        Tuple of (moment, direction).
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    blade_indices = torch.tensor([23,20,18,21,24,25])
    od = tensor[...,blade_indices]
    moment, direction = od[...,:3],-od[...,3:]
    scaling = torch.tensor([1,-1,1]).to(moment)
    return moment*scaling, direction

def cga2line(ga,tensor):
    """Convert CGA tensor to line representation.
    
    Returns:
        Tuple of (points, direction).
    """
    if not isinstance(tensor, MultiVector): 
        lines = ga(tensor)
    else:
        lines = tensor
    
    ORIG = ga(ga.e3 - ga.e4)
    d = (lines|ORIG)*lines
    points = cga2point(ga,d)
    moment, direction = cga2line_(ga,tensor)
    return points, direction
    
    

# Dual encoding
# 

def dual_plane2pga(ga, distance, normal):
    """Convert plane to PGA using dual encoding.
    
    Args:
        ga: GeometricAlgebra instance.
        distance: Distance parameter.
        normal: Normal vector.
    
    Returns:
        PGA tensor.
    """
    n = ga(ga.from_tensor(normal,torch.tensor([2,3,4])))
    delta = ga(ga.from_tensor(distance,torch.tensor([0])))
    tensor = n - delta*ga(ga.e0)
    # tensor = n - ga(distance*ga.e0)
    return tensor.tensor

def dual_pga2plane(ga,tensor):
    """Convert PGA tensor to plane using dual encoding.
    
    Returns:
        Tuple of (distance, normal).
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    normal, distance = tensor[...,2:5],-tensor[...,1]
    return distance, normal

def dual_line2pga(ga, orthogonal_shift, direction):
    """Convert line to PGA using dual encoding."""
    n1n2 = ga(ga.from_tensor(direction,torch.tensor([2,3,4])))
    distance = ga(ga.from_tensor(orthogonal_shift,torch.tensor([2,3,4])))
    tensor = n1n2 - ga(ga.e0)*distance*n1n2
    return tensor.tensor

def dual_pga2line(ga,tensor):
    """Convert PGA tensor to line using dual encoding.
    
    Returns:
        Tuple of (orthogonal_shift, direction).
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    n1n2 = ga(ga.from_tensor(tensor[...,2:5],torch.tensor([2,3,4])))
    tensor = ga(tensor) - n1n2  
    gad = GeometricAlgebra([1]*ga.p+[-1]*ga.q)
    dn1n2 = gad(gad.from_tensor(tensor.tensor[...,[1,-5,-4,-3]],torch.tensor([0,4,5,6])))
    n1n2 = gad(gad.from_tensor(n1n2.tensor[...,[2,3,4]],torch.tensor([1,2,3])))
    distance = dn1n2/n1n2
    orthogonal_shift, direction = -distance.tensor[...,[1,2,3]], n1n2.tensor[...,[1,2,3]]
    return orthogonal_shift, direction

def dual_point2pga(ga, point):
    """Convert point p in R3 to PGA using dual encoding."""
    x = ga(ga.from_tensor(point,torch.tensor([2,3,4])))
    tensor = (ga(ga.e_)-ga(ga.e0)*x)*ga(ga.e123)
    return tensor.tensor

def dual_pga2point(ga,tensor):
    """Convert PGA tensor to point using dual encoding.
    
    Returns:
        Point coordinates.
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    point = ga(tensor).dual()
    tensor = point.tensor[...,2:5]
    return tensor


# Need to be verified!!!
# 
# 

# Operations
def reflection2pga(ga, distance, normal_dir):
    """Convert reflection through plane to PGA representation.
    
    Args:
        ga: GeometricAlgebra instance.
        distance: Origin shift d in R.
        normal_dir: Normal n in R3.
    
    Returns:
        PGA tensor.
    """
    inputs = torch.cat([distance,normal_dir],dim=-1)
    blade_indices = (ga.blade_degrees==1).long()
    blade_indices = torch.where(blade_indices)[0]
    tensor = ga.from_tensor(inputs, blade_indices)
    return tensor

def pga2reflection(ga,tensor):
    """Convert PGA tensor to reflection representation.
    
    Returns:
        Tuple of (distance, normal_dir).
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    blade_indices = (ga.blade_degrees==1).long()
    blade_indices = torch.where(blade_indices)[0]
    dn = tensor[...,blade_indices]
    distance,normal_dir = dn[...,:1],dn[...,1:]
    return distance,normal_dir

# translation 
def translation2pga(ga, translation):
    """Convert translation t in R3 to PGA representation."""
    _one = torch.ones([*translation.shape[:-1],1]).to(translation)
    inputs = torch.cat([_one,translation*0.5],dim=-1)
    blade_indices = (ga.blade_degrees==2).long() 
    blade_indices = torch.where(blade_indices)[0][:3]
    tensor = ga.from_tensor(inputs, [0]+blade_indices)
    return tensor

def pga2translation(ga,tensor):
    """Convert PGA tensor to translation vector.
    
    Returns:
        Translation vector.
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    blade_indices = (ga.blade_degrees==2).long()
    blade_indices = torch.where(blade_indices)[0]
    od = tensor[...,blade_indices[:3]]
    _one, translation = od[...,:3],od[...,3:]
    return translation*2

# rotation 
def rotation2pga(ga, rotation):
    """Convert rotation to PGA representation."""
    
    inputs = rotation
    blade_indices = (ga.blade_degrees==2).long() 
    blade_indices = torch.where(blade_indices)[0][3:]
    tensor = ga.from_tensor(inputs, [0]+blade_indices)
    return tensor

def pga2rotation(ga,tensor):
    """Convert PGA tensor to rotation representation.
    
    Returns:
        Rotation parameters.
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    blade_indices = (ga.blade_degrees==2).long()
    blade_indices = torch.where(blade_indices)[0]
    rotation = tensor[...,[0]+blade_indices[3:]]
    return rotation

# Point reflection
def pointreflection2pga(ga, point):
    """Convert point reflection through p in R3 to PGA representation."""
    inputs = torch.cat([point,torch.ones([*point.shape[:-1],1]).to(point)],dim=-1)
    blade_indices = (ga.blade_degrees==3).long()
    blade_indices = torch.where(blade_indices)[0]
    tensor = ga.from_tensor(inputs, blade_indices)
    return tensor

def pga2pointreflection(ga,tensor):
    """Convert PGA tensor to point reflection.
    
    Returns:
        Point coordinates.
    """
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    blade_indices = (ga.blade_degrees==3).long()
    blade_indices = torch.where(blade_indices)[0]
    od = tensor[...,blade_indices]
    point, one_ = od[...,:3],od[...,3:]
    if False: point = point/one_
    return point

