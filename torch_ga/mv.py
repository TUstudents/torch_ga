"""Defines the MultiVector class for convenient geometric algebra operations.

The MultiVector class wraps GeometricAlgebra operations in a more Pythonic interface using operator overloading.
"""
from __future__ import annotations
from typing import Union

from .blades import BladeKind

import torch
# from .torch_ga import GeometricAlgebra
import torch_ga

# from icecream import ic
class MultiVector:
    """Wrapper for geometric algebra tensors using GeometricAlgebra operations.
    
    Provides a less verbose interface to GeometricAlgebra operations using operators.
    """

    def __init__(self, blade_values: torch.Tensor, algebra: torch_ga.GeometricAlgebra):
        """Initialize a MultiVector from a geometric algebra tensor.
        
        Args:
            blade_values: Geometric algebra torch.Tensor with as many elements
                on its last axis as blades in the algebra.
            algebra: GeometricAlgebra instance corresponding to the  geometric
                algebra tensor.
        """
        self._blade_values = blade_values
        self._algebra = algebra

    def split(self):
        """Return blades split by grade."""
        return self._algebra.split(self._blade_values)

    @property
    def blades(self):
        """Return list of blade names."""
        return self._algebra.blades

    @property
    def blades_numbers(self):
        """Return blade counts for each grade."""
        return self._algebra.blades_numbers

    @property
    def scalar(self):
        """Return scalar part of multivector."""
        return self._blade_values[...,0]
    @property
    def pseudo_scalar(self):
        """Return pseudoscalar part of multivector."""
        return self._blade_values[...,-1]
    @property
    def vector(self):
        """Return vector part of multivector."""
        return self._blade_values[...,1:self.algebra.dim+1]
    @property
    def pseudo_vector(self):
        """Return pseudovector part of multivector."""
        return self._blade_values[...,-self.algebra.dim-1:-1]

    @property
    def bivector(self):
        """Return bivector part of multivector."""         
        return self._blade_values[...,self.algebra.blade_degrees==2]
    @property
    def pseudo_bivector(self):
        """Return pseudo-bivector part of multivector."""
        return self._blade_values[...,self.algebra.blade_degrees==(self.algebra.dim-2)]

    @property
    def trivector(self):
        """Return trivector part of multivector."""        
        return self._blade_values[...,self.algebra.blade_degrees==3]
    @property
    def pseudo_trivector(self):
        """Return pseudo-trivector part of multivector."""
        return self._blade_values[...,self.algebra.blade_degrees==(self.algebra.dim-3)]
        
    @property
    def tensor(self):
        """Geometric algebra tensor holding the values of this multivector."""
        return self._blade_values

    @property
    def algebra(self):
        """`GeometricAlgebra` instance this multivector belongs to."""
        return self._algebra

    @property
    def batch_shape(self):
        """Return batch shape of the multivector.
        
        The shape of all axes except for the last one in the geometric algebra tensor.
        """
        return self._blade_values.shape[:-1]

    def __len__(self) -> int:
        """Return number of elements on the first axis.
        
        Number of elements on the first axis of the geometric algebra tensor.
        """
        return self._blade_values.shape[0]

    def __iter__(self):
        """Iterate over multivector elements."""
        for n in range(self._blade_values.shape[0]):
            # If we only have one axis left, return the
            # actual numbers, otherwise return a new
            # multivector.
            if self._blade_values.shape.ndims == 1:
                yield self._blade_values[n]
            else:
                yield MultiVector(
                    self._blade_values[n],
                    self._algebra
                )

    def __lshift__(self, other: MultiVector) -> MultiVector:
        """Left contraction operator."""
        assert isinstance(other, MultiVector), f"{type(other)}"
        return MultiVector(
            self._algebra.right_contraction(self._blade_values, other._blade_values),
            self._algebra
        )
    def __rshift__(self, other: MultiVector) -> MultiVector:
        """Right contraction operator."""
        assert isinstance(other, MultiVector), f"{type(other)}"
        return MultiVector(
            self._algebra.right_contraction(self._blade_values, other._blade_values),
            self._algebra
        )
    

    def __xor__(self, other: MultiVector) -> MultiVector:
        """Exterior product. See GeometricAlgebra.ext_prod()."""
        assert isinstance(other, MultiVector), f"{type(other)}"

        return MultiVector(
            self._algebra.ext_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __or__(self, other: MultiVector) -> MultiVector:
        """Inner product. See GeometricAlgebra.inner_prod()."""
        assert isinstance(other, MultiVector), f"{type(other)}"

        # return MultiVector(
        #     self._algebra.inner_prod(self._blade_values, other._blade_values),
        #     self._algebra
        # )
        return MultiVector(self._algebra.inner_prod(self._blade_values, other._blade_values),self._algebra)

    def __mul__(self, other: MultiVector) -> MultiVector:
        """Geometric product. See GeometricAlgebra.geom_prod()."""
        if type(other) in (int, float, torch.Tensor):
            return MultiVector(self._blade_values*other,self._algebra)
        
        assert isinstance(other, MultiVector), f"{type(other)}"

        return MultiVector(self._algebra.geom_prod(self._blade_values, other._blade_values),self._algebra)
    __rmul__=__mul__
    
    def __truediv__(self, other: MultiVector) -> MultiVector:
        """Division, ie. multiplication with the inverse."""
        if isinstance(other,(int,float,torch.Tensor)):
            return MultiVector(
                self._blade_values/other,                
                self._algebra
            )
        
        assert isinstance(other, MultiVector), f"{type(other)}"
        return MultiVector(
            self._algebra.geom_prod(
                self._blade_values,
                self._algebra.inverse(other._blade_values)
            ),
            self._algebra
        )

    def __and__(self, other: MultiVector) -> MultiVector:
        """Regressive product. See GeometricAlgebra.reg_prod()."""        
        assert isinstance(other, MultiVector), f"{type(other)}"                
        return MultiVector(
            self._algebra.reg_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __invert__(self) -> MultiVector:
        """Reversion. See GeometricAlgebra.reversion()."""
        return MultiVector(
            self._algebra.reversion(self._blade_values),
            self._algebra
        )

    def __neg__(self) -> MultiVector:
        """Negation."""
        return MultiVector(
            -self._blade_values,
            self._algebra
        )

    def __add__(self, other: MultiVector) -> MultiVector:
        """Addition of multivectors."""
        if type(other) in (int, float):
            _tmp = self._blade_values.clone()
            _tmp[...,0]+=other
            return MultiVector(_tmp,self._algebra)        
        
        assert isinstance(other, MultiVector), f"{type(other)}"
        
        return MultiVector(
            self._blade_values + other._blade_values,
            self._algebra
        )
    __radd__=__add__
    
    def __sub__(self, other: MultiVector) -> MultiVector:
        """Subtraction of multivectors."""
        if type(other) in (int, float):
            _tmp = self._blade_values.clone()
            _tmp[...,0]-=other
            return MultiVector(_tmp,self._algebra)        
        
        assert isinstance(other, MultiVector), f"{type(other)}"

        return MultiVector(
            self._blade_values - other._blade_values,
            self._algebra
        )
    def __rsub__(a,b):
        """Reverse subtraction."""
        return b + -1 * a        

    def __pow__(self, n: int) -> MultiVector:
        """Multivector raised to an integer power."""
        return MultiVector(
            self._algebra.int_pow(self._blade_values, n),
            self._algebra
        )

    def __getitem__(self, key: Union[str, list[str]]) -> MultiVector:
        """Return MultiVector with only passed blade names as non-zeros."""
        if type(key) in [int, float, slice]:
            return MultiVector(
                self._blade_values[key],
                self._algebra
            )
            
        return MultiVector(
            self._algebra.keep_blades_with_name(self._blade_values, key),
            self._algebra
        )

    def __call__(self, key: Union[str, list[str]]):
        """Return torch.Tensor with passed blade names on last axis."""
        return self._algebra.select_blades_with_name(self._blade_values, key)

    def __repr__(self) -> str:
        """Return string representation."""
        return self._algebra.mv_repr(self._blade_values)

    def proj(self, other: MultiVector) -> MultiVector:
        """Projection onto multivector."""
        assert isinstance(other, MultiVector), f"{type(other)}"
        return MultiVector(
            self._algebra.projection(other._blade_values, self._blade_values),
            self._algebra
        )
    def proj_on(self, other: MultiVector) -> MultiVector:
        """Projection of this multivector onto another."""
        assert isinstance(other, MultiVector), f"{type(other)}"
        return MultiVector(
            self._algebra.projection(self._blade_values, other._blade_values),
            self._algebra
        )

    def sandwitch(self, other: MultiVector) -> MultiVector:
        """Sandwich product: b*a*b~ where b is other, a is self."""
        assert isinstance(other, MultiVector), f"{type(other)}"
        return MultiVector(
            self._algebra.sandwitch_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def meet(self, other: MultiVector) -> MultiVector:
        """Meet (intersection) of multivectors."""
        assert isinstance(other, MultiVector), f"{type(other)}"
        return MultiVector(
            self._algebra.meet(self._blade_values, other._blade_values),
            self._algebra
        )
    def join(self, other: MultiVector) -> MultiVector:
        """Join (union) of multivectors."""
        assert isinstance(other, MultiVector), f"{type(other)}"
        return MultiVector(
            self._algebra.join(self._blade_values, other._blade_values),
            self._algebra
        )

    def inverse(self) -> MultiVector:
        """Return inverse of multivector."""
        return MultiVector(
            self._algebra.inverse(self._blade_values),
            self._algebra
        )
        
    def inv(self) -> MultiVector:
        """Return inverse of multivector."""
        return self.inverse()
    
    @property
    def shape(self):
        """Return shape of the multivector."""
        return self._blade_values.shape[:-1]

    def simple_inverse(self) -> MultiVector:
        """Return simple inverse of multivector."""
        return MultiVector(
            self._algebra.simple_inverse(self._blade_values),
            self._algebra
        )

    def dual(self) -> MultiVector:
        """Return dual of multivector."""
        return MultiVector(
            self._algebra.dual(self._blade_values),
            self._algebra
        )
    def grade(self,_grade=None):
        """Return the blade of the specified grade."""
        if _grade is None:
            return self._algebra.grade(self._blade_values)
        
        return MultiVector(
            self._algebra.grade(self._blade_values,_grade),
            self._algebra
        )
    # def grade(self):
    #     """
    #     Retruns the max grade of all non zeros blades, if zero, return zero. 
    #     """
    #     return self._algebra.grade(self._blade_values)

    def norm(self):
        """Return norm of multivector."""
        return self._algebra.norm(self._blade_values)
    
    def inorm(self):
        """Return inner norm of multivector."""
        return self._algebra.inorm(self._blade_values)
    
    def normalized(self):
        """Return normalized multivector."""
        # return self._algebra.normalized(self._blade_values)  
        return MultiVector(
            self._algebra.normalized(self._blade_values),
            self._algebra
        )


    def conjugation(self) -> MultiVector:
        """Return conjugation of multivector."""
        return MultiVector(
            self._algebra.conjugation(self._blade_values),
            self._algebra
        )
        
    def reversion(self) -> MultiVector:
        """Return reversion of multivector."""
        return MultiVector(
            self._algebra.reversion(self._blade_values),
            self._algebra
        )

    def grade_automorphism(self) -> MultiVector:
        """Return grade automorphism of multivector."""
        return MultiVector(
            self._algebra.grade_automorphism(self._blade_values),
            self._algebra
        )

    def approx_exp(self, order: int = 50) -> MultiVector:
        """Return approximate exponential of multivector."""
        return MultiVector(
            self._algebra.approx_exp(self._blade_values, order=order),
            self._algebra
        )

  
    def exp(self, square_scalar_tolerance: Union[float, None] = 1e-4) -> MultiVector:
        """Return exponential of multivector."""
        return MultiVector(
            self._algebra.exp(
                self._blade_values,
                square_scalar_tolerance=square_scalar_tolerance
            ),
            self._algebra
        )

    def approx_log(self, order: int = 50) -> MultiVector:
        """Return approximate logarithm of multivector."""
        return MultiVector(
            self._algebra.approx_log(self._blade_values, order=order),
            self._algebra
        )

    def is_pure_kind(self, kind: BladeKind) -> bool:
        """Whether the `MultiVector` is of a pure kind."""
        return self._algebra.is_pure_kind(self._blade_values, kind=kind)

    def geom_conv1d(self, kernel: MultiVector,
                    stride: int, padding: str,
                    dilations: Union[int, None] = None) -> MultiVector:
        """Perform 1D geometric convolution."""
        return MultiVector(
            self._algebra.geom_conv1d(
                self._blade_values, kernel._blade_values,
                stride=stride, padding=padding, dilations=dilations
            ),
            self._algebra
        )
    def __pow__(a,b):
        """Power operator for integer powers."""
        assert(isinstance(b,int) and b >= 0), "power only defined on integers"
        if b==0: return a._algebra.e_
        tmp = a
        for ki in range(b-1):
           tmp = tmp*a
        return tmp
