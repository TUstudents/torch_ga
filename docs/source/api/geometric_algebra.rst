GeometricAlgebra Class
======================

The :class:`~torch_ga.GeometricAlgebra` class is the main interface for creating and working with geometric algebras.

.. currentmodule:: torch_ga

.. autoclass:: GeometricAlgebra
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Creating an Algebra
-------------------

A geometric algebra is defined by its metric signature:

.. code-block:: python

   from torch_ga import GeometricAlgebra
   
   # 3D Euclidean space
   ga_3d = GeometricAlgebra([1, 1, 1])
   
   # 3D Projective Geometric Algebra (PGA)
   pga = GeometricAlgebra([0, 1, 1, 1])
   
   # Spacetime Algebra (STA)
   sta = GeometricAlgebra([1, -1, -1, -1])

Key Capabilities
----------------

**Tensor Creation**

* ``from_tensor(tensor)`` - Create a multivector from tensor values
* ``from_tensor_with_kind(tensor, kind)`` - Create from tensor with specified blade kind  
* ``from_scalar(scalar)`` - Create a scalar multivector
* ``e(name)`` / ``emv(name)`` - Get basis blade by name (as tensor or MultiVector)

**Products**

* ``geom_prod(a, b)`` - Geometric (Clifford) product
* ``ext_prod(a, b)`` - Exterior (wedge) product
* ``inner_prod(a, b)`` - Inner (dot) product
* ``reg_prod(a, b)`` - Regressive product

**Transformations**

* ``dual(a)`` - Hodge dual
* ``reversion(a)`` - Reversion (grade-based sign flip)
* ``grade_automorphism(a)`` - Grade involution
* ``conjugation(a)`` - Clifford conjugation

**Inversion**

* ``inverse(a)`` - Multiplicative inverse
* ``simple_inverse(a)`` - Simple inverse for blades

**Properties**

* ``metric`` - The metric signature
* ``blades`` - List of blade names
* ``num_blades`` - Total number of blades (2^n)
* ``blade_degrees`` - Grades of each blade
* ``cayley`` - Cayley multiplication tables
