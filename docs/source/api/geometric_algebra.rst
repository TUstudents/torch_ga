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

Key Methods
-----------

Tensor Creation
^^^^^^^^^^^^^^^

.. automethod:: GeometricAlgebra.from_tensor
.. automethod:: GeometricAlgebra.from_tensor_with_kind
.. automethod:: GeometricAlgebra.from_scalar

Basis Blades
^^^^^^^^^^^^

.. automethod:: GeometricAlgebra.e

Products
^^^^^^^^

.. automethod:: GeometricAlgebra.geom_prod
.. automethod:: GeometricAlgebra.ext_prod
.. automethod:: GeometricAlgebra.inner_prod

Transformations
^^^^^^^^^^^^^^^

.. automethod:: GeometricAlgebra.dual
.. automethod:: GeometricAlgebra.reversion
.. automethod:: GeometricAlgebra.grade_automorphism
.. automethod:: GeometricAlgebra.conjugation

Inversion
^^^^^^^^^

.. automethod:: GeometricAlgebra.inverse
.. automethod:: GeometricAlgebra.simple_inverse

Properties
----------

.. autoproperty:: GeometricAlgebra.metric
.. autoproperty:: GeometricAlgebra.cayley
.. autoproperty:: GeometricAlgebra.blades
.. autoproperty:: GeometricAlgebra.num_blades
.. autoproperty:: GeometricAlgebra.blade_degrees
