Quickstart Guide
================

This guide will help you get started with torch_ga and perform basic geometric algebra operations.

Installation
------------

Install torch_ga using pip:

.. code-block:: bash

   pip install torch_ga

Or with uv:

.. code-block:: bash

   uv add torch_ga

Creating a Geometric Algebra
-----------------------------

A geometric algebra is defined by its metric signature. Common algebras include:

.. code-block:: python

   from torch_ga import GeometricAlgebra
   
   # 2D Euclidean space (VGA)
   ga_2d = GeometricAlgebra([1, 1])
   
   # 3D Euclidean space (VGA)  
   ga_3d = GeometricAlgebra([1, 1, 1])
   
   # 3D Projective Geometric Algebra (PGA)
   pga = GeometricAlgebra([0, 1, 1, 1])
   
   # Spacetime Algebra (STA)
   sta = GeometricAlgebra([1, -1, -1, -1])

Working with Basis Blades
--------------------------

Create basis blades using the ``e()`` method:

.. code-block:: python

   # Scalar
   scalar = pga.e('')
   
   # Basis vectors
   e0 = pga.e('0')
   e1 = pga.e('1')
   e2 = pga.e('2')
   e3 = pga.e('3')
   
   # Bivectors
   e01 = pga.e('01')
   e12 = pga.e('12')
   
   # Pseudoscalar
   I = pga.e('0123')

Linear combinations work as expected:

.. code-block:: python

   # Create a multivector
   mv = 2*e1 + 3*e2 + 5*e12
   print(mv.shape)  # torch.Size([16]) for PGA

The Geometric Product
---------------------

The geometric product is the fundamental operation in GA:

.. code-block:: python

   import torch
   
   # Create two vectors
   a = pga.e('1') + 2*pga.e('2')
   b = 3*pga.e('1') + pga.e('2')
   
   # Geometric product
   result = pga.geom_prod(a, b)
   
   # result contains both scalar (dot product) and bivector (wedge) parts

The geometric product decomposes as: ``ab = a·b + a∧b`` (inner + outer product)

Products and Operations
-----------------------

torch_ga supports all common GA operations:

.. code-block:: python

   a = pga.e('1') + pga.e('2')
   b = pga.e('2') + pga.e('3')
   
   # Exterior (wedge) product
   wedge = pga.ext_prod(a, b)
   
   # Inner product
   inner = pga.inner_prod(a, b)
   
   # Dual
   a_dual = pga.dual(a)
   
   # Reversion (reverse order of basis vectors)
   a_rev = pga.reversion(a)
   
   # Inverse
   a_inv = pga.inverse(a)

Working with Tensors
--------------------

Create multivectors from PyTorch tensors:

.. code-block:: python

   from torch_ga.blades import BladeKind
   import torch
   
   # Create a batch of vectors
   vectors = torch.randn(32, 4)  # 32 vectors, 4 components
   
   # Convert to geometric algebra multivectors
   mv_batch = pga.from_tensor_with_kind(vectors, BladeKind.VECTOR)
   
   print(mv_batch.shape)  # torch.Size([32, 16])

Batch Operations
----------------

All operations work with batched tensors:

.. code-block:: python

   # Batch of 100 vectors
   batch_a = torch.randn(100, 4)
   batch_b = torch.randn(100, 4)
   
   # Convert to multivectors
   mv_a = pga.from_tensor_with_kind(batch_a, BladeKind.VECTOR)
   mv_b = pga.from_tensor_with_kind(batch_b, BladeKind.VECTOR)
   
   # Batched geometric product
   results = pga.geom_prod(mv_a, mv_b)
   
   print(results.shape)  # torch.Size([100, 16])

Using MultiVector Objects
--------------------------

For a more object-oriented approach, use the MultiVector class:

.. code-block:: python

   # Create multivectors
   v1 = pga.emv('1') + 2*pga.emv('2')
   v2 = pga.emv('2') + pga.emv('3')
   
   # Operator overloading
   result = v1 * v2  # Geometric product
   wedge = v1 ^ v2   # Exterior product
   inner = v1 | v2   # Inner product
   dual = ~v1        # Dual
   
   # Method calls
   inv = v1.inverse()
   norm = v1.normalize()

Next Steps
----------

* Learn about :doc:`cuda` for GPU acceleration
* Explore the :doc:`../api/index` for detailed documentation
* Check out the example notebooks in the repository
