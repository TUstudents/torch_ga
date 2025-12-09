MultiVector Class
=================

The :class:`~torch_ga.MultiVector` class provides an object-oriented interface for working with geometric algebra elements.

.. currentmodule:: torch_ga

.. autoclass:: MultiVector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __add__, __sub__, __mul__, __truediv__, __xor__, __or__, __invert__

Overview
--------

A MultiVector wraps a PyTorch tensor and provides geometric algebra operations as methods.

.. code-block:: python

   from torch_ga import GeometricAlgebra
   
   ga = GeometricAlgebra([1, 1, 1])
   
   # Create multivectors
   v1 = ga.emv('1') + 2 * ga.emv('2')
   v2 = ga.emv('2') + ga.emv('3')
   
   # Operations use operator overloading
   result = v1 * v2  # Geometric product
   dual_result = v1.dual()

Operators
---------

The MultiVector class supports intuitive operator overloading:

* ``a + b`` - Addition
* ``a - b`` - Subtraction  
* ``a * b`` - Geometric product
* ``a / b`` - Division (multiplication by inverse)
* ``~a`` - Reversion
* ``a | b`` - Inner product
* ``a ^ b`` - Exterior (wedge) product
* ``a << b`` - Left contraction
* ``a >> b`` - Right contraction
* ``a & b`` - Regressive product

Key Properties
--------------

* ``tensor`` - The underlying PyTorch tensor holding blade values
* ``algebra`` - The GeometricAlgebra instance this multivector belongs to
* ``shape`` - Shape of the multivector tensor
* ``batch_shape`` - Shape of all axes except the last (blade) axis

Key Methods
-----------

**Grade Operations**

* ``grade(g)`` - Extract components of a specific grade
* ``scalar`` - Get scalar (grade-0) part
* ``vector`` - Get vector (grade-1) part
* ``bivector`` - Get bivector (grade-2) part

**Transformations**

* ``dual()`` - Compute the dual
* ``reversion()`` - Compute the reversion (tilde operator)
* ``inverse()`` - Compute the multiplicative inverse
* ``conjugation()`` - Clifford conjugation
* ``grade_automorphism()`` - Grade automorphism

**Norms**

* ``norm()`` - Compute the norm
* ``normalized()`` - Return unit multivector

**Advanced**

* ``exp()`` - Exponential of multivector
* ``approx_exp(order)`` - Approximate exponential via Taylor series
* ``approx_log(order)`` - Approximate logarithm via Taylor series
* ``sandwitch(other)`` - Sandwich product: other * self * ~other
* ``proj(other)`` - Project onto another multivector
* ``meet(other)`` - Meet (intersection) 
* ``join(other)`` - Join (union)
