MultiVector Class
=================

The :class:`~torch_ga.MultiVector` class provides an object-oriented interface for working with geometric algebra elements.

.. currentmodule:: torch_ga

.. autoclass:: MultiVector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __add__, __sub__, __mul__, __truediv__

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

Properties
----------

.. autoproperty:: MultiVector.tensor
.. autoproperty:: MultiVector.algebra
.. autoproperty:: MultiVector.device
.. autoproperty:: MultiVector.dtype
.. autoproperty:: MultiVector.shape

Methods
-------

Products
^^^^^^^^

.. automethod:: MultiVector.geom_prod
.. automethod:: MultiVector.inner_prod
.. automethod:: MultiVector.outer_prod

Transformations
^^^^^^^^^^^^^^^

.. automethod:: MultiVector.dual
.. automethod:: MultiVector.reversion
.. automethod:: MultiVector.inverse

Grade Operations
^^^^^^^^^^^^^^^^

.. automethod:: MultiVector.grade

