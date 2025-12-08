torch_ga Documentation
======================

**torch_ga** is a PyTorch library for Clifford and Geometric Algebra computations with full CUDA support.

Geometric Algebra (GA) provides a unified mathematical framework for representing geometric transformations, 
rotations, and complex geometric relationships. This library implements GA operations efficiently using PyTorch, 
enabling gradient-based optimization and deep learning applications.

.. note::
   As of version 0.0.6, torch_ga now includes full CUDA support and all dependencies are properly managed.

Features
--------

* **Full CUDA Support**: Run GA operations on GPU with automatic device handling
* **Gradient Flow**: All operations support automatic differentiation via PyTorch's autograd
* **Multiple Algebras**: Support for various geometric algebras (PGA, STA, dual numbers, etc.)
* **Efficient Implementation**: Optimized tensor operations using einsum and custom kernels
* **Neural Network Layers**: GA-aware layers for deep learning applications

Quick Example
-------------

.. code-block:: python

   import torch
   from torch_ga import GeometricAlgebra
   
   # Create a 3D Projective Geometric Algebra (PGA)
   ga = GeometricAlgebra([0, 1, 1, 1])
   
   # Create basis vectors on CUDA
   e1 = ga.e('1').cuda()
   e2 = ga.e('2').cuda()
   
   # Geometric product
   result = ga.geom_prod(e1, e2)
   print(result)  # Bivector e12

Installation
------------

.. code-block:: bash

   pip install torch_ga

Or with uv:

.. code-block:: bash

   uv add torch_ga

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials/quickstart
   tutorials/cuda
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
