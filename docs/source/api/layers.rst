Neural Network Layers
=====================

torch_ga provides PyTorch nn.Module layers that incorporate geometric algebra operations.

.. currentmodule:: torch_ga

All Layers
----------

.. automodule:: torch_ga.layers
   :members:
   :undoc-members:
   :show-inheritance:

Layer Types
-----------

Geometric Algebra Layers
^^^^^^^^^^^^^^^^^^^^^^^^

These layers operate in geometric algebra space:

* **GeometricProductLayer** - Applies geometric product transformations
* **GALinear** - Linear transformation in GA space
* **GAConv1d** - 1D convolution with GA operations

Example Usage
-------------

.. code-block:: python

   import torch
   from torch_ga import GeometricAlgebra
   from torch_ga.layers import GeometricProductLayer
   
   # Create algebra
   ga = GeometricAlgebra([1, 1, 1])
   
   # Create a GA layer
   layer = GeometricProductLayer(
       algebra=ga,
       in_features=8,
       out_features=16,
       kind='VECTOR'
   )
   
   # Use in a model
   x = torch.randn(32, 8, ga.num_blades).cuda()
   output = layer(x)  # (32, 16, num_blades)

CUDA Support
------------

All layers support CUDA out of the box. Simply move your tensors and model to CUDA:

.. code-block:: python

   # Move model to CUDA
   model = model.cuda()
   
   # Input tensors on CUDA
   x = torch.randn(32, 8, ga.num_blades, device='cuda')
   
   # Forward pass on CUDA
   output = model(x)

Gradient Flow
-------------

All GA layers preserve gradient flow for backpropagation:

.. code-block:: python

   # Enable gradients
   x = torch.randn(32, 8, ga.num_blades, requires_grad=True).cuda()
   
   # Forward pass
   output = layer(x)
   
   # Backward pass
   loss = output.sum()
   loss.backward()
   
   # Gradients are computed
   print(x.grad.shape)  # Same as x.shape
