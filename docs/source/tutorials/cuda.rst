CUDA Support
============

torch_ga provides full CUDA support for GPU-accelerated geometric algebra operations.

.. versionadded:: 0.0.6
   Full CUDA support with automatic device handling.

Device Management
-----------------

Moving Tensors to CUDA
^^^^^^^^^^^^^^^^^^^^^^

All torch_ga operations respect the device of input tensors:

.. code-block:: python

   import torch
   from torch_ga import GeometricAlgebra
   from torch_ga.blades import BladeKind
   
   # Create algebra
   ga = GeometricAlgebra([1, 1, 1])
   
   # Create vectors on CUDA
   vectors = torch.randn(32, 3, device='cuda')
   mv = ga.from_tensor_with_kind(vectors, BladeKind.VECTOR)
   
   print(mv.device)  # cuda:0

Automatic Device Transfer
^^^^^^^^^^^^^^^^^^^^^^^^^^

torch_ga automatically handles device transfers for internal tensors (like Cayley tables):

.. code-block:: python

   # CPU tensors
   a_cpu = torch.randn(10, 3)
   mv_cpu = ga.from_tensor_with_kind(a_cpu, BladeKind.VECTOR)
   
   # CUDA tensors
   a_cuda = torch.randn(10, 3, device='cuda')
   mv_cuda = ga.from_tensor_with_kind(a_cuda, BladeKind.VECTOR)
   
   # Both work correctly - device is handled automatically
   result_cpu = ga.geom_prod(mv_cpu, mv_cpu)
   result_cuda = ga.geom_prod(mv_cuda, mv_cuda)

Operations on CUDA
------------------

All Geometric Products
^^^^^^^^^^^^^^^^^^^^^^

All GA operations work seamlessly on CUDA:

.. code-block:: python

   # Create multivectors on CUDA
   a = torch.randn(100, 3, device='cuda', requires_grad=True)
   b = torch.randn(100, 3, device='cuda', requires_grad=True)
   
   mv_a = ga.from_tensor_with_kind(a, BladeKind.VECTOR)
   mv_b = ga.from_tensor_with_kind(b, BladeKind.VECTOR)
   
   # Geometric product on CUDA
   geom_result = ga.geom_prod(mv_a, mv_b)
   
   # Exterior product on CUDA
   ext_result = ga.ext_prod(mv_a, mv_b)
   
   # Inner product on CUDA
   inner_result = ga.inner_prod(mv_a, mv_b)
   
   # All results are on CUDA
   assert geom_result.device.type == 'cuda'

Gradient Flow on CUDA
----------------------

Automatic Differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^^

All operations support backpropagation on CUDA:

.. code-block:: python

   # Enable gradients
   vectors = torch.randn(32, 3, device='cuda', requires_grad=True)
   mv = ga.from_tensor_with_kind(vectors, BladeKind.VECTOR)
   
   # Perform operations
   result = ga.geom_prod(mv, mv)
   
   # Compute loss and backpropagate
   loss = result.sum()
   loss.backward()
   
   # Gradients are computed and on CUDA
   print(vectors.grad.device)  # cuda:0
   print(vectors.grad.shape)   # torch.Size([32, 3])

Training Neural Networks
^^^^^^^^^^^^^^^^^^^^^^^^

Use GA operations in PyTorch models with CUDA:

.. code-block:: python

   import torch.nn as nn
   from torch_ga.layers import GeometricProductLayer
   
   class GAModel(nn.Module):
       def __init__(self, ga):
           super().__init__()
           self.ga_layer = GeometricProductLayer(
               algebra=ga,
               in_features=16,
               out_features=32,
               kind='VECTOR'
           )
           
       def forward(self, x):
           return self.ga_layer(x)
   
   # Create model and move to CUDA
   model = GAModel(ga).cuda()
   
   # Train on CUDA
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   for epoch in range(100):
       # CUDA input
       x = torch.randn(64, 16, ga.num_blades, device='cuda')
       y_target = torch.randn(64, 32, ga.num_blades, device='cuda')
       
       # Forward pass on CUDA
       y_pred = model(x)
       loss = ((y_pred - y_target) ** 2).mean()
       
       # Backward pass
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()

Performance Tips
----------------

Batch Processing
^^^^^^^^^^^^^^^^

Process data in batches for better GPU utilization:

.. code-block:: python

   # Good: Batch processing
   batch_vectors = torch.randn(1000, 3, device='cuda')
   batch_mv = ga.from_tensor_with_kind(batch_vectors, BladeKind.VECTOR)
   batch_result = ga.geom_prod(batch_mv, batch_mv)  # Fast!
   
   # Bad: Sequential processing
   for i in range(1000):
       single_vector = torch.randn(3, device='cuda')
       single_mv = ga.from_tensor_with_kind(single_vector, BladeKind.VECTOR)
       single_result = ga.geom_prod(single_mv, single_mv)  # Slow!

Mixed Precision
^^^^^^^^^^^^^^^

Use mixed precision training for better performance:

.. code-block:: python

   from torch.cuda.amp import autocast, GradScaler
   
   model = GAModel(ga).cuda()
   optimizer = torch.optim.Adam(model.parameters())
   scaler = GradScaler()
   
   for epoch in range(100):
       x = torch.randn(64, 16, ga.num_blades, device='cuda')
       y = torch.randn(64, 32, ga.num_blades, device='cuda')
       
       with autocast():
           y_pred = model(x)
           loss = ((y_pred - y) ** 2).mean()
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
       optimizer.zero_grad()

Multi-GPU Support
-----------------

torch_ga works with PyTorch's multi-GPU features:

.. code-block:: python

   # Data parallel
   model = nn.DataParallel(GAModel(ga)).cuda()
   
   # Distributed data parallel
   from torch.nn.parallel import DistributedDataParallel as DDP
   model = DDP(GAModel(ga).cuda(), device_ids=[local_rank])

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Out of Memory**: Reduce batch size or use gradient checkpointing

**Slow Performance**: Ensure operations are batched and use mixed precision

**Device Mismatch**: All input tensors must be on the same device

Checking CUDA Availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   
   # Check if CUDA is available
   if torch.cuda.is_available():
       device = 'cuda'
       print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
   else:
       device = 'cpu'
       print("CUDA not available, using CPU")
   
   # Use the detected device
   vectors = torch.randn(32, 3, device=device)
