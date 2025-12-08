"""
Test suite for torch_ga bug fixes.

This module tests:
1. Device compatibility (CPU and CUDA)
2. Gradient flow through torch_ga operations
3. Integration with layers and geometric products

Created to verify fixes for:
- Missing dependencies (icecream, einops)
- CUDA device mismatch in from_tensor() method
"""

import unittest
import torch
from torch_ga import GeometricAlgebra
from torch_ga.blades import BladeKind


class TestDeviceCompatibility(unittest.TestCase):
    """Test that torch_ga operations work on both CPU and CUDA."""

    def setUp(self):
        """Set up test fixtures."""
        self.pga_signature = [0, 1, 1, 1]
        self.sta_signature = [1, -1, -1, -1]
        self.ga = GeometricAlgebra(self.pga_signature)
        self.has_cuda = torch.cuda.is_available()

    def test_from_tensor_cpu(self):
        """Test from_tensor creates output on CPU when input is on CPU."""
        tensor = torch.randn(2, 3)
        blade_indices = torch.tensor([0, 1, 2])
        
        result = self.ga.from_tensor(tensor, blade_indices)
        
        self.assertEqual(result.device.type, 'cpu')
        self.assertEqual(result.dtype, tensor.dtype)
        self.assertEqual(result.shape[0], tensor.shape[0])
        self.assertEqual(result.shape[1], self.ga.num_blades)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_from_tensor_cuda(self):
        """Test from_tensor creates output on CUDA when input is on CUDA."""
        tensor = torch.randn(2, 3, device='cuda')
        blade_indices = torch.tensor([0, 1, 2])
        
        result = self.ga.from_tensor(tensor, blade_indices)
        
        self.assertEqual(result.device.type, 'cuda')
        self.assertEqual(result.dtype, tensor.dtype)
        self.assertEqual(result.shape[0], tensor.shape[0])
        self.assertEqual(result.shape[1], self.ga.num_blades)

    def test_from_tensor_with_kind_cpu(self):
        """Test from_tensor_with_kind on CPU."""
        # Get the actual number of vector blades for this algebra
        num_vectors = len(self.ga.get_kind_blade_indices(BladeKind.VECTOR))
        tensor = torch.randn(2, num_vectors)
        
        result = self.ga.from_tensor_with_kind(tensor, BladeKind.VECTOR)
        
        self.assertEqual(result.device.type, 'cpu')
        self.assertEqual(result.dtype, tensor.dtype)
        self.assertEqual(result.shape[0], tensor.shape[0])
        self.assertEqual(result.shape[1], self.ga.num_blades)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_from_tensor_with_kind_cuda(self):
        """Test from_tensor_with_kind on CUDA."""
        # Get the actual number of vector blades for this algebra
        num_vectors = len(self.ga.get_kind_blade_indices(BladeKind.VECTOR))
        tensor = torch.randn(2, num_vectors, device='cuda')
        
        result = self.ga.from_tensor_with_kind(tensor, BladeKind.VECTOR)
        
        self.assertEqual(result.device.type, 'cuda')
        self.assertEqual(result.dtype, tensor.dtype)
        self.assertEqual(result.shape[0], tensor.shape[0])
        self.assertEqual(result.shape[1], self.ga.num_blades)

    def test_dtype_preservation_cpu(self):
        """Test that dtype is preserved in from_tensor on CPU."""
        for dtype in [torch.float32, torch.float64]:
            tensor = torch.randn(2, 3, dtype=dtype)
            blade_indices = torch.tensor([0, 1, 2])
            
            result = self.ga.from_tensor(tensor, blade_indices)
            
            self.assertEqual(result.dtype, dtype)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_dtype_preservation_cuda(self):
        """Test that dtype is preserved in from_tensor on CUDA."""
        for dtype in [torch.float32, torch.float64]:
            tensor = torch.randn(2, 3, dtype=dtype, device='cuda')
            blade_indices = torch.tensor([0, 1, 2])
            
            result = self.ga.from_tensor(tensor, blade_indices)
            
            self.assertEqual(result.dtype, dtype)


class TestGradientFlow(unittest.TestCase):
    """Test that gradients flow correctly through torch_ga operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.pga_signature = [0, 1, 1, 1]
        self.ga = GeometricAlgebra(self.pga_signature)
        self.has_cuda = torch.cuda.is_available()

    def test_gradient_flow_from_tensor_cpu(self):
        """Test that gradients flow through from_tensor on CPU."""
        tensor = torch.randn(2, 3, requires_grad=True)
        blade_indices = torch.tensor([0, 1, 2])
        
        result = self.ga.from_tensor(tensor, blade_indices)
        loss = result.sum()
        loss.backward()
        
        self.assertIsNotNone(tensor.grad)
        self.assertEqual(tensor.grad.shape, tensor.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gradient_flow_from_tensor_cuda(self):
        """Test that gradients flow through from_tensor on CUDA."""
        tensor = torch.randn(2, 3, device='cuda', requires_grad=True)
        blade_indices = torch.tensor([0, 1, 2])
        
        result = self.ga.from_tensor(tensor, blade_indices)
        loss = result.sum()
        loss.backward()
        
        self.assertIsNotNone(tensor.grad)
        self.assertEqual(tensor.grad.shape, tensor.shape)
        self.assertEqual(tensor.grad.device.type, 'cuda')

    def test_gradient_flow_from_tensor_with_kind_cpu(self):
        """Test that gradients flow through from_tensor_with_kind on CPU."""
        num_vectors = len(self.ga.get_kind_blade_indices(BladeKind.VECTOR))
        tensor = torch.randn(2, num_vectors, requires_grad=True)
        
        result = self.ga.from_tensor_with_kind(tensor, BladeKind.VECTOR)
        loss = result.sum()
        loss.backward()
        
        self.assertIsNotNone(tensor.grad)
        self.assertEqual(tensor.grad.shape, tensor.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gradient_flow_from_tensor_with_kind_cuda(self):
        """Test that gradients flow through from_tensor_with_kind on CUDA."""
        num_vectors = len(self.ga.get_kind_blade_indices(BladeKind.VECTOR))
        tensor = torch.randn(2, num_vectors, device='cuda', requires_grad=True)
        
        result = self.ga.from_tensor_with_kind(tensor, BladeKind.VECTOR)
        loss = result.sum()
        loss.backward()
        
        self.assertIsNotNone(tensor.grad)
        self.assertEqual(tensor.grad.shape, tensor.shape)
        self.assertEqual(tensor.grad.device.type, 'cuda')

    def test_gradient_flow_geom_prod_cpu(self):
        """Test that gradients flow through geometric product on CPU."""
        a = self.ga.e('1') + 2 * self.ga.e('2')
        a = a.unsqueeze(0).requires_grad_(True)
        b = 3 * self.ga.e('1') + self.ga.e('2')
        b = b.unsqueeze(0)
        
        result = self.ga.geom_prod(a, b)
        loss = result.sum()
        loss.backward()
        
        self.assertIsNotNone(a.grad)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gradient_flow_geom_prod_cuda(self):
        """Test that gradients flow through geometric product on CUDA."""
        a = self.ga.e('1') + 2 * self.ga.e('2')
        a = a.unsqueeze(0).cuda().requires_grad_(True)
        b = 3 * self.ga.e('1') + self.ga.e('2')
        b = b.unsqueeze(0).cuda()
        
        result = self.ga.geom_prod(a, b)
        loss = result.sum()
        loss.backward()
        
        self.assertIsNotNone(a.grad)
        self.assertEqual(a.grad.device.type, 'cuda')


class TestIntegration(unittest.TestCase):
    """Integration tests for torch_ga operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.pga_signature = [0, 1, 1, 1]
        self.ga = GeometricAlgebra(self.pga_signature)

    def test_e_basis_cpu(self):
        """Test basis blade creation on CPU."""
        e1 = self.ga.e('1')
        e2 = self.ga.e('2')
        
        self.assertEqual(e1.device.type, 'cpu')
        self.assertEqual(e2.device.type, 'cpu')
        
        # Test geometric product
        result = self.ga.geom_prod(e1, e2)
        self.assertEqual(result.device.type, 'cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_tensor_operations(self):
        """Test complete workflow on CUDA."""
        # Create multivectors from tensors on CUDA
        num_vectors = len(self.ga.get_kind_blade_indices(BladeKind.VECTOR))
        vec1 = torch.randn(4, num_vectors, device='cuda')
        vec2 = torch.randn(4, num_vectors, device='cuda')
        
        mv1 = self.ga.from_tensor_with_kind(vec1, BladeKind.VECTOR)
        mv2 = self.ga.from_tensor_with_kind(vec2, BladeKind.VECTOR)
        
        # Perform operations
        result_geom = self.ga.geom_prod(mv1, mv2)
        result_ext = self.ga.ext_prod(mv1, mv2)
        
        # Verify all results are on CUDA
        self.assertEqual(mv1.device.type, 'cuda')
        self.assertEqual(mv2.device.type, 'cuda')
        self.assertEqual(result_geom.device.type, 'cuda')
        self.assertEqual(result_ext.device.type, 'cuda')

    def test_mixed_operations_cpu(self):
        """Test mixed operations (scalars, vectors, bivectors) on CPU."""
        num_vectors = len(self.ga.get_kind_blade_indices(BladeKind.VECTOR))
        scalar = self.ga.from_scalar(2.0)
        # Create a random vector for testing
        vec_values = torch.randn(1, num_vectors)
        vector = self.ga.from_tensor_with_kind(vec_values, BladeKind.VECTOR)
        
        # Geometric product of scalar and vector
        result = self.ga.geom_prod(scalar.unsqueeze(0), vector)
        
        self.assertEqual(result.device.type, 'cpu')
        # Result should be 2 * vector
        vec_indices = self.ga.get_kind_blade_indices(BladeKind.VECTOR)
        torch.testing.assert_close(result[0, vec_indices], vector[0, vec_indices] * 2.0, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_batch_operations_cuda(self):
        """Test batch operations on CUDA."""
        batch_size = 16
        num_vectors = len(self.ga.get_kind_blade_indices(BladeKind.VECTOR))
        
        vectors = torch.randn(batch_size, num_vectors, device='cuda', requires_grad=True)
        mvs = self.ga.from_tensor_with_kind(vectors, BladeKind.VECTOR)
        
        # Self geometric product
        result = self.ga.geom_prod(mvs, mvs)
        
        # Backward pass
        loss = result.sum()
        loss.backward()
        
        self.assertEqual(result.device.type, 'cuda')
        self.assertIsNotNone(vectors.grad)
        self.assertEqual(vectors.grad.device.type, 'cuda')


class TestDependencies(unittest.TestCase):
    """Test that dependencies are properly available."""

    def test_icecream_import(self):
        """Test that icecream can be imported."""
        try:
            from icecream import ic
            self.assertTrue(True)
        except ImportError:
            self.fail("icecream package not available")

    def test_einops_import(self):
        """Test that einops can be imported."""
        try:
            import einops
            self.assertTrue(True)
        except ImportError:
            self.fail("einops package not available")

    def test_jacobian_module(self):
        """Test that jacobian module can be imported (uses einops)."""
        try:
            from torch_ga.jacobian import get_jacobian
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"jacobian module import failed: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
