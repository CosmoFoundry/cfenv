import unittest
import torch

class TestPyTorch(unittest.TestCase):
    def test_import(self):
        """torch can be imported."""
        try:
            import torch
        except Exception as e:          # pragma: no cover
            self.fail(f"Failed to import torch: {e}")

    def test_basic_cpu_operation(self):
        """A simple torch tensor operation works."""
        # create two 1‑D tensors
        a = torch.arange(5)          # tensor([0, 1, 2, 3, 4])
        b = torch.arange(5) * 2      # tensor([0, 2, 4, 6, 8])
        c = a + b                    # element‑wise add
        expected = torch.tensor([0, 3, 6, 9, 12])
        self.assertTrue(torch.equal(c, expected))

    # verify CUDA (GPU) availability
    def test_cuda_available(self):
        """torch.cuda.is_available() runs without error."""
        self.assertTrue(torch.cuda.is_available())

    # Run CUDA-based functionality tests only if CUDA is available

    # Simple CUDA calculation
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_basic_cuda_operation(self):
        """Run the same tiny op on the GPU and verify the result."""
        device = torch.device("cuda")          # the first GPU
        # create tensors directly on the GPU
        a = torch.arange(5, device=device, dtype=torch.float32)
        b = torch.arange(5, device=device, dtype=torch.float32) * 2
        c = a + b                               # still on GPU

        # bring result back to CPU for easy comparison
        c_cpu = c.to("cpu")
        expected = torch.tensor([0, 3, 6, 9, 12], dtype=torch.float32)
        self.assertTrue(torch.equal(c_cpu, expected))

    # CUDA‑to‑CPU round‑trip
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_tensor_transfer(self):
        """Tensor can move CPU → GPU → CPU without data loss."""
        src = torch.randn(3, 4)                # CPU tensor
        gpu = src.to("cuda")                   # copy to GPU
        back = gpu.to("cpu")                   # copy back
        self.assertTrue(torch.allclose(src, back))


if __name__ == "__main__":
    unittest.main()

