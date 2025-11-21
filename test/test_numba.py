# test_numba_basic.py
import sys
import unittest

# ----------------------------------------------------------------------
# Import Numba once – abort early if the package cannot be loaded.
# ----------------------------------------------------------------------
try:
    import numba
    import numba.cuda
    from numba import njit, prange
    import numpy as np
except Exception as e:                     # pragma: no cover
    sys.exit(f"Cannot import numba: {e}")

# ----------------------------------------------------------------------
# Helper: does the current environment have at least one CUDA device?
# ----------------------------------------------------------------------
def _cuda_available() -> bool:
    try:
        return numba.cuda.is_available()
    except Exception:   # on systems without the CUDA driver
        return False


class TestNumbaInstallation(unittest.TestCase):

    # ------------------- 1. import & version -------------------------
    def test_import(self):
        """Numba can be imported."""
        self.assertIsNotNone(numba)

    def test_version(self):
        """Numba exposes a PEP‑440 version string."""
        self.assertTrue(hasattr(numba, "__version__"))
        self.assertRegex(numba.__version__, r'^\d+\.\d+\.\d+')

    # ------------------- 2. Simple CPU JIT -------------------------
    def test_basic_cpu_jit(self):
        """A trivial @njit function works on the CPU."""

        @njit
        def sum_range(n):
            s = 0
            for i in prange(n):      # prange works like range in nopython mode
                s += i
            return s

        self.assertEqual(sum_range(10), sum(range(10)))
        self.assertEqual(sum_range(0), 0)

    # ------------------- 3. CUDA availability -----------------------
    def test_cuda_available(self):
        """numba.cuda.is_available() returns True."""
        self.assertTrue(numba.cuda.is_available())

    # ------------------- 4. Simple CUDA kernel ----------------------
    @unittest.skipUnless(_cuda_available(), "CUDA device not available")
    def test_basic_cuda_kernel(self):
        """Launch a tiny CUDA kernel and verify the result."""

        @numba.cuda.jit
        def add_kernel(a, b, out):
            i = numba.cuda.grid(1)
            if i < a.size:
                out[i] = a[i] + b[i]

        N = 64
        a = np.arange(N, dtype=np.float32)
        b = np.arange(N, dtype=np.float32) * 2
        out = np.empty_like(a)

        # Allocate device memory and copy inputs
        d_a = numba.cuda.to_device(a)
        d_b = numba.cuda.to_device(b)
        d_out = numba.cuda.device_array_like(a)

        threads_per_block = 32
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
        add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_out)

        # Copy result back to host
        d_out.copy_to_host(out)

        expected = a + b
        np.testing.assert_allclose(out, expected)

    # ------------------- 5. Host ⇄ GPU round‑trip -------------------
    @unittest.skipUnless(_cuda_available(), "CUDA device not available")
    def test_transfer_roundtrip(self):
        """Transfer a NumPy array to the GPU and back without loss."""
        host = np.random.randn(5, 3).astype(np.float32)

        d_arr = numba.cuda.to_device(host)   # host → device
        back = d_arr.copy_to_host()          # device → host

        np.testing.assert_allclose(host, back)


# ----------------------------------------------------------------------
# Run the suite when the file is executed directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
