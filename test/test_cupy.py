# Basic cupy-installation tests

import sys
import unittest

# ----------------------------------------------------------------------
# Import CuPy once – abort early if the package cannot be loaded.
# ----------------------------------------------------------------------
try:
    import cupy as cp
except Exception as e:                     # pragma: no cover
    sys.exit(f"Cannot import cupy: {e}")

# ----------------------------------------------------------------------
# Helper: does the current environment have at least one CUDA device?
# ----------------------------------------------------------------------
def _cuda_available() -> bool:
    try:
        # cupy raises an error on systems without a CUDA driver
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


class TestCuPyInstallation(unittest.TestCase):

    # ------------------- 1. import & version -------------------------
    def test_import(self):
        """CuPy can be imported."""
        self.assertIsNotNone(cp)

    def test_version(self):
        """CuPy exposes a PEP‑440 version string."""
        self.assertTrue(hasattr(cp, "__version__"))
        self.assertRegex(cp.__version__, r'^\d+\.\d+\.\d+')

    # ------------------- 2. CPU‑only (host) operation ---------------
    def test_basic_host_operation(self):
        """A simple operation that does not require a GPU works."""
        import numpy as np
        a = np.arange(5, dtype=np.float32)
        b = a * 2
        expected = np.array([0, 2, 4, 6, 8], dtype=np.float32)
        np.testing.assert_array_equal(b, expected)

    # ------------------- 3. Simple CuPy (GPU) operation ------------
    @unittest.skipUnless(_cuda_available(), "CUDA device not available")
    def test_basic_cuda_operation(self):
        """Create CuPy arrays on the GPU, perform a calculation, and compare."""
        a = cp.arange(5, dtype=cp.float32)          # on GPU
        b = a * 2                                    # element‑wise multiply
        c = a + b                                    # add → [0,3,6,9,12]

        expected = cp.array([0, 3, 6, 9, 12], dtype=cp.float32)

        # `.get()` copies back to host (NumPy) for comparison
        cp.testing.assert_allclose(c, expected)

    # ------------------- 4. Host ↔ GPU round‑trip -----------------
    @unittest.skipUnless(_cuda_available(), "CUDA device not available")
    def test_transfer_roundtrip(self):
        """Move a NumPy array to the GPU and back without data loss."""
        import numpy as np
        host = np.random.randn(3, 4).astype(np.float32)

        gpu = cp.asarray(host)          # host → GPU
        back = cp.asnumpy(gpu)          # GPU → host

        np.testing.assert_allclose(host, back)


# ----------------------------------------------------------------------
# Run the test suite when the file is executed directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
