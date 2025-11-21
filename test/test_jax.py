"""
Test that jax was installed properly and has basic functionality
"""

import unittest

class TestJax(unittest.TestCase):
    def test_import(self):
        """JAX can be imported."""
        try:
            import jax
        except Exception as e:          # pragma: no cover
            self.fail(f"Failed to import jax: {e}")

    def test_basic_operation(self):
        """A simple JAX array operation works."""
        import jax.numpy as jnp
        a = jnp.arange(5)          # [0,1,2,3,4]
        b = jnp.arange(5) * 2      # [0,2,4,6,8]
        c = a + b                  # element‑wise add
        expected = [0, 3, 6, 9, 12]
        # `c` is a DeviceArray; converting to a Python list for comparison
        self.assertListEqual(c.tolist(), expected)

if __name__ == "__main__":
    unittest.main()

