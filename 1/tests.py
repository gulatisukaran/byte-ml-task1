import math
import unittest

class TestMicrograd(unittest.TestCase):
    def test_simple_addition(self):
        # Test basic addition
        x = Value(2.0)
        y = Value(3.0)
        z = x + y
        z.backward()
        
        self.assertEqual(z.data, 5.0)
        self.assertEqual(x.grad, 1.0)
        self.assertEqual(y.grad, 1.0)

    def test_multiplication(self):
        # Test multiplication
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        z.backward()
        
        self.assertEqual(z.data, 6.0)
        self.assertEqual(x.grad, 3.0)
        self.assertEqual(y.grad, 2.0)

    def test_power(self):
        # Test power operation
        x = Value(2.0)
        z = x ** 2
        z.backward()
        
        self.assertEqual(z.data, 4.0)
        self.assertEqual(x.grad, 4.0)  # derivative of x^2 is 2x

    def test_complex_expression(self):
        # Test a more complex expression: f(x,y) = x^2 * y + y
        x = Value(2.0)
        y = Value(3.0)
        
        z = (x ** 2) * y + y
        z.backward()
        
        self.assertEqual(z.data, 15.0)  # 2^2 * 3 + 3 = 15
        self.assertEqual(x.grad, 12.0)  # ∂z/∂x = 2x * y = 12
        self.assertEqual(y.grad, 5.0)   # ∂z/∂y = x^2 + 1 = 5

    def test_tanh(self):
        # Test tanh activation function
        x = Value(0.0)
        z = x.tanh()
        z.backward()
        
        self.assertAlmostEqual(z.data, 0.0)
        self.assertAlmostEqual(x.grad, 1.0)  # derivative of tanh at 0 is 1

if __name__ == '__main__':
    unittest.main()