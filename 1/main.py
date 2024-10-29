from autograd import Value

def main():
    # Create some values
    x = Value(2.0)
    y = Value(3.0)
    
    # Perform computations
    z = x * y + x ** 2
    
    # Compute gradients
    z.backward()
    
    # Print results
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"z = {z.data}")
    print(f"x.grad = {x.grad}")  # Will show the gradient with respect to x
    print(f"y.grad = {y.grad}")  # Will show the gradient with respect to y
    
    # Let's break down what these gradients mean:
    print("\nExplanation:")
    print("z = x * y + x^2")
    print("∂z/∂x = y + 2x = 3 + 4 = 7")  # This explains x.grad
    print("∂z/∂y = x = 2")               # This explains y.grad

if __name__ == "__main__":
    main()