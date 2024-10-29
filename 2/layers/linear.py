import mlx.core as mx

class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize parameters using He initialization
        scale = mx.sqrt(2.0 / input_dim)
        self.W = mx.random.normal((input_dim, output_dim)) * scale
        self.b = mx.zeros((output_dim,))
        
    def forward(self, x):
        self.input = x
        return mx.dot(x, self.W) + self.b
    
    def backward(self, grad_output):
        # Compute gradients
        grad_input = mx.dot(grad_output, self.W.T)
        grad_W = mx.dot(self.input.T, grad_output)
        grad_b = mx.sum(grad_output, axis=0)
        
        return grad_input, {'W': grad_W, 'b': grad_b}
    
    def update(self, grads, learning_rate):
        self.W -= learning_rate * grads['W']
        self.b -= learning_rate * grads['b']