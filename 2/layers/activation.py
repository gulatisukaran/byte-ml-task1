import mlx.core as mx

class TanhActivation:
    def forward(self, x):
        self.output = mx.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * (1 - self.output * self.output)