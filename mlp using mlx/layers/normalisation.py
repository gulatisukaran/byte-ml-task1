import mlx.core as mx

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = mx.ones((num_features,))
        self.beta = mx.zeros((num_features,))
        
        # Running statistics
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
    
    def forward(self, x, training=True):
        if training:
            batch_mean = mx.mean(x, axis=0)
            batch_var = mx.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (
                self.momentum * self.running_mean +
                (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var +
                (1 - self.momentum) * batch_var
            )
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        # Normalize
        x_normalized = (x - batch_mean) / mx.sqrt(batch_var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_normalized + self.beta
        
        # Cache for backward pass
        self.cache = (x, x_normalized, batch_mean, batch_var)
        
        return out
    
    def backward(self, grad_output):
        x, x_normalized, batch_mean, batch_var = self.cache
        N = grad_output.shape[0]
        
        # Gradients for gamma and beta
        grad_gamma = mx.sum(grad_output * x_normalized, axis=0)
        grad_beta = mx.sum(grad_output, axis=0)
        
        # Gradient for normalized input
        grad_normalized = grad_output * self.gamma
        
        # Gradient for variance
        std_inv = 1.0 / mx.sqrt(batch_var + self.eps)
        grad_input = grad_normalized * std_inv
        
        # Gradient for mean
        grad_mean = -mx.sum(grad_input, axis=0)
        
        # Final gradient for input
        grad_input = grad_input + grad_mean / N
        
        return grad_input, {'gamma': grad_gamma, 'beta': grad_beta}
    
    def update(self, grads, learning_rate):
        self.gamma -= learning_rate * grads['gamma']
        self.beta -= learning_rate * grads['beta']