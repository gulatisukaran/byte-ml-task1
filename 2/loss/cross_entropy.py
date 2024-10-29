import mlx.core as mx

class CrossEntropyLoss:
    def forward(self, x, y):
        exp_x = mx.exp(x - mx.max(x, axis=1, keepdims=True))
        self.probs = exp_x / mx.sum(exp_x, axis=1, keepdims=True)
        self.y = y
        self.N = x.shape[0]
        
        loss = -mx.sum(mx.log(self.probs[mx.arange(self.N), y])) / self.N
        return loss
    
    def backward(self):
        dx = self.probs.copy()
        dx[mx.arange(self.N), self.y] -= 1
        dx /= self.N
        return dx