import mlx.core as mx
from ..layers import Linear, TanhActivation, BatchNorm
from ..loss import CrossEntropyLoss

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize layers
        self.layer1 = Linear(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.tanh = TanhActivation()
        self.layer2 = Linear(hidden_dim, output_dim)
        self.bn2 = BatchNorm(output_dim)
        self.loss_fn = CrossEntropyLoss()
    
    def forward(self, x, y, training=True):
        # Layer 1
        h1 = self.layer1.forward(x)
        h1_bn = self.bn1.forward(h1, training)
        h1_act = self.tanh.forward(h1_bn)
        
        # Layer 2
        h2 = self.layer2.forward(h1_act)
        h2_bn = self.bn2.forward(h2, training)
        
        # Loss
        loss = self.loss_fn.forward(h2_bn, y)
        
        return loss
    
    def backward(self):
        # Backprop through loss
        grad = self.loss_fn.backward()
        
        # Backprop through batch norm 2
        grad, grads_bn2 = self.bn2.backward(grad)
        
        # Backprop through layer 2
        grad, grads_2 = self.layer2.backward(grad)
        
        # Backprop through tanh
        grad = self.tanh.backward(grad)
        
        # Backprop through batch norm 1
        grad, grads_bn1 = self.bn1.backward(grad)
        
        # Backprop through layer 1
        _, grads_1 = self.layer1.backward(grad)
        
        return {
            'layer1': grads_1,
            'bn1': grads_bn1,
            'layer2': grads_2,
            'bn2': grads_bn2
        }
    
    def update_parameters(self, grads, learning_rate):
        self.layer1.update(grads['layer1'], learning_rate)
        self.bn1.update(grads['bn1'], learning_rate)
        self.layer2.update(grads['layer2'], learning_rate)
        self.bn2.update(grads['bn2'], learning_rate)
    
    def train_step(self, x, y, learning_rate):
        # Forward pass
        loss = self.forward(x, y, training=True)
        
        # Backward pass
        grads = self.backward()
        
        # Update parameters
        self.update_parameters(grads, learning_rate)
        
        return loss