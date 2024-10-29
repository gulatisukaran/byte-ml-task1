import mlx.core as mx
from model import MLP

def train():
    # Model hyperparameters
    input_dim = 784  # e.g., MNIST
    hidden_dim = 256
    output_dim = 10
    batch_size = 32
    learning_rate = 0.01
    n_epochs = 5
    
    # Initialize model
    model = MLP(input_dim, hidden_dim, output_dim)
    
    # Generate random data for example
    x = mx.random.normal((batch_size, input_dim))
    y = mx.random.randint(0, output_dim, (batch_size,))
    
    # Training loop
    for epoch in range(n_epochs):
        loss = model.train_step(x, y, learning_rate)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

if __name__ == "__main__":
    train()