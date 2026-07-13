from lil_torch import tensor
from lil_torch.optim import SGD

# 1. Initialize learnable parameters
w = tensor([2.0], requires_grad=True)
b = tensor([1.0], requires_grad=True)

# 2. Instantiate the optimizer engine
optimizer = SGD([w, b], lr=0.1)

# --- The Training Epoch Step ---
# Reset old accumulated gradients
optimizer.zero_grad()

# Forward Pass (Calculates tracking graphs via Solution 2 wrappers)
y_pred = w * tensor([3.0]) + b
loss = (y_pred - tensor([10.0])).tanh() # Employs our Tanh node

# Backward Pass (Traverses the graph via VJPs)
loss.backward()

# Update parameters using our SGD math engine
optimizer.step()