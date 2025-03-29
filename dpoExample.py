import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Synthetic data for demonstration
# Each sample: (input_features, preferred_output, dispreferred_output)
data = [
    (torch.tensor([1.0, 0.5]), torch.tensor([1.0]), torch.tensor([0.0])),
    (torch.tensor([0.2, 0.8]), torch.tensor([0.0]), torch.tensor([1.0])),
    (torch.tensor([0.6, 0.6]), torch.tensor([1.0]), torch.tensor([0.0])),
]

# Simple neural network
class SimplePolicy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# DPO loss as per the method's derivation
def dpo_loss(logits_chosen, logits_rejected, beta=0.1):
    # logits are pre-sigmoid logit scores
    diff = (logits_chosen - logits_rejected) / beta
    return -F.logsigmoid(diff).mean()

# Initialize model
model = SimplePolicy(input_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Training loop
for epoch in range(50):
    total_loss = 0.0
    for x, y_chosen, y_rejected in data:
        optimizer.zero_grad()
        logit_chosen = model(x * y_chosen)
        logit_rejected = model(x * y_rejected)
        loss = dpo_loss(logit_chosen, logit_rejected)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
