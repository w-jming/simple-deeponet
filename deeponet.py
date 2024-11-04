import torch
import torch.nn as nn

class BasicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.network(x)

class DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim):
        super().__init__()
        self.branch_net = BasicNet(branch_dim[0], branch_dim[1], branch_dim[2])
        self.trunk_net = BasicNet(trunk_dim[0], trunk_dim[1], trunk_dim[2])
        
    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        output = torch.sum(branch_output * trunk_output, dim=1) 
        return output
