import torch
import torch.nn as nn

class BasicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim):
        # branch_input_dim 应为含三个整数的数组[a, b, c]
        super().__init__()
        self.branch_net = BasicNet(branch_dim[0], branch_dim[1], branch_dim[2])
        self.trunk_net = BasicNet(trunk_dim[0], trunk_dim[1], trunk_dim[2])
        
    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        output = torch.sum(branch_output * trunk_output, dim=1) # 点乘输出
        return output  