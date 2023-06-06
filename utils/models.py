import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv, global_add_pool, global_sort_pool, global_max_pool, global_mean_pool, LayerNorm
from torch_geometric.nn.norm import graph_norm
import torch.nn.functional as F
import torch

class CNN(nn.Module):

    def __init__(self, n_channels=128, n_kernels=8):
        super().__init__()
        self.n_channels = n_channels 
        self.n_kernels = n_kernels
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_kernels, kernel_size=32),
            nn.BatchNorm1d(n_kernels, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.encode = nn.Sequential(
            nn.Linear(n_channels*n_kernels, 40),
            nn.LeakyReLU(),
        )
        self.pool = nn.AvgPool2d(kernel_size=(1,128), stride=(1,64))
        self.output = nn.Sequential(
            nn.Linear(40*8, 1),
            nn.Sigmoid(),
        )
        

    def forward(self, x):
        n_times = x.shape[-1]
        x = x.view(-1,1,n_times)
        x = self.conv1d(x)

        n_times = x.shape[-1]
        x = x.view(-1,self.n_channels*self.n_kernels)
        x = self.encode(x)

        x = x.view(-1, 40, n_times)
        x = self.pool(x)

        x = x.view(-1, 40*8)
        x = self.output(x)

        return x
    
    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv1d[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.encode[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.output[0].weight, nonlinearity='relu')


class LIN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lin = nn.Linear(input_size, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.lin(x)
        x = self.sig(x)
        return x 
    

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
        )
        self.hidden = nn.Sequential(
            nn.Linear(128, 128),
            #nn.LeakyReLU(),
            nn.Dropout(),
        )
        self.output = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x
    
    def init_weights(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.input[0].weight, gain)
        nn.init.xavier_normal_(self.hidden[0].weight, gain)
        nn.init.xavier_normal_(self.output[0].weight)


class GNN(nn.Module):
    def __init__(self, in_channels=4, num_layers=3, hid_channels=64):
        super().__init__()
        self.conv1 = GATConv(in_channels, hid_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hid_channels, hid_channels))
        self.lin1 = nn.Linear(hid_channels, hid_channels)
        self.lin2 = nn.Linear(hid_channels, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = F.dropout(x, p=0.5) 
        x = self.sig(x)
         
        return x
    
class GNN2(nn.Module):
    def __init__(self, in_channels=4, hid_channels=64):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hid_channels, add_self_loops=False)
        self.norm  = LayerNorm(hid_channels)
        self.lin1 = nn.Linear(hid_channels, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.norm(x)
        x = F.dropout(x, p=0.7)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.dropout(x, p=0.7)
        x = self.sig(x)
        return x
    

class GNN3(nn.Module):
    def __init__(self, in_channels=4, hid_channels=64, num_layers=3):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hid_channels, add_self_loops=False)
        self.norm  = LayerNorm(hid_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hid_channels, hid_channels))
        
        self.lin1 = nn.Linear(128*hid_channels, 128)
        self.lin2 = nn.Linear(128, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #x = F.dropout(x, p=0.3)
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.norm(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        #x = F.dropout(x, p=0.5)
        x = self.linearize(x, batch)
        x = F.leaky_relu(self.lin1(x))
        #x = F.dropout(x, p=0.5)
        x = self.lin2(x)
        x = self.sig(x)
        return x
    
    def linearize(self, x, batch):
        features = []
        for sample in torch.unique(batch):
            rows = (batch == sample)
            features.append(x[rows].flatten())

        return torch.stack(features)
