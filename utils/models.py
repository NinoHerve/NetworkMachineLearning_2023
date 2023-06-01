import torch.nn as nn


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


class CNN2(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.conv2d  = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,1))
        self.bn      = nn.BatchNorm2d(1, track_running_stats=False)
        self.linear  = nn.Linear(1*128*625, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = x.view(-1,1,128,625)
        x = self.conv2d(x)
        x = self.bn(x)
        x = x.view(-1,1*128*625)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x
    
    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv1d[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.output[0].weight, nonlinearity='relu')