import torch
import torch.nn as nn

class DilatedCausalConv(nn.Module):
    def __init__(self, channels,dilation) -> None:
        super(DilatedCausalConv, self).__init__()
        self.conv = nn.Conv1d(channels,channels, 2,1,padding = 0, dilation = dilation, bias=False)
    
    def init_weights(self,value):
        self.conv.weight.data.fill_(value)

    def forward(self, x):
        ## the input x should be proparly padded beforhand
        out = self.conv(x)
        return out

class DilatedConv(nn.Module):
    def __init__(self, channels,dilation) -> None:
        super(DilatedConv, self).__init__()
        self.conv = nn.Conv1d(channels,channels,3,1,"same",dilation, bias = False)

    def forward(self, x):
        out = self.conv(x)
        return out
    
class CustomBlock(nn.Module):
    def __init__(self, channels,dilation) -> None:
        super(CustomBlock, self).__init__()
        self.dilatedCausalConv = DilatedCausalConv(channels, dilation)
        self.v = nn.Conv1d(channels,channels,1) # with or without bias ?
        self.relu = nn.ReLU() # if this work change this to a sigmoid * tanh


    def init_weights(self):
        self.dilatedCausalConv.init_weights(1)
        self.v.weight.data.fill_(1)
        self.v.bias.data.fill_(0)


    def forward(self, hx):
        h, x = hx
        
        x = self.dilatedCausalConv(x)
        h = self.v(h)

        out = self.relu(h[:,:,-x.size(2):] + x)
        return h, out
        
class Dense(nn.Module):
    def __init__(self, channels) -> None:
        super(Dense, self).__init__()
        self.conv1 = nn.Conv1d(channels,channels, 1)
        self.conv2 = nn.Conv1d(channels,channels, 1)

        self.relu = nn.ReLU()
        # self.softmax = torch.nn.Softmax(dim=1)
    def forward(self,x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)

        # output = self.softmax(output) # I think you can eleminate this during training, you can use it when generating

        return output

class CustomStack(nn.Module):
    def __init__(self, channels, n_layers) -> None:
        super(CustomStack, self).__init__()
        self.dilations = self.generate_dilations(n_layers)
        self.layers = []
        for dilation in self.dilations:
            self.layers.append(CustomBlock(channels, dilation))

    def generate_dilations(self, n_layers):
        return [2**i for i in range(n_layers)]
    
    def forward(self, hx):
        out = hx
        for layer in self.layers:
            out = layer(out)

        return out[1]
    
class MyWaveNet(nn.Module):
    def __init__(self, channels, n_layers) -> None:
        super(MyWaveNet, self).__init__()
        self.stack = CustomStack(channels, n_layers)
        self.dense = Dense(channels)
    
    def forward(self, x):
        out = self.stack(x)
        out = self.dense(out)
    
        return out.transpose(1,2).contiguous()