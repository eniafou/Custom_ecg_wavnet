import torch
import torch.nn as nn
import os

class DilatedCausalConv(nn.Module):
    def __init__(self, channels,dilation) -> None:
        super(DilatedCausalConv, self).__init__()
        self.conv = nn.Conv1d(channels,channels, 2,1,padding = 0, dilation = dilation, bias=False)
    
    def init_weights(self,value):
        self.conv.weight.data.fill_(value)

    def forward(self, x):
        ## the input x should be proparly padded beforehand
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
        self.layers = nn.ModuleList()
        for dilation in self.dilations:
            self.layers.append(self._block(channels, dilation))
    
    @staticmethod
    def _block(channels, dilation):
        block = CustomBlock(channels, dilation)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)

        if torch.cuda.is_available():
            block.cuda()

        return block
    
    def generate_dilations(self, n_layers):
        return [2**i for i in range(n_layers)]
    
    def forward(self, hx):
        out = hx
        for layer in self.layers:
            out = layer(out)

        return out[1]


### W A V E N E T
class Wavenet(nn.Module):
    def __init__(self, channels, n_layers) -> None:
        super(Wavenet, self).__init__()
        self.stack = CustomStack(channels, n_layers)
        self.dense = Dense(channels)

        self.channels = channels
        
    
    def forward(self, hx):
        out = self.stack(hx)
        out = self.dense(out)
    
        return out.transpose(1,2).contiguous()
    
######################################################################


class RawBlock(nn.Module):
    def __init__(self, dilation) -> None:
        super(RawBlock, self).__init__()
        self.h_dilatedCausalConv = DilatedCausalConv(1, dilation)
        self.x_dilatedCausalConv = DilatedCausalConv(1, dilation)
        self.tanh = nn.Tanh()

    def forward(self, hx):
        h, x = hx
   

        h = self.h_dilatedCausalConv(h)
        x = self.x_dilatedCausalConv(x)
        
   
        out = self.tanh(h + x)
        return h, out


class RawStack(nn.Module):
    def __init__(self, n_layers) -> None:
        super(RawStack, self).__init__()
        self.dilations = self.generate_dilations(n_layers)
        self.layers = nn.ModuleList()
        for dilation in self.dilations:
            self.layers.append(self._block(dilation))
    
    @staticmethod
    def _block(dilation):
        block = RawBlock(dilation)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)

        if torch.cuda.is_available():
            block.cuda()

        return block
    
    def generate_dilations(self, n_layers):
        return [2**i for i in range(n_layers)]
    
    def forward(self, hx):
        out = hx
        for layer in self.layers:
            out = layer(out)

        return out[1]

### R A W N E T
class Rawnet(nn.Module):
    def __init__(self, n_layers) -> None:
        super(Rawnet, self).__init__()
        self.stack = RawStack(n_layers)
       
    def forward(self, hx):
        out = self.stack(hx)
        
        return out.transpose(1,2).contiguous()
    
# wavenet = Wavenet(256,5)
# rawnet = Rawnet(5)

# for name, param in wavenet.named_parameters():
#     if param.requires_grad:
#         print(f"Layer name: {name}, Trainable Parameters: {param.shape}")

# print("###########################")

# for name, param in rawnet.named_parameters():
#     if param.requires_grad:
#         print(f"Layer name: {name}, Trainable Parameters: {param.shape}")
    