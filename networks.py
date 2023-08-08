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


### M Y W A V E N E T
class MyWavenet(nn.Module):
    def __init__(self, channels, n_layers) -> None:
        super(MyWavenet, self).__init__()
        self.stack = CustomStack(channels, n_layers)
        self.dense = Dense(channels)

        self.channels = channels
        
    
    def forward(self, hx):
        out = self.stack(hx)
        out = self.dense(out)
    
        return out
    
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
        
        return out
    

########################################################

class Wavenet(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=10,
                 blocks=5,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=512,
                 classes=256,
                 output_length=32,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False,
                 fast=False):

        super(Wavenet, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.fast = fast

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias,
                                                   dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias,
                                                 dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=skip_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        self.output_size = output_length
        self.receptive_field = receptive_field
        self.input_size = receptive_field + output_length - 1

    def forward(self, input, mode="normal"):
        if mode == "save":
            self.inputs = [None]* (self.blocks * self.layers)

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            if mode == "save":
                self.inputs[i] = x[:,:,-(dilation*(self.kernel_size-1) + 1):]
            elif mode == "step":
                self.inputs[i] = torch.cat([self.inputs[i][:,:,1:], x], dim=2)
                x = self.inputs[i]

            # dilated convolution
            residual = x

            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = self.skip_convs[i](x)
            if skip is not 0:
                skip = skip[:, :, -s.size(2):]
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, dilation * (self.kernel_size - 1):]

        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x














# wavenet = Wavenet(256,5)
# rawnet = Rawnet(5)

# for name, param in wavenet.named_parameters():
#     if param.requires_grad:
#         print(f"Layer name: {name}, Trainable Parameters: {param.shape}")

# print("###########################")

# for name, param in rawnet.named_parameters():
#     if param.requires_grad:
#         print(f"Layer name: {name}, Trainable Parameters: {param.shape}")
    