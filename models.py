import torch
from networks import * 
from sequnet import Sequnet
from abc import ABC, abstractmethod

class Model(ABC):
    
    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print("{0} GPUs are detected.".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net)

        if torch.cuda.is_available():
            self.net.cuda()
    
    def get_model_path(self, model_dir, step=0):

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(self.name, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(self.name))
        
    
    def load(self, model_dir, step=0):
        print("Loading model from {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        self.net.load_state_dict(torch.load(model_path))

    def save(self, model_dir, step=0):
        print("Saving model into {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        torch.save(self.net.state_dict(), model_path)

    @abstractmethod
    def _loss(self):
        pass

    @abstractmethod
    def train(self, inputs, targets):
        pass

    @abstractmethod
    def train(self):
        pass



class MyWavenet_model(Model):
    def __init__(self, args) -> None:   
        self.name = "MyWavenet"
        self.net = MyWavenet(args.channels, args.n_layers)
        self.receptive_field = self.calc_receptive_field(args.n_layers)
        self.loss = self._loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), args.lr)

        self._prepare_for_gpu()

    
    def calc_receptive_field(self, n_layers):
        # this is actually the receptive_field - 1
        return int(sum([2**i for i in range(n_layers)]))

    
    def _loss(self):
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss
    
    def train(self, inputs, targets):
        outputs = self.net(inputs) # slow
        

        loss = self.loss(outputs,
                         targets.long())
        self.optimizer.zero_grad()

        loss.backward() #slow

        self.optimizer.step()

        return loss.item()

    def val(self):
        pass


class Rawnet_model(Model):
    def __init__(self, args) -> None:
        self.name = "Rawnet"
        self.net = Rawnet(args.n_layers)

        self.receptive_field = self.calc_receptive_field(args.n_layers)
        self.loss = self._loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), args.lr)

        self._prepare_for_gpu()

    
    def calc_receptive_field(self, n_layers):
        # this is actually the receptive_field - 1
        return int(sum([2**i for i in range(n_layers)]))

    
    def _loss(self):
        loss = torch.nn.MSELoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss
    
    def train(self, inputs, targets):

  
        outputs = self.net(inputs) # slow
        

        loss = self.loss(outputs,
                         targets)
        self.optimizer.zero_grad()

        loss.backward() #slow

        self.optimizer.step()

        return loss.item()

    def val(self):
        pass


class Wavenet_model(Model):
    def __init__(self, args) -> None:
        
        self.name = "Wavenet"
        self.net = Wavenet(layers=args.n_layers, blocks=args.n_blocks, classes=args.channels)
        self.receptive_field = self.calc_receptive_field(args.n_layers, args.n_blocks)
        self.loss = self._loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), args.lr)

        self._prepare_for_gpu()

    
    def calc_receptive_field(self,n_layers, n_blocks):
        # this is actually the receptive_field - 1
        return int(sum([2**i for i in range(n_layers)]))*n_blocks

    
    def _loss(self):
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss
    
    def train(self, inputs, targets):
        outputs = self.net(inputs) # slow
        

        loss = self.loss(outputs,
                         targets.long())
        self.optimizer.zero_grad()

        loss.backward() #slow

        self.optimizer.step()

        return loss.item()

    def val(self):
        pass


class Wavenet_hx_model(Model):

    def __init__(self, args) -> None:
        self.name = "Wavenet_hx"
        self.net = Wavenet_hx(layers=args.n_layers, blocks=args.n_blocks, classes=args.channels)
        self.receptive_field = self.calc_receptive_field(args.n_layers, args.n_blocks)
        self.loss = self._loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), args.lr)

        self._prepare_for_gpu()

    
    def calc_receptive_field(self,n_layers, n_blocks):
        # this is actually the receptive_field - 1
        return int(sum([2**i for i in range(n_layers)]))*n_blocks

    
    def _loss(self):
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss
    
    def train(self, inputs, targets):
        outputs = self.net(inputs) # slow
        
        loss = self.loss(outputs,
                         targets.long())
        self.optimizer.zero_grad()

        loss.backward() #slow

        self.optimizer.step()

        return loss.item()

    def val(self):
        pass



class Sequnet_model(Model):

    def __init__(self, args) -> None:
        self.name = "Sequnet"
        self.set_num_channels(args.channels,args.n_layers)
        self.net = Sequnet(args.channels, self.num_channels, args.channels, kernel_size=3, causal=True, dropout=0.2, target_output_size=None)
        self.loss = self._loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), args.lr)

        self._prepare_for_gpu()

    
    def set_num_channels(self, channels,n_layers):
        num_channels = [channels]
        for i in range(n_layers):
            num_channels.append(channels * 2)

        num_channels.append(channels)

        self.num_channels = num_channels

    
    def _loss(self):
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss
    
    def train(self, inputs, targets):
        outputs = self.net(inputs) # slow
        
        loss = self.loss(outputs,
                         targets.long())
        self.optimizer.zero_grad()

        loss.backward() #slow

        self.optimizer.step()

        return loss.item()

    def val(self):
        pass


    