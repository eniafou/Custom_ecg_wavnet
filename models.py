import torch
from networks import * 
from abc import ABC, abstractmethod

class Model(ABC):
    
    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print("{0} GPUs are detected.".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net)

        if torch.cuda.is_available():
            self.net.cuda()
    
    def get_model_path(self, model_dir, step=0):
        basename = 'wavenet'

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(basename))
        
    
    def load(self, model_dir, step=0):
        print("Loading model from {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        self.load_state_dict(torch.load(model_path))

    def save(self, model_dir, step=0):
        print("Saving model into {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        torch.save(self.state_dict(), model_path)

    @abstractmethod
    def _loss(self):
        pass

    @abstractmethod
    def train(self, inputs, targets):
        pass

    @abstractmethod
    def train(self):
        pass



class Wavenet_model(Model):
    def __init__(self, args) -> None:

        self.net = Wavenet(args.channels, args.n_layers)
        self.receptive_fields = self.calc_receptive_fields(args.n_layers)
        self.loss = self._loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), args.lr)

        self._prepare_for_gpu()

    
    def calc_receptive_fields(self, n_layers):
        return int(sum([2**i for i in range(n_layers)]))

    
    def _loss(self):
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss
    
    def train(self, inputs, targets):
        outputs = self.net(inputs) # slow
        

        loss = self.loss(outputs.view(-1, self.net.channels),
                         targets.long().view(-1))
        self.optimizer.zero_grad()

        loss.backward() #slow

        self.optimizer.step()

        return loss.item()

    def val(self):
        pass


class Rawnet_model(Model):
    def __init__(self, args) -> None:
    
        self.net = Rawnet(args.n_layers)

        self.receptive_fields = self.calc_receptive_fields(args.n_layers)
        self.loss = self._loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), args.lr)

        self._prepare_for_gpu()

    
    def calc_receptive_fields(self, n_layers):
        return int(sum([2**i for i in range(n_layers)]))

    
    def _loss(self):
        loss = torch.nn.MSELoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss
    
    def train(self, inputs, targets):

  
        outputs = self.net(inputs) # slow
        

        loss = self.loss(outputs.view(-1),
                         targets.view(-1))
        self.optimizer.zero_grad()

        loss.backward() #slow

        self.optimizer.step()

        return loss.item()

    def val(self):
        pass