from network import MyWaveNet
from data import Dataset
import torch.utils.data as data
import torch

class WNModel():
    def __init__(self, channels, n_layers, lr) -> None:
        
        self.net = MyWaveNet(channels, n_layers)
        self.receptive_fields = self.calc_receptive_fields(n_layers)
        self.channels = channels
        self.loss = self._loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)

        self._prepare_for_gpu()

    def calc_receptive_fields(self, n_layers):
        return int(sum([2**i for i in range(n_layers)]))
    
    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print("{0} GPUs are detected.".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net)

        if torch.cuda.is_available():
            self.net.cuda()
    
    def _loss(self):
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss
    
    def train(self, inputs, targets):
        outputs = self.net(inputs) # slow
        

        loss = self.loss(outputs.view(-1, self.channels),
                         targets.long().view(-1))
        self.optimizer.zero_grad()

        loss.backward() #slow

        self.optimizer.step()

        return loss.item()

class Trainer():
    def __init__(self, args) -> None: 
        """
        args contains :
        channels, n_layers, lr, data_dir, batch_size, num_epoch, data_len
        
        """
        self.args = args
        self.model = WNModel(args.channels, args.n_layers, args.lr)
        self.dataset = Dataset(args.data_dir, self.model.receptive_fields, args.channels, data_len = args.data_len)

        self.data_loader = data.DataLoader(self.dataset, batch_size=args.batch_size,shuffle=True)

    def run(self):
        
        num_epoch = self.args.num_epoch
        loss_per_epoch = []
        for epoch in range(num_epoch):
            for i, (inputs, targets) in enumerate(self.data_loader):
                print(i)
                loss = self.model.train(inputs, targets)
                if True :#(i+1)%5 == 0:
                    print('[{0}/{1}] loss: {2}'.format(epoch + 1, num_epoch, loss))
            
            loss_per_epoch.append(loss)
        
        print(loss_per_epoch)

class JsonConfig():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

if __name__ == '__main__':
    args = {"channels" : 256, "n_layers" : 5, "lr" : 0.01 , "data_dir" : "../data/ptb-xl/", "batch_size" : 32, "num_epoch" : 100, "data_len" : 100}
    args = JsonConfig(**args)

    trainer = Trainer(args)

    trainer.run()