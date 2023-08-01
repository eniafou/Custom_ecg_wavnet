from network import MyWaveNet
from data_utils import Dataset
import torch.utils.data as data
import torch
from utils import JsonConfig
import logging

logging.basicConfig(filename='train.log', level=logging.INFO,
                    format='%(asctime)s:%(message)s')

class WNModel():
    def __init__(self, net, lr) -> None:
        
        self.net = net
        self.loss = self._loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)

        self._prepare_for_gpu()

    
    
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
        

        loss = self.loss(outputs.view(-1, self.net.channels),
                         targets.long().view(-1))
        self.optimizer.zero_grad()

        loss.backward() #slow

        self.optimizer.step()

        return loss.item()

    def val(self):
        pass

class Trainer():
    def __init__(self, net, args) -> None: 
        """
        args contains :
        channels, n_layers, lr, data_dir, batch_size, num_epoch, data_len
        
        """
        self.args = args
        self.model = WNModel(net, args.lr)
        self.dataset = Dataset(args.data_dir, net.receptive_fields, args.channels, data_len = args.data_len)

        self.data_loader = data.DataLoader(self.dataset, batch_size=args.batch_size,shuffle=True)

    def run(self):
        logging.info("Started training using the following arguments : " + str(self.args))
        num_epoch = self.args.num_epoch
        loss_per_epoch = []
        val_loss_per_epoch = []
        for epoch in range(num_epoch):
            for i, (inputs, targets) in enumerate(self.data_loader):
                loss = self.model.train(inputs, targets)
                if True :#(i+1)%5 == 0:
                    print('[{0}/{1}] loss: {2}'.format(epoch + 1, num_epoch, loss))
            
            # her you should calculate a loss over a validation set
            val_loss = self.model.val()

            val_loss_per_epoch.append(val_loss)
            loss_per_epoch.append(loss)
        logging.info("The loss per epoch : " + str(loss_per_epoch))
        return loss_per_epoch # the training changes the provided net in time, we are talking about the same object



if __name__ == '__main__':
    args = {"channels" : 256, "n_layers" : 5, "lr" : 0.01 , "data_dir" : "../data/ptb-xl/", "batch_size" : 32, "num_epoch" : 100, "data_len" : 100}
    args = JsonConfig(**args)
    
    net = MyWaveNet(args.channels, args.n_layers)
    trainer = Trainer(net , args)

    trainer.run()