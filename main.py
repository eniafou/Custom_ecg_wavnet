from networks import *
from models import *
from data_utils import *
import torch.utils.data as data
from utils import JsonConfig
import logging
import time

logging.basicConfig(filename='train.log', level=logging.INFO,
                    format='%(asctime)s:%(message)s')



class Trainer():
    def __init__(self, model, dataset, num_epoch, batch_size) -> None: 
        
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.model = model
        self.dataset = dataset

        self.data_loader = data.DataLoader(self.dataset, batch_size=self.batch_size,shuffle=True)

    def run(self):
        
        num_epoch = self.num_epoch
        loss_per_epoch = []
        # val_loss_per_epoch = []
        for epoch in range(num_epoch):
            for i, (inputs, targets) in enumerate(self.data_loader):
                loss = self.model.train(inputs, targets)
                if True :#(i+1)%5 == 0:
                    print('[iter [{0}] epoch [{1}/{2}]] loss: {3}'.format(i + 1, epoch + 1, num_epoch, loss))
            
            # her you should calculate a loss over a validation set
            # val_loss = self.model.val()

            # val_loss_per_epoch.append(val_loss)
            loss_per_epoch.append(loss)
        
        return loss_per_epoch # the training changes the provided net in time, we are talking about the same object



if __name__ == '__main__':
    args = {"channels" : 256, "n_layers" : 5, "lr" : 0.01 , "data_dir" : "../data/ptb-xl/", "batch_size" : 32, "num_epoch" : 100, "data_len" : 100}
    logging.info("Started training using the following arguments : \n" + str(args))
    args = JsonConfig(**args)

    # model = Wavenet_model(args)
    # dataset = Dataset(args.data_dir, model.receptive_fields, args.channels, args.data_len)

    model = Rawnet_model(args)
    dataset = RawDataset(args.data_dir, model.receptive_fields)

    trainer = Trainer(model, dataset, args.num_epoch, args.batch_size)

    start_time = time.time()

    loss_per_epoch = trainer.run()

    end_time = time.time()
    duration = end_time - start_time

    logging.info("Training duration : " + str(duration) + " s")
    logging.info("The loss per epoch : " + str(loss_per_epoch))