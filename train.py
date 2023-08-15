from networks import *
from models import *
from data_utils import *
import torch.utils.data as data
from utils import JsonConfig
import time
#from torch.utils.tensorboard import SummaryWriter





class Trainer():
    def __init__(self, model, dataset, num_epoch, batch_size, out_dir, log_dir) -> None: 
        
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.model = model
        self.dataset = dataset
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.data_loader = data.DataLoader(self.dataset, batch_size=self.batch_size,shuffle=True)

    def run(self):
        #writer = SummaryWriter(self.log_dir)

        num_epoch = self.num_epoch
        loss_per_epoch = []
        # val_loss_per_epoch = []
        for epoch in range(num_epoch):
            for i, (inputs, targets) in enumerate(self.data_loader):
                loss = self.model.train(inputs, targets)

                #writer.add_scalar('Loss/train', loss, epoch * len(self.data_loader) + i)
                if True :#(i+1)%5 == 0:
                    print('[iter [{0}] epoch [{1}/{2}]] loss: {3}'.format(i + 1, epoch + 1, num_epoch, loss))
            
            # her you should calculate a loss over a validation set
            # val_loss = self.model.val()

            # val_loss_per_epoch.append(val_loss)

            loss_per_epoch.append(loss)
        
        #writer.close()
        self.model.save(self.out_dir, step = int(time.time()))
        
        return loss_per_epoch # the training changes the provided net in time, we are talking about the same object



log_dir = "./logs"
out_dir = './output'
data_dir = "../data/ptb-xl/"

if __name__ == '__main__':
    
    args = {"channels" : 256, "n_layers" : 2,"n_blocks":3, "lr" : 0.003 , "data_dir" : data_dir, "batch_size" : 32, "num_epoch" : 100, "data_len" : 10, "conditioned": False,"out_dir": out_dir, "log_dir":log_dir}
    args = JsonConfig(**args)

    # model = MyWavenet_model(args)
    # dataset = Dataset(args.data_dir, model.receptive_field, args.channels, args.data_len)

    # model = Rawnet_model(args)
    # dataset = RawDataset(args.data_dir, model.receptive_field,data_len=1000)

    # model = Wavenet_model(args)
    # dataset = Dataset(args.data_dir, model.receptive_field, args.channels, args.data_len, conditioned=args.conditioned)

    # model = Wavenet_hx_model(args)
    # dataset = Dataset(args.data_dir, model.receptive_field, args.channels, data_len = args.data_len)

    model = Sequnet_model(args)
    dataset = Dataset(args.data_dir, in_channels=args.channels, conditioned=args.conditioned, data_len=args.data_len)


    trainer = Trainer(model, dataset, args.num_epoch, args.batch_size, args.out_dir, args.log_dir)

    start_time = time.time()

    loss_per_epoch = trainer.run()

    end_time = time.time()
    duration = end_time - start_time




"""
writer = SummaryWriter(log_dir)
dataset = Dataset(data_dir, 15, 256, 10)
model = MyWavenet(256, 4)

# Dummy input for visualization (adjust according to your input size)
# dummy_input_1 = torch.randn(1, 256, 1000)
# dummy_input_2 = torch.randn(1,256,1000)
# dummy_input = tuple((dummy_input_1,dummy_input_2))

# Logging network structure to TensorBoard
print(model(dataset[0]).shape)
writer.add_graph(model, dataset[0])

"""
