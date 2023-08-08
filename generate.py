from data_utils import Dataset, one_hot_decode
import torch
from utils import JsonConfig
from networks import MyWaveNet


# net
# h, x
# h and x should be cuda variables

# h is fixed size, x is changing (++)

# can the model run with an empty x ?

# you didin't implement softmax at the end of the net, you should add it

# --- input : net





class Generator():
    def __init__(self, net, args) -> None:
       self.net = net
       self.softmax = torch.nn.Softmax(dim=1)
       self.dataset = Dataset(args.data_dir, net.receptive_field, args.channels, args.data_len)
    
    @staticmethod
    def _prepare(dataset , index):
        h, x = dataset[index][0]
        return h.unsqueeze(0), x.unsqueeze(0)
    
    def simple_generate(self, index, mu_decoded = False):
        h, x = self._prepare(self.dataset, index)
        out = self.net((h,x))
        out = self.softmax(out)
        out = out[0].cpu().data.numpy()
        out = one_hot_decode(out, axis=1)

        y = self.dataset[index][1].cpu().data.numpy()
        if mu_decoded:
            pass
            
        return out, y
    
    @staticmethod
    def f(x , new):
        j = new.argmax(dim = 1).item()
        new = torch.tensor([[[0 for i in range(new.size(1))]]])
        new.transpose_(1,2)
        new[:,j,:] = 1
        torch.cat((x, new), dim=2)
    
    def generate(self, index):
        h, _ = self._prepare(self.dataset, index)
        x = torch.tensor([[[0 for i in range(net.channels)]]])
        x = x.transpose(1,2)

        while x.size(2) <= h.size(2) + 1: 
            out = self.net((h,x))
            out = self.softmax(out) # [1,c, 1] x(b, c, n) n = 1, c = 1 

            x = self.f(x, out[:,:,-1])
        
        gen = x[:,:,1:]
        gen = gen.transpose(1,2)
        gen = gen[0].cpu().data.numpy()
        gen = one_hot_decode(gen, axis=1)

        return gen
            
        


if __name__ == '__main__':
    args = {"channels" : 256, "n_layers" : 5, "lr" : 0.01 , "data_dir" : "../data/ptb-xl/", "batch_size" : 32, "num_epoch" : 100, "data_len" : 100}
    args = JsonConfig(**args)

    net = MyWaveNet(args.channels, args.n_layers)
    generator = Generator(net, args)

    

    print(generator.generate(0))