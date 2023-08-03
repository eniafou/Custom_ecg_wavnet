import unittest
import networks
import torch
from data_utils import Dataset

class TestNetwork(unittest.TestCase):
    """
    you should always check the parameter of your network
    """
    # for name, param in wavenet.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer name: {name}, Trainable Parameters: {param.shape}")


    def setUp(self):
        self.causalConv = networks.DilatedCausalConv(1, 1)
        self.causalConv.init_weights(1)
        self.x = torch.ones((1,100)) # c , l

    def test_dumb_forward_DilatedCausal(self):
        print("dilated causal output size :")
        print(self.causalConv(self.x).shape)
    
    def test_forward_DilatedCausal(self):
        out = self.causalConv(self.x)
        self.assertEqual(out[:,0], 2)

    def test_custom_block(self):
        block = networks.CustomBlock(1, 1)
        block.init_weights()

        x = torch.ones((1, 1, 100))
        h = torch.ones((1, 1, 100))

        print(block((h,x)))

    def test_dense(self):
        x = torch.ones((1,1,69)) # b c l
        dense = networks.Dense(1)
        out = dense(x)
        
        print(out[0,:,:].sum())
        print(out[:,0,:].sum())
        print(out[:,:,0].sum())

    def test_customStack(self):
        stack = networks.CustomStack(1, 5)
        # print(stack.dilations)
        x = torch.ones((1, 1, 100))
        h = torch.ones((1, 1, 100))

        print(stack((h,x)).size())
    



class TestData(unittest.TestCase):
    def test_input_values(self):
        dataset = Dataset("../data/ptb-xl/", receptive_fields= 10, in_channels= 5, data_len = 10)

        hx, target = dataset[0]
        print("target :")
        print(target) # should be (1000, 1)
        print(target.shape)

        h,x = hx
        print("h :")
        print(h) # (5, 1010)
        print(h.shape)

        print("x ")
        print(x) # (5, 1010)
        print(x.shape)

class testGenerator():
    pass



# if __name__ == "__main__":
#     unittest.main()
