class JsonConfig():
    def __init__(self, **kwargs):
        self.args = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def toStr(self):
        return self.args



class Tester():
    def __init__(self) -> None:
        pass





"""
## things to try

- adding batchnorm
- changing activation functions (leakyrelu, sigmoid * tanh)
- filtering data
- using the 5000 dataset

## things to do
- data visualization
- add a validaiton loss curve
- add some metrics evolution curve

"""

"""
create a new type of dataset (raw values)
cannels inside the network are increasing then decreasing

"""
    