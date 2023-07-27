class JsonConfig():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)



class Tester():
    def __init__(self) -> None:
        pass

    