class Attach(object):
    def __init__(self, dst):
        self.dst = dst

    def __call__(self, obj, name=None):
        if name is None:
            # Automatically get names of functions and classes
            name = obj.__name__
        if hasattr(self.dst, name):
            raise RuntimeError(
                f"{self.dst} already has the attribute {name}, which is {getattr(self.dst, name)}."
            )
        setattr(self.dst, name, obj)
        if hasattr(self.dst, '__all__'):
            self.dst.__all__.append(name)
        return obj

    @staticmethod
    def to(dst):
        return Attach(dst)
