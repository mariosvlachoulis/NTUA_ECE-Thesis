import numpy as np
from torch.utils import data


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


# This is a Python code snippet that defines two classes, InfiniteSampler and InfiniteSamplerWrapper, 
# which can be used as a custom data sampler for PyTorch's DataLoader class.

# The InfiniteSampler class generates an infinite sequence of random indices from a given range of values. 
# This is useful when you want to iterate over a dataset multiple times in a random order. The InfiniteSamplerWrapper class wraps 
# the InfiniteSampler class and provides a PyTorch-compatible interface for it.

# The __init__ method of the InfiniteSamplerWrapper class takes a data_source argument, which is a PyTorch dataset object. 
# The num_samples attribute of the dataset is used to determine the number of samples in the dataset.

# The __iter__ method of the InfiniteSamplerWrapper class returns an iterator that generates an infinite sequence of random indices 
# using the InfiniteSampler class.

# The __len__ method of the InfiniteSamplerWrapper class returns a very large number (2 ** 31) to ensure that the DataLoader class
# does not terminate prematurely.

# Overall, this code can be used to create a custom data sampler for PyTorch's DataLoader class that allows you to iterate over a dataset
# an infinite number of times in a random order.