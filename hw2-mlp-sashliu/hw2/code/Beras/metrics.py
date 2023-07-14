import numpy as np

from .core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        """Categorical accuracy forward pass!"""
        super().__init__()
        # TODO: Compute and return the categorical accuracy of your model given the output probabilities and true labels
        maxp = lambda x: np.argmax(x) # idenitfy the index of the maximum value
        trueMax = np.array([maxp(i) for i in labels]) 
        predMax = np.array([maxp(i) for i in probs])
        # count of accurately predicted records divided by the count of total records
        summ = sum(predMax == trueMax)/len(predMax)
        return summ