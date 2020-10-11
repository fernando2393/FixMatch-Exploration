import random
import numpy as np
from collections import namedtuple

augment = {}  # Create a dictionary where the information from each transformation will be stored. The key will be the name of the transformation
augment_tuple = namedtuple('augment_tuple', ('transformation', 'bins'))


def get_augmentation_bins(bins, transform):
    augment[transform] = augment_tuple()


class CTAugment:
    def __init__(self, depth=2, t=0.8, ro=0.99):
        self.ro = ro
        self.depth = 2
        self.t = t
        self.rates = {}  # this is the variable thar will store all the weights for each of the transformations
        for function_name, op in augment.items():
            self.rates[function_name] = list([np.ones(x, 'f') for x in op.bins])  # Each transformation has different parameters. The range that these parameters can take is divided into bins.
                                                                                  # This list will contain as many elements (arrays) as parameters and each element/array will have as many ones as in size.
    

    def bin_weights_to_p(self, rate):  # Here we will receive each rate/bin weight and from a specific transformation and convert it to a probability and then set it to 0 if it is not above the trheshold self.t.
        probability = rate + (1 - self.ro)
        probability = probability / probability.max()
        probability[probability < self.t] = 0
        
        return probability


    def policy(self, probe):  # This function will define the policy according to which the weights of the bins will be updated
        transformation_list = list(augment.keys())
        new_bins = list()
        if probe:
            for _ in range(self.depth):
                transformation = random.choice(transformation_list)
                bins = self.rates[transformation] # Transformation parameters bins (this will return as many arrays as parameters). Each array will contain 17 elements (bins). It's a way of discretizing the parameters ranges
                sample = np.random.uniform(0, 1, len(bins)) 
                new_bins.append(augment_tuple(transformation, sample.tolist()))
                
            return new_bins

        for _ in range(self.depth):
            new_bin_elements = list()
            transformation = random.choice(transformation_list)
            bins = self.rates[transformation]
            sample = np.random.uniform(0, 1, len(bins))
            new_bins.append(augment_tuple(transformation, sample.tolist()))
            for rand_number, bin_ in zip(sample, bins):  # bin_ corresponds to each bin array per transformation parameter
                p = self.bin_weights_to_p(bin_)
                selected_bin = np.random.choice(p.shape[0], p=(p / p.sum())) 
                new_bin_elements.append((selected_bin + rand_number) / p.shape[0])  # Choose random number within bin range
            new_bins.append(augment_tuple(transformation, new_bin_elements))

        return new_bins