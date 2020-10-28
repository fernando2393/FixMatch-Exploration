"""
Adapted from original Google Research Code:
https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
"""
import inspect
import random
import numpy as np
import Transformations
import torch


def linenum(x):
    return x.__code__.co_firstlineno


class CTAugment:
    def __init__(self, depth=2, t=0.8, ro=0.99):
        self.ro = ro
        self.policy_list = []
        self.depth = depth
        self.t = t
        self.rates = {}  # this is the variable thar will store all the weights for each of the transformations
        self.registry = {}
        transformations = dict(inspect.getmembers(Transformations, inspect.isfunction))  # Get function names
        transformations = sorted(transformations.values(), key=lambda x: linenum(x))
        transformations = [transformations[i].__name__ for i in range(len(transformations))]
        for i in range(len(Transformations.BIN_LIST)):  # Iterate over function tuples
            self.registry[transformations[i]] = Transformations.BIN_LIST[i]  # Fill registry with name and bins
        for function_name, bin_list in self.registry.items():
            self.rates[function_name] = list([np.ones(x, 'f') for x in bin_list])  # Each transformation has different
            # parameters. The range that these parameters can take is divided into bins. This list will contain as
            # many elements (arrays) as parameters and each element/array will have as many ones as in size.

    def bin_weights_to_p(self,
                         rate):  # Here we will receive each rate/bin weight array and from a specific transformation
        # and convert it to a probability and then set it to 0 if it is not above the threshold self.t
        probability = rate + (1 - self.ro)
        probability = probability / probability.max()
        probability[probability < self.t] = 0

        return probability

    def policy(self,
               probe):  # This function will define the policy according to which the weights of the bins will be
        # updated
        transformation_list = list(self.rates.keys())
        new_bins = list()
        if probe:
            for _ in range(self.depth):
                transformation = random.choice(transformation_list)
                bins = self.rates[
                    transformation]  # Transformation parameters bins (this will return as many arrays as
                # parameters). Each array will contain 17 elements (bins). It's a way of discretizing the parameters
                # ranges
                sample = np.random.uniform(0, 1, len(bins))
                new_bins.append((transformation, sample.tolist()))  # Returning a tuple with the transformation name
                # and the new sample

            return new_bins

        for _ in range(self.depth):
            new_bin_elements = list()
            transformation = random.choice(transformation_list)
            bins = self.rates[transformation]
            sample = np.random.uniform(0, 1, len(bins))
            new_bins.append((transformation, sample.tolist()))
            for rand_number, bin_ in zip(sample,
                                         bins):  # bin_ corresponds to each bin array per transformation parameter
                p = self.bin_weights_to_p(bin_)
                selected_bin = np.random.choice(p.shape[0], p=(p / p.sum()))
                new_bin_elements.append(
                    (selected_bin + rand_number) / p.shape[0])  # Choose random number within bin range
            new_bins.append((transformation, new_bin_elements))

        return new_bins

    def update_bin_weights(self, policy_val,
                           w):  # Here w is going to be the formula from Remixmatch paper: w = 1 - 1/2L
        # * sum(abs(p_model - p)). This measures the extent to which the model’s prediction matches the label
        for operation, bin_vals in policy_val:
            for rate, bin_val in zip(self.rates[operation], bin_vals):
                bin_position = max(int(bin_val*len(rate) - 0.001), 0)  # Selecting in which bin position we are in (
                # e.g. if the parameter is divided into 17 bins, we are selecting which of these bins we are
                # working with (0, 1, 2,...16)).
                rate[bin_position] = rate[bin_position] * self.ro + w * (
                        1 - self.ro)  # Updating the weight of the bin we selected in bin_position

    def update_CTA(self, model, labeled_augmented_images, device):
        model.eval()
        with torch.no_grad():
            for batch_idx, (image_batch, label) in enumerate(labeled_augmented_images):
                # Detach the linear prediction
                image_batch = image_batch.to(device)
                logits = model(image_batch.detach())[0]
                # Compute pseudo probabilities
                probs = torch.softmax(logits, dim=1)

                for prob, l, policy in zip(probs, label, self.policy_list):
                    error = prob
                    error[l] -= 1
                    error = torch.abs(error).sum()
                    self.update_bin_weights(policy, 1.0 - 0.5 * error.item())  # TODO Check for L
        self.policy_list = []


def augment(cta, probe=True):
    def myfun(x):
        policy = cta.policy(probe=probe)
        if probe:
            cta.policy_list.append(policy)
        transformed_image = x
        for pol in policy:
            if pol[0] != 'identity':
                method_to_call = getattr(Transformations, pol[0])
                if len(pol[1]) > 1:
                    transformed_image = method_to_call(transformed_image, pol[1])
                else:
                    transformed_image = method_to_call(transformed_image, pol[1][0])

        return transformed_image

    return myfun


def cutout_strong(level=1):
    def myfun_cutout(x):
        transformed_image = Transformations.cutout(x, level)
        return transformed_image

    return myfun_cutout
