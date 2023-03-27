import torch
from torch import Tensor
from torch.nn.functional import pdist
from typing import Optional


class LatinHypercubeSampling:
    r"""
    Latin hypercube sampling.

    Generates a space-filling design. Two options are possible: sampling from a
    random or a maximin Latin hypercube. To sample :math:`n` points, the random
    Latin hypercube divides each dimension into :math:`n` equal parts and
    places :math:`n` points such that for every dimension each equal part
    contains exactly one point. The maximin Latin hypercube takes a simple
    approach and draws a large number of random Latin hypercube samples and
    returns the one with the largest minimal distance between points.

    Attributes
    ----------
    dims : ``int``
        Number of dimensions
    """

    def __init__(self, dims: int) -> None:
        """
        Parameters
        ----------
        dims : ``int``
            Number of dimensions.
        """

        self.dims = dims

    def random(self, points: int) -> Tensor:
        r"""
        Draw a random Latin hypercube sample.

        To sample :math:`n` points, the random Latin hypercube divides each
        dimension into :math:`n` equal parts and places :math:`n` points such
        that for every dimension each equal part contains exactly one point.
        
        Parameters
        ----------
        points : ``int``
            Number of points.

        Returns
        -------
        ``torch.Tensor``
            (size `points` x `dims`) Random Latin hypercube sample.
        """

        hypercube = torch.empty((points, self.dims), dtype=torch.float64)

        # pick random permutation of (1, ..., points) for each dimension
        for dim in range(self.dims):
            hypercube[:, dim] = torch.randperm(points)

        # translate each dimension to [0, 1] range
        increment = 1/points
        hypercube = hypercube * increment

        # pick random value within increment
        jitter = torch.rand(size=hypercube.size(), dtype=torch.float64) * increment
        hypercube = hypercube + jitter

        return hypercube

    def maximin(self,
                points: int,
                samples: Optional[int]=1000) -> Tensor:
        r"""
        Draw a maximin Latin hypercube sample. 
        
        Draws a large number of random Latin hypercube samples and selects the
        one with the largest minimal distance between points.

        Parameters
        ----------
        points : ``int``
            Number of points.
        samples : ``int``
            Number of random Latin hypercube samples.

        Returns
        -------
        ``torch.Tensor``
            (size `points` x `dims`) Maximin Latin hypercube sample.
        """

        hypercubes = torch.empty((samples, points, self.dims), dtype=torch.float64)
        min_dist = torch.empty(samples, dtype=torch.float64)

        for sample in range(samples):

            # sample random Latin hypercubes
            hypercubes[sample, :, :] = self.random(points)

            # compute minimal distance of all samples
            min_dist[sample] = torch.min(pdist(hypercubes[sample, :, :]))

        # pick hypercube with maximal minimal distance
        maximin_index = torch.argmax(min_dist)
        maximin_hypercube = hypercubes[maximin_index]

        return maximin_hypercube
