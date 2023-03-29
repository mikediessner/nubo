import torch

a = torch.tensor(1)
print(a)
print(type(a))
print(a.size())
import numpy as np
a = 1.0
print(a.shape)
b = a.reshape(1)
print(b)
print(type(b))
print(b.size())
print(b.reshape([]).size())

# x_test = torch.tensor([[1., 2., 3.],
#                        [4., 5., 6.],
#                        [7., 8., 9.]])

# x_pending = torch.tensor([[0., 0., 0.],
#                           [1., 1., 1.]])

# x_test = x_test.unsqueeze(1)
# x_pending = x_pending.tile((3, 1, 1))
# x_test = torch.cat([x_test, x_pending], dim=1)

# from gpytorch.distributions import MultivariateNormal

# means = torch.tensor([[1., 2., 3.],
#                       [10., 20., 30.]])
# cov = torch.tensor([[[1., 0., 0.],
#                      [0., 1., 0.],
#                      [0., 0., 1.]],
                     
#                     [[1., 0., 0.],
#                      [0., 1., 0.],
#                      [0., 0., 1.]]])
# mvn = MultivariateNormal(means, cov)

# samples = mvn.rsample(torch.Size([10]))
# print(samples)
# print(samples.size())

# ei = samples - torch.tensor([[5., 5., 5.]])
# print(ei)
# ei = ei.max(dim=2).values
# print(ei)
# ei = ei.mean(dim=0)
# print(ei)