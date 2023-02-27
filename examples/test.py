import torch
import gpytorch

samples = 10000000

mean = torch.tensor([5., 20., -10.])
covariance = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
mvn = gpytorch.distributions.MultivariateNormal(mean, covariance)

base_samples = mvn.get_base_samples(torch.Size([samples]))
samples = mvn.rsample(torch.Size([samples]), base_samples=base_samples).double()
print(samples.mean(0))

samples = 10000000

mean = torch.tensor([5.])
covariance = torch.tensor([[1.]])
mvn = gpytorch.distributions.MultivariateNormal(mean, covariance)

base_samples = mvn.get_base_samples(torch.Size([samples]))
samples = mvn.rsample(torch.Size([samples]), base_samples=base_samples).double()
print(samples.mean(0))
