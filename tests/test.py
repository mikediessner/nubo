import torch
from torch import Tensor


dims = 3
bounds = Tensor([[0.0, ] * dims, [1.0, ] * dims])
optimum = {"inputs": Tensor([[0.114614, 0.555649, 0.852547]]), 
                "ouput": Tensor([[-3.86278]])}

a = Tensor([1.0, 1.2, 3.0, 3.2])
A = Tensor([[3.0, 10.0, 30.0],
                    [0.1, 10.0, 35.0],
                    [3.0, 10.0, 30.0],
                    [0.1, 10.0, 35.0]])
P = 10**-4 * Tensor([[3689.0, 1170.0, 2673.0],
                            [4699.0, 4387.0, 7470.0],
                            [1091.0, 8732.0, 5547.0],
                            [ 381.0, 5743.0, 8828.0]])


x = Tensor([[0.5, 0.5, 0.5]])
# compute output
y = -torch.sum(a * torch.exp(-torch.sum(A * (x - P)**2, dim=-1)), dim=-1)

print (y)