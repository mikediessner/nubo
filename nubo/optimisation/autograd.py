import torch
from torch import Tensor
from typing import Callable, Tuple


def get_fun_and_jac(x: Tensor, func: Callable) -> Tuple[Tensor, Tensor]:

    x = Tensor(x).double().requires_grad_()
    x = torch.reshape(x, (1, -1))
    loss = func(x)

    x_grad = x.values() if isinstance(x, dict) else x

    grads = torch.autograd.grad(loss, x_grad)[0]

    return loss.cpu().detach().numpy(), grads.cpu().detach().numpy()
