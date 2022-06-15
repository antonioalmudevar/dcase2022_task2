import numpy as np
import torch

def mixup_data(x, y=None, alpha=1.0):

    if y is None:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(x.device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            return mixed_x
        else:
            return x

    else:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(x.device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            mixed_y = lam * y + (1 - lam) * y[index, :]
            return mixed_x, mixed_y
        else:
            return x, y