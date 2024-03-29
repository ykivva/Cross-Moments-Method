import pandas as pd
import numpy as np

import random, math, os
import torch

SCALE = 1

def transformation(x, scale=SCALE):
    return 10*torch.tanh(x/scale)


class ArtificialDataset(torch.utils.data.IterableDataset):
    distributions = {
        "Exponential": torch.distributions.exponential.Exponential,
        "Gaussian": torch.distributions.normal.Normal,
        "Uniform": torch.distributions.uniform.Uniform,
    }

    def __init__(self, alpha_z, alpha_d, beta, gamma, 
                 n_samples=1000, dist_conf=None, seed=None):
        super().__init__()
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.alpha_z = alpha_z
        self.alpha_d = alpha_d
        self.beta = beta
        self.gamma = gamma

        self.n_samples = n_samples
        dist_name, dist_param = dist_conf
        self.dist_name = dist_name if dist_name!=None else "Gaussian"

        self.dist_param_u = dist_param["U"] if dist_param["U"]!=None else {"loc":0, "scale":1}
        self.dist_u = self.distributions[dist_name](**self.dist_param_u)
        self.eps_u = self.dist_u.sample([n_samples])

        self.dist_param_z = dist_param["Z"] if dist_param["Z"]!=None else {"loc":0, "scale":1}
        self.dist_z = self.distributions[dist_name](**self.dist_param_z)
        self.eps_z = self.dist_z.sample([n_samples])

        self.dist_param_d = dist_param["D"] if dist_param["D"]!=None else {"loc":0, "scale":1}
        self.dist_d = self.distributions[dist_name](**self.dist_param_d)
        self.eps_d = self.dist_d.sample([n_samples])
        
        self.dist_param_y = dist_param["Y"] if dist_param["Y"]!=None else {"loc":0, "scale":1}
        self.dist_y = self.distributions[dist_name](**self.dist_param_y)
        self.eps_y = self.dist_y.sample([n_samples])

        self.eps_u -= torch.mean(self.eps_u)
        self.eps_z -= torch.mean(self.eps_z)
        self.eps_d -= torch.mean(self.eps_d)
        self.eps_y -= torch.mean(self.eps_y)

        self.Z = self.eps_u*alpha_z + self.eps_z
        self.D = self.eps_u*alpha_d + self.eps_d
        self.Y = self.D*beta + self.eps_u*gamma + self.eps_y

        self.data = torch.cat((torch.unsqueeze(self.Z, 1), torch.unsqueeze(self.D, 1), torch.unsqueeze(self.Y, 1)), dim=1)
        

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.n_samples
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(self.data[iter_start:iter_end])

    def save(self, file_name):
        data = self.data.numpy()
        np.savetxt(file_name, data, delimiter=",")


class ArtificialDataset_NonLin(torch.utils.data.IterableDataset):
    distributions = {
        "Exponential": torch.distributions.exponential.Exponential,
        "Gaussian": torch.distributions.normal.Normal,
        "Uniform": torch.distributions.uniform.Uniform,
    }

    def __init__(self, alpha_z, alpha_d, beta, gamma, 
                 n_samples=1000, dist_conf=None, seed=None, tran_scale=None):
        super().__init__()
        self.seed = seed
        self.tran_scale = tran_scale or SCALE
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.alpha_z = alpha_z
        self.alpha_d = alpha_d
        self.beta = beta
        self.gamma = gamma

        self.n_samples = n_samples
        dist_name, dist_param = dist_conf
        self.dist_name = dist_name if dist_name!=None else "Gaussian"

        self.dist_param_u = dist_param["U"] if dist_param["U"]!=None else {"loc":0, "scale":1}
        self.dist_u = self.distributions[dist_name](**self.dist_param_u)
        self.eps_u = self.dist_u.sample([n_samples])

        self.dist_param_z = dist_param["Z"] if dist_param["Z"]!=None else {"loc":0, "scale":1}
        self.dist_z = self.distributions[dist_name](**self.dist_param_z)
        self.eps_z = self.dist_z.sample([n_samples])

        self.dist_param_d = dist_param["D"] if dist_param["D"]!=None else {"loc":0, "scale":1}
        self.dist_d = self.distributions[dist_name](**self.dist_param_d)
        self.eps_d = self.dist_d.sample([n_samples])
        
        self.dist_param_y = dist_param["Y"] if dist_param["Y"]!=None else {"loc":0, "scale":1}
        self.dist_y = self.distributions[dist_name](**self.dist_param_y)
        self.eps_y = self.dist_y.sample([n_samples])

        self.eps_u -= torch.mean(self.eps_u)
        self.eps_z -= torch.mean(self.eps_z)
        self.eps_d -= torch.mean(self.eps_d)
        self.eps_y -= torch.mean(self.eps_y)

        self.Z = transformation(self.eps_u*alpha_z, scale=self.tran_scale) + self.eps_z
        self.D = transformation(self.eps_u*alpha_d, scale=self.tran_scale) + self.eps_d
        self.Y = self.D*beta + transformation(self.eps_u*gamma, scale=self.tran_scale) + self.eps_y

        self.data = torch.cat((torch.unsqueeze(self.Z, 1), torch.unsqueeze(self.D, 1), torch.unsqueeze(self.Y, 1)), dim=1)
        

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.n_samples
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(self.data[iter_start:iter_end])

    def save(self, file_name):
        data = self.data.numpy()
        np.savetxt(file_name, data, delimiter=",")