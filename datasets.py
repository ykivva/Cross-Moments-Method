import pandas as pd
import numpy as np

import random, math, os
import torch


class FastFoodDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, method="cross-moments", del_cols=None):
        super().__init__()
        self.method = method
        self.data_pd = pd.read_csv(path)
        if method=="cross-moments":
            data_pre_treat = self.data_pd[self.data_pd["Treatment"]==0].reset_index()
            data_post_treat = self.data_pd[self.data_pd["Treatment"]==1].reset_index()
            assert (data_pre_treat["Group"]==data_post_treat["Group"]).all(), "Discrepancy in data"

            self.Z = torch.from_numpy(data_pre_treat["Empl"].to_numpy())
            self.D = torch.from_numpy(data_pre_treat["Group"].to_numpy(dtype=np.float32))
            self.Y = torch.from_numpy(data_post_treat["Empl"].to_numpy())

            mask = torch.logical_or(self.Z.isnan(), self.D.isnan())
            mask = torch.logical_or(mask,  self.Y.isnan())
            mask = torch.logical_not(mask)
            self.Z = self.Z[mask]
            self.D = self.D[mask]
            self.Y = self.Y[mask]

            self.Z -= torch.mean(self.Z)
            self.D -= torch.mean(self.D)
            self.Y -= torch.mean(self.Y)

            self.n_samples = len(self.Z)

            self.data = torch.cat((torch.unsqueeze(self.Z, 1), torch.unsqueeze(self.D, 1), torch.unsqueeze(self.Y, 1)), dim=1)
        elif method=="regression":
            if del_cols!=None:
                self.data_pd.drop(del_cols, axis=1)
            self.data_pd.dropna()
            self.data_pd["PostTreatment"] = self.data_pd["Group"]*self.data_pd["Treatment"]

            self.labels = self.data_pd["Empl"].to_numpy()
            self.data = self.data_pd.drop(["Empl"], axis=1).to_numpy()
            self.n_samples = len(self.labels)

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
        if self.method=="cross-moments":
            return iter(self.data[iter_start:iter_end])
        elif self.method=="regression":
            return zip(self.data[iter_start:iter_end], self.labels[iter_start:iter_end])

    def save(self, file_name):
        data = self.data.numpy()
        np.savetxt(file_name, data, delimiter=",")


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


if __name__=="__main__":
    seed = 13
    np.random.seed(seed)
    alpha_d = -2 + 4 * np.random.rand()
    alpha_z = -2 + 4 * np.random.rand()
    beta = -5 + 10 * np.random.rand()
    gamma = -5 + 10 * np.random.rand()
    n_samples = 1000

    # dist_conf = ("Gaussian", {"loc": 0, "scale": 1})
    # artific_data = ArtificialDataset(alpha_d, alpha_z, beta, gamma, n_samples, dist_conf, seed)
    # file_name = "artificial_gaussian.csv"
    # artific_data.save(file_name)

    # dist_conf = ("Exponential", {"rate": 1/2})
    # artific_data = ArtificialDataset(alpha_d, alpha_z, beta, gamma, n_samples, dist_conf, seed)
    # file_name = "artificial_exponential.csv"
    # artific_data.save(file_name)

    file = "data/krueger/krueger_fast_food.csv"
    file_path = os.path.join(os.getcwd(), file)
    fastfood_data = FastFoodDataset(file)
    file_name = "fastfood_krueger.csv"
    fastfood_data.save(file_name)
