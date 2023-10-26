import numpy as np
import math

import torch
from sklearn.linear_model import LinearRegression


def compute_ratio(Z, D, deg=2):
    """
    deg - is equal to the (n-1) from the paper
    """
    var_u = torch.mean(Z*D)
    sign = torch.sign(var_u)
    
    diff_normal_D = torch.mean(D**(deg)*Z) - deg*var_u*torch.mean(D**(deg-1))
    diff_normal_Z = torch.mean(Z**(deg)*D) - deg*var_u*torch.mean(Z**(deg-1))
    
    alpha_sq = ((diff_normal_D) / (diff_normal_Z))
    if alpha_sq < 0:
        alpha_sq = -(abs(alpha_sq)**(1/(deg-1)))
    else:
        alpha_sq = alpha_sq**(1/(deg-1))
    alpha_sq = abs(alpha_sq)*sign
    
    return alpha_sq


def compute_beta(Z, D, Y, deg=2):
    denominator = 0
    while denominator==0:
        alpha_sq = compute_ratio(Z, D, deg)
        numerator = torch.mean(D*Y) - alpha_sq*torch.mean(Y*Z)
        denominator = torch.mean(D*D) - alpha_sq*torch.mean(D*Z)
        deg += 1
    return numerator / denominator


def get_beta_sensor_fusion(W, Z, D, Y, deg=2, n_iter=100, percentage=0.9):
    length = len(W)
    sample_set = np.arange(length)
    n_samples = math.ceil(percentage*length)
    betas_est1 = np.zeros(n_iter)
    betas_est2 = np.zeros(n_iter)
    for it in range(n_iter):
        args_id = np.random.choice(sample_set, size=n_samples)
        W_tmp = W[args_id]
        Z_tmp = Z[args_id]
        D_tmp = D[args_id]
        Y_tmp = Y[args_id]
        
        beta_est1 = compute_beta(Z_tmp, D_tmp, Y_tmp, deg=deg)
        beta_est2 = compute_beta(W_tmp, D_tmp, Y_tmp, deg=deg)
        
        betas_est1[it] = beta_est1
        betas_est2[it] = beta_est2
    
    beta_est1 = np.mean(betas_est1)
    beta_est2 = np.mean(betas_est2)
    std1 = np.std(betas_est1)
    if std1==0: 
        return beta_est1    
    std2 = np.std(betas_est2)
    if std2==0:
        print(beta_est2)
        return beta_est2
    coef1 = 1./std1**2
    coef2 = 1./std2**2
    beta = (coef1*beta_est1 + coef2*beta_est2) / (coef1 + coef2)
    return beta   


def compute_beta_2proxies(W, Z, D, Y):
    covDY = torch.mean(D*Y)
    covDZ = torch.mean(D*Z)
    covDW = torch.mean(D*W) 
    covWZ = torch.mean(W*Z)
    covWY = torch.mean(W*Y)
    varD = torch.mean(D*D)
    numerator = covDY*covWZ - covDZ*covWY
    denominator = varD*covWZ - covDW*covDZ
    return numerator / denominator


def tyc20(Z, W, D, Y):
    data_tmp = np.hstack((Z[:, None], D[:, None]))
    reg1 = LinearRegression().fit(data_tmp, W)
    W_new = reg1.predict(data_tmp)

    data_tmp = np.hstack((D[:, None], W_new[:, None]))
    reg2 = LinearRegression().fit(data_tmp, Y)
    return reg2.coef_[0]