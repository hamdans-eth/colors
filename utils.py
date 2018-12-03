import torch
from torch.nn import functional
from torch.autograd import Variable
from scipy.optimize import curve_fit
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomTrace(torch.autograd.Function):

    def forward(self, input):
        self.isize = input.size()
        return input.new([torch.trace(input)])

    def backward(self, grad_output):
        isize = self.isize
        grad_input = grad_output.new(isize).copy_(torch.eye(*isize))
        grad_input.mul_(grad_output[0])
        return grad_input

def cholesky(A):
    L = torch.zeros_like(A)

    for i in range(A.shape[-1]):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s = s + L[i,k].clone() * L[j,k].clone()

            L[i,j] = torch.sqrt(A[i,i] - s) if (i == j) else \
                      (1.0 / L[j,j].clone() * (A[i,j] - s))
    return L





def get_priors(RGB):
    #return clusters with mu and sigma for each RGB
    means = {}
    variances = {}
    for key in RGB:
        R = [v[0] for v in RGB[key]]
        G = [v[1] for v in RGB[key]]
        B = [v[2] for v in RGB[key]]
        means[key] = np.array([np.mean(R),np.mean(G), np.mean(B)])
        variances[key] = np.array([np.var(R), np.var(G), np.var(B)])
        #variances[key] = np.cov( np.array([R, G, B]))

    return means,variances
