# import 
import numpy as np
import pandas as pd
import sklearn
import torch.distributions as dist
import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

# ************************************************
#
#             ABSTRACT CLASS MODEL
#
# ************************************************

class ADVIModel(ABC):


    """
    Abstract class for ADVI models : user is required to implement the following methods
        * log_p_x_theta : log joint density p(x, theta)
        * Tinv : inverse transformation from unconstrained to constrained space
        * log_det_jac_Tinv : log determinant of the Jacobian of the inverse transformation Tinv
    
    The class provides default implementations for automatic differentiations on these methods
    """
    @abstractmethod
    def log_p_x_theta(self, x, theta):
        """
        Define the log_joint density p(x, theta)

        Args:
            x (torch.Tensor): data
            theta (torch.Tensor): parameters
        """
        return None

    @abstractmethod
    def Tinv(self, zeta):
        """
        Inverse transformation to map from the unconstrained space to the constrained space.

        Args:
            zeta (torch.Tensor): Unconstrained variables.

        Returns:
            torch.Tensor: Transformed variables in the constrained space.
        """
        return None

    @abstractmethod
    def log_det_jac_Tinv(self, zeta):
        """
        Compute the log determinant of the Jacobian of the inverse transformation Tinv

        Args:
            zeta (torch.Tensor): unconstrained variables

        Returns:
            log_det (torch.Tensor): log determinant
        """
        return None
        

    def grad_log_p_x_theta(self, x, theta):
        """
        Compute the gradient of the log_joint density 
        p(x, theta) with respect to theta using torch 
        automatic differentiation

        Args:
            x (torch.Tensor): data
        
        Returns:
            grad (torch.Tensor): gradient
        """
        theta = theta.clone().detach().requires_grad_(True)
        log_p = self.log_p_x_theta(x, theta)
        log_p.backward()
        grad = theta.grad.detach()
        theta.grad = None
        del theta, log_p
        return grad
    
    def grad_Tinv(self, zeta):
        """
        Compute the gradient of the inverse transformation Tinv
        with respect to zeta using torch automatic differentiation

        Args:
            zeta (torch.Tensor): unconstrained variables

        Returns:
            grad (torch.Tensor): gradient
        """
        zeta = zeta.clone().detach().requires_grad_(True)
        Tinv = torch.sum(self.Tinv(zeta)) # scalar sum for backpropagation
        # print("  - Tinv:",Tinv.shape)
        Tinv.backward()
        grad = zeta.grad.detach()
        zeta.grad = None
        zeta.grad = None
        del Tinv, zeta
        return grad
    
    def grad_log_det_jac_Tinv(self, zeta):
        """
        Compute the gradient of the log determinant of the Jacobian of the inverse transformation Tinv
        with respect to zeta using torch automatic differentiation

        Args:
            zeta (torch.Tensor): unconstrained variables

        Returns:
            grad (torch.Tensor): gradient
        """
        zeta = zeta.clone().detach().requires_grad_(True)
        log_det = torch.sum(self.log_det_jac_Tinv(zeta)) # scalar sum for backpropagation
        log_det.backward()
        grad = zeta.grad.detach()
        zeta.grad = None
        del log_det, zeta
        return grad

# ************************************************
#
#      MODELS AND AUTOMATIC DIFFERENTIATION
#
# ************************************************

class PPCA_ARD(ADVIModel):

    def __init__(self, x, latent_dim):
        """
        Initialize the PPCA model with ARD prior

        Args:
            x (torch.Tensor): data
            latent_dim (int): dimension of the latent space
        """

        # attributes
        self.n_points, self.data_dim = x.shape
        self.latent_dim = latent_dim
        self.param_dim = (self.data_dim + self.n_points + 1) * self.latent_dim + 1

        # priors of theta = (w, z, sigma, alpha)
        self.log_p_z = lambda z: - .5 * torch.sum(z**2)
        self.log_p_w = lambda w: - .5 * torch.sum(w**2)
        self.log_p_sigma = lambda sigma: -.5 * (torch.log(sigma) ** 2) - torch.log(sigma)
        self.log_p_alpha = lambda alpha: torch.sum(dist.InverseGamma(1, 1).log_prob(alpha))
        self.log_p_w_vert_alpha = lambda w, sigma, alpha: torch.sum(dist.MultivariateNormal(loc=torch.zeros(self.latent_dim), covariance_matrix=(sigma ** 2) * torch.diag(alpha.squeeze())).log_prob(w))
        
        # conditional distribution
        self.log_p_x_vert_theta = lambda x, w, z, sigma: -0.5 * torch.sum((x - torch.matmul(z, w.T)) ** 2) / (sigma ** 2)

    def extract_params(self, theta):
        """
        Extract parameters (w, z, sigma, alpha) from theta

        Args:
            theta (torch.Tensor): parameters

        Returns:
            w, z, sigma, alpha (torch.Tensor): extracted parameters
        """
        w_size = self.data_dim * self.latent_dim
        z_size = self.n_points * self.latent_dim
        w = theta[:w_size].view(self.data_dim, self.latent_dim)
        z = theta[w_size:w_size + z_size].view(self.n_points, self.latent_dim)
        sigma = theta[w_size+z_size]
        alpha = theta[w_size+z_size+1:]
        return w, z, sigma, alpha

    def log_p_x_theta(self, x, theta):
        """
        Compute log p(x, theta) = log p(x|theta) + log p(theta) distribution 
        
        Args:
            x (torch.Tensor): data
            theta (torch.Tensor): parameters (w, z, sigma, alpha)
        
        Returns:
            log_p (torch.Tensor): log probability
        """
        if x.shape[0] != self.n_points:
            raise ValueError(f"Data dimension mismatch : expected {self.n_points} points, got {x.shape[0]}")

        if x.shape[1] != self.data_dim:
            raise ValueError(f"Data dimension mismatch : expected {self.n_points} dimensions, got {x.shape[0]}")

        self.n_points, self.data_dim = x.shape
        self.param_dim = (self.data_dim + self.n_points + 1) * self.latent_dim + self.n_points * self.latent_dim + 1 + self.latent_dim

        # extracting parameters
        w, z, sigma, alpha = self.extract_params(theta)
        log_p = torch.zeros(1)

        # conditional distribution        
        log_p += self.log_p_x_vert_theta(x, w, z, sigma)

        # priors
        log_p += self.log_p_z(z)
        log_p += self.log_p_w(w)
        log_p += self.log_p_sigma(sigma)
        log_p += self.log_p_alpha(alpha)
        log_p += self.log_p_w_vert_alpha(w, sigma, alpha)

        return log_p

    def Tinv(self, zeta):
        """
        Inverse transformation to map from the unconstrained space to the constrained space.

        Args:
            zeta (torch.Tensor): Unconstrained variables.

        Returns:
            torch.Tensor: Transformed variables in the constrained space.
        """
        w_size = self.data_dim * self.latent_dim
        z_size = self.n_points * self.latent_dim
        sigma = zeta[w_size+z_size]
        alpha = zeta[w_size+z_size+1:]
        return torch.cat([zeta[:w_size+z_size],  torch.exp(sigma).unsqueeze(1), torch.exp(alpha)], dim=0)
    
    def log_det_jac_Tinv(self, zeta):
        """
        Compute the log determinant of the Jacobian of the inverse transformation Tinv

        Args:
            zeta (torch.Tensor): unconstrained variables

        Returns:
            log_det (torch.Tensor): log determinant
        """
        w_size = self.data_dim * self.latent_dim
        z_size = self.n_points * self.latent_dim
        sigma = zeta[w_size+z_size]
        alpha = zeta[w_size+z_size+1:]
        return torch.sum(torch.abs(sigma)) + torch.sum(torch.abs(alpha))


# ************************************************
#
#                 ADVI ALGORITHM
#
# ************************************************

class ADVI(sklearn.base.BaseEstimator):

    def __init__(self, advi_model, mc_size=1, threshold=1e-4, lr=0.1, max_iter=1000):
        
        self.model = advi_model
        self.param_dim = advi_model.param_dim
        self.mu = torch.zeros(self.param_dim)
        self.omega = torch.zeros(self.param_dim)

        self.max_iter = max_iter
        self.mc_size = mc_size
        self.threshold = threshold
        self.lr = lr

        self.__epsstep = 1e-16
        self.__taustep = 1
        self.__alphastep = 0.1

    def Sinv(self, eta, mu, omega):
        return (torch.exp(omega) * eta) + mu

    def fit(self, X):

        X = torch.tensor(X)
        mu = torch.zeros(self.param_dim, 1)
        omega = torch.zeros(self.param_dim, 1)
        s = torch.zeros(2*self.param_dim, 1)

        # Optimization loop
        for  i  in tqdm(range(1, self.max_iter+1)):
            mc_sampling = torch.randn(self.param_dim, self.mc_size)
            zeta = self.Sinv(mc_sampling, mu, omega)
            grad_mu, grad_omega = self.gradient(X, mu, omega, zeta)
            rho, s = self.adaptive_step(torch.concat([grad_mu, grad_omega]), i, s)
            mu = mu + (rho[:self.param_dim,:] * grad_mu).detach()
            omega = omega + (rho[self.param_dim:,:] * grad_omega).detach()

        self.mu = mu
        self.omega = omega

    def gradient(self, x, mu, omega, zeta):
        grad_mu = torch.zeros(self.param_dim, 1)
        grad_omega = torch.zeros(self.param_dim, 1)
        for s in range(self.mc_size) :
            sample = zeta[:, s].unsqueeze(1)
            theta = self.model.Tinv(sample)
            grad_log_joint = self.model.grad_log_p_x_theta(x, theta)
            grad_Tinv = self.model.grad_Tinv(sample)
            grad_log_jac_Tinv = self.model.grad_log_det_jac_Tinv(sample)
            expect =  (grad_log_joint * grad_Tinv + grad_log_jac_Tinv).detach()
            grad_mu += expect
            grad_omega += expect * (torch.exp(omega) * sample).detach()
        grad_mu /= self.mc_size
        grad_omega /= self.mc_size
        grad_omega += 1
        return grad_mu, grad_omega
    
    def adaptive_step(self, grad_theta, i, s):
        rho = torch.zeros(2*self.param_dim)
        s = (1 - self.__alphastep) * s + self.__alphastep * (grad_theta**2)
        rho = self.lr * (i ** (-.5+self.__epsstep)) / (self.__taustep + torch.sqrt(s))
        return rho, s