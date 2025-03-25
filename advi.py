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
#           ABSTRACT CLASS ADVI MODEL
#
# ************************************************

class ADVIModel(ABC):

    """
    Abstract class for ADVI models : user is required to implement the following methods for any model
    which inherits from this class :
        * log_p_x_theta : log joint density p(x, theta)
        * Tinv : inverse transformation from unconstrained to constrained space
        * log_det_jac_Tinv : log determinant of the Jacobian of the inverse transformation Tinv
        * extract_params : extracts the parameters from a full latent parameters tensor theta
        * param_dim() : compute the total dimension of the latent parameters
        
    Then the model which is defined can be optimized with Automatic Differentiation Variational Inference
    using the methods already defined in this module
    """
    def __init__(self, mc_size=1, threshold=1e-4, lr=0.1, max_iter=1000):
        
        self.mu = None                                  # mu parameter 
        self.omega = None                               # omega parameter
        self.elbo = []                                  # elbo history after convergence

        self.max_iter = max_iter                        # maximum iterations for the optimization
        self.mc_size = mc_size                          # monte carlo sample size for gradient estimation
        self.threshold = threshold                      # elbo criterion to break optimization loop
        self.lr = lr                                    # learning rate

        self.__epsstep = 1e-16                          # epsilon for adaptive learning rate
        self.__taustep = 1                              # tau for adaptive learning rate
        self.__alphastep = 0.1                          # alpha for adaptive learning rate


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
        Tinv = torch.sum(self.Tinv(zeta)) 
        Tinv.backward()
        grad = zeta.grad.detach()
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
        log_det = torch.sum(self.log_det_jac_Tinv(zeta))
        log_det.backward()
        grad = zeta.grad.detach()
        return grad

    def Sinv(self, eta, mu, omega):
        """
        Inverse transformation of S

        Args:
            eta (torch.Tensor) : monte carlo sample
            mu (torch.Tensor) : mu parameter
            omega (torch.Tensor) : omega parameter

        Returns:
            (torch.Tensor) : S^(-1)(eta)
        """
        return (torch.exp(omega) * eta) + mu             

    def fit(self, X):

        """
        Fit the model to X

        Params:
            X (ndarray or torch.Tensor) : data for the model training
        """
        X = torch.tensor(X)
        mu = torch.zeros(self.size_params(), 1)
        omega = torch.zeros(self.size_params(), 1)
        s = torch.zeros(2*self.size_params(), 1)
        elbo_list = [-torch.inf]

        # Optimization loop
        for  i  in tqdm(range(1, self.max_iter+1)):

            # monte carlo sample 
            mc_sampling = torch.randn(self.size_params(), self.mc_size)
            
            # Sinv transformation
            zeta = self.Sinv(mc_sampling, mu, omega)
            
            # gradients and elbo estimation
            grad_mu, grad_omega, elbo = self.gradient(X, mu, omega, zeta)

            # new learning rate
            rho, s = self.adaptive_step(torch.concat([grad_mu, grad_omega]), i, s)
            
            # update parameters with gradient ascent
            mu = mu + (rho[:self.size_params(),:] * grad_mu).detach()
            omega = omega + (rho[self.size_params():,:] * grad_omega).detach()

            if torch.abs(elbo_list[-1] - elbo) < self.threshold:
                break

            elbo_list.append(elbo.item())

        self.mu = mu
        self.omega = omega
        self.elbo = elbo_list[1:]

    def phi_estimates(self):
        """
        Get the estimates of phi after optimization

        Returns:
            mu (torch.Tensor) : mu estimation
            omega (torch.Tensor) : omega estimation
        """
        return self.mu, self.omega
    
    def theta_estimates(self, n_estimators=500, random_state=42):
        """
        Draw an estimate of theta after optimization

        Returns:
            tuple(torch.Tensor) : estimated parameters of the model
        """
        torch.manual_seed(random_state)
        eta = torch.randn(self.size_params(), n_estimators)
        zeta = self.Sinv(eta, self.mu, self.omega)
        theta = torch.zeros(self.size_params(), 1)
        for i in range(n_estimators):
            theta += self.Tinv(zeta[:,i].unsqueeze(1))
        theta /= n_estimators
        return self.extract_params(theta)

    def elbo_history(self):
        """
        Get the elbo evolution and convergence for the last optimization 

        Returns:
            elbos (list) : elbo convergence
        """
        return self.elbo

    def gradient(self, x, mu, omega, zeta):
        """
        Get the gradient for mu and omega with data x and zeta

        Returns:
            grad_mu (torch.Tensor) : grad mu estimation
            grad_omega (torch.Tensor) : grad omega estimation
        """
        # gradient and elbo initialization
        grad_mu = torch.zeros(self.size_params(), 1)
        grad_omega = torch.zeros(self.size_params(), 1)
        elbo = torch.zeros(1)

        for s in range(self.mc_size) :
            
            # mc sample for the current iteration
            sample = zeta[:, s].unsqueeze(1)

            # theta extraction
            theta = self.Tinv(sample)
            
            # gradients inside expectation
            grad_log_joint = self.grad_log_p_x_theta(x, theta)
            grad_Tinv = self.grad_Tinv(sample)
            grad_log_jac_Tinv = self.grad_log_det_jac_Tinv(sample)

            # adding for the estimation of the expectation with monte carlo
            expect =  (grad_log_joint * grad_Tinv + grad_log_jac_Tinv).detach()
            grad_mu += expect
            grad_omega += expect * (torch.exp(omega) * sample).detach()
            elbo += (self.log_p_x_theta(x, theta) + self.log_det_jac_Tinv(sample))

        # final gradients and elbo estimates
        grad_mu = grad_mu/self.mc_size
        grad_omega = grad_omega/self.mc_size + 1
        elbo = elbo/self.mc_size + (self.size_params()/2) * (torch.log(torch.Tensor([2*torch.pi])) + 1) + 2*torch.sum(omega)
        
        return grad_mu, grad_omega, elbo
    
    def adaptive_step(self, grad_theta, i, s):
        """
        Adaptive learning rate for the variational inference algorithm

        Args:
            grad_theta (torch.Tensor) : gradient of theta
            i (int) : iteration
            s (torch.Tensor) : previous value of s for the new learning rate

        Returns:
            rho (torch.Tensor) : new learning rate
            s (torch.Tensor) : value of s for the next iteration
        """
        rho = torch.zeros(2*self.size_params())
        s = (1 - self.__alphastep) * s + self.__alphastep * (grad_theta**2)
        rho = self.lr * (i ** (-.5+self.__epsstep)) / (self.__taustep + torch.sqrt(s))
        return rho, s
    
    def save_pth(self, filepath):
        params = {
            "mu":self.mu.squeeze().detach().tolist(),
            "omega":self.mu.squeeze().detach().tolist(),
            "elbo_history":self.elbo,
        }
        torch.save(params, filepath)

    def load_pth(self, filepath):
        params = torch.load(filepath)
        self.mu = torch.Tensor(params['mu']).unsqueeze(1)
        self.omega = torch.Tensor(params['omega']).unsqueeze(1)
        self.elbo = params['elbo_history']


    # -----------------------------------------------------------
    #                    ABSTRACT METHODS  
    # -----------------------------------------------------------

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
            Tinv (torch.Tensor) : Transformed variables in the constrained space.
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
    
    @abstractmethod
    def extract_params(self, theta):
        """
        Extract the different parameters from a single torch.Tensor theta

        Args:
            theta (torch.Tensor) : full latent parameters

        Returns:
            args tuple(torch.Tensor) : separated latent parameters
        """
        return None

    @abstractmethod    
    def size_params(self):
        """
        Get the total dimension of the latent parameters.
        This method returns the total dimension of the latent parameters, which is required for
        initializing the mu and omega parameters in the ADVI optimization process.

        Returns:
            (int) : The total dimension of the latent parameters.
        """
        return None