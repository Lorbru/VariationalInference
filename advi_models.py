from advi import *


# ===============================================
#    Probabilitsic Principal Component Analysis 
#     with Automatic Relevance Determination
# ===============================================

class PPCA_ARD(ADVIModel):

    def __init__(self, x, latent_dim, mc_size=1, threshold=1e-4, lr=0.1, max_iter=1000):
        """
        Initialize the PPCA model with ARD prior

        Args:
            x (torch.Tensor): data
            latent_dim (int): dimension of the latent space
        """

        super().__init__(mc_size=mc_size, threshold=threshold, lr=lr, max_iter=max_iter)

        # attributes
        self.n_points, self.data_dim = x.shape
        self.latent_dim = latent_dim
        self.param_dim = (self.data_dim + self.n_points + 1) * self.latent_dim + 1

        # priors of theta = (w, z, sigma, alpha)
        self.log_p_z = lambda z: torch.sum(dist.Normal(loc=0., scale=1.).log_prob(z))
        self.log_p_w = lambda w: torch.sum(dist.Normal(loc=0., scale=1.).log_prob(w))
        self.log_p_sigma = lambda sigma: dist.LogNormal(loc=0., scale=1.).log_prob(sigma)   
        self.log_p_alpha = lambda alpha: torch.sum(dist.InverseGamma(1, 1).log_prob(alpha))
        self.log_p_w_vert_alpha = lambda w, sigma, alpha: torch.sum(dist.MultivariateNormal(loc=torch.zeros(self.latent_dim), covariance_matrix=sigma * torch.diag(alpha.squeeze())).log_prob(w))
        
        # conditional distribution
        self.log_p_x_vert_theta = lambda x, w, z, sigma: torch.sum(dist.Normal(loc=torch.matmul(z, w.T), scale=sigma).log_prob(x))
        # self.log_p_x_vert_theta = lambda x, w, z, sigma: -0.5 * torch.sum((x - torch.matmul(z, w.T)) ** 2) / (sigma ** 2)

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
            raise ValueError(f"Data dimension mismatch : expected {self.data_dim} dimensions, got {x.shape[1]}")

        self.n_points, self.data_dim = x.shape
        self.param_dim = (self.data_dim + self.n_points + 1) * self.latent_dim + 1

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
        return torch.abs(sigma) + torch.sum(torch.abs(alpha))
    
    def size_params(self):
        return self.param_dim
    