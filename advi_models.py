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

class GammaPoissonNMF(ADVIModel):
    def __init__(self, X, latent_dim=10,mc_size=1,threshold=1e-4, lr=0.1, max_iter=1000):
        """
        Initialize the Gamma-Poisson Non-Negative Matrix Factorization (NMF) model.

        Args:
            X (torch.Tensor): Data matrix (observations × features).
            latent_dim (int): Dimension of the latent space.
            mc_size (int): Monte Carlo sample size.
            threshold (float): Convergence threshold for optimization.
            lr (float): Learning rate for optimization.
            max_iter (int): Maximum number of iterations.
        """

        super().__init__(mc_size=mc_size, threshold=threshold, lr=lr, max_iter=max_iter)
        self.X = X
        self.latent_dim = latent_dim
        self.a_w, self.b_w = 0.1, 0.1  #  prior Gamma W
        self.a_h, self.b_h = 0.1, 0.1  #  prior Gamma H
        self.theta = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Random initialization of W and H using Gamma distributions.

        Returns:
            theta (torch.nn.Parameter): Flattened tensor containing the parameters W and H.
        """

        n_features = self.X.shape[1]
        n_samples = self.X.shape[0]
        k = self.latent_dim

        # Initialisation avec des distributions Gamma (conforme à l'article)
        W_init = torch.distributions.Gamma(self.a_w, self.b_w).sample((n_features, k))
        H_init = torch.distributions.Gamma(self.a_h, self.b_h).sample((k, n_samples))

        # Application de la contrainte de monotonie sur H
        H_init = torch.cumsum(H_init.abs(), dim=1)

        # Conversion en paramètre optimisable
        theta = torch.cat([W_init.flatten(), H_init.flatten()])
        return nn.Parameter(theta)

    def log_p_x_theta(self, x, theta):
        """
        Compute the log-likelihood log p(x|theta) and priors.

        Args:
            x (torch.Tensor): Data matrix.
            theta (torch.Tensor): Model parameters (W and H).

        Returns:
            log_p (torch.Tensor): Log probability value.
        """

        # 1. Extraire W et H à partir de theta
        W, H = self.extract_params(theta)
       # print("W shape:", W.shape, "| H shape:", H.shape)
        # 2. Contrainte d'ordre sur H (monotonicité)
        H = torch.cumsum(H.abs(), dim=1)
        
        # 3. Likelihood Poisson (V ~ Poisson(WH))
        lam = torch.mm(H.T, W.T)  # [400,10] @ [10,4096] → [400,4096]
        
            
        log_lik = torch.sum(dist.Poisson(lam).log_prob(x))
        
        # 4. Priors Gamma sur W et H
        log_prior_W = torch.sum(dist.Gamma(self.a_w, self.b_w).log_prob(W))
        log_prior_H = torch.sum(dist.Gamma(self.a_h, self.b_h).log_prob(H))
        
        return log_lik + log_prior_W + log_prior_H
    
    def Tinv(self, zeta):
          
        return torch.exp(zeta)  
    
    def log_det_jac_Tinv(self, zeta):
        return torch.sum(zeta)
    
    def extract_params(self, theta):
        """
        Extract parameters W and H from the flattened parameter vector theta.

        Args:
            theta (torch.Tensor): Model parameters.

        Returns:
            W, H (torch.Tensor): Extracted matrices.
        """
        theta = theta.flatten()  
        
        n_features = self.X.shape[1]  # 4096
        k = self.latent_dim           # 10
        n_samples = self.X.shape[0]   # 400
        
        total_params_expected = n_features * k + k * n_samples  # 4096*10 + 10*400 = 44960
        if len(theta) != total_params_expected:
            raise ValueError(
                f"Taille incorrecte de theta : reçu {len(theta)}, "
                f"attendu {total_params_expected} "
                f"(W:{n_features*k} + H:{k*n_samples})"
            )
        
        W_size = n_features * k
        W = theta[:W_size].reshape(n_features, k)
        
        H = theta[W_size:W_size + k * n_samples].reshape(k, n_samples)
        
        return W,H 
    def size_params(self):
        n_samples, n_features = self.X.shape
        return (n_features + n_samples) * self.latent_dim
    
