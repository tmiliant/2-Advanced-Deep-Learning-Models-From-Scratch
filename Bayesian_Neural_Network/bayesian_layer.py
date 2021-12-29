import torch
import torch.nn.functional as F
import torch.nn as nn
import gaussian.Gaussian

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0, prior_sigma=1):
        ''' The paper uses a gaussian mixture as a prior each weight and bias.
        This implementation uses as prior a single gaussian.        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define weight parameterized by mean and rho, according to paper about Bayes by Backprop.
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0., 0.5))  
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-5.0, 0.5))  
        # Define the distribution of the generic weight using the above paramaters.
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Define bias parameterized using the same idea seen in the case of the weight.
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0., 0.5))  
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-5.0, 0.5)) 
        # Define the distribution of the generic bias using the above paramters.
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        
        # Define the parameters of the prior distribution of each individual weight and bias.
        self.prior_mu = prior_mu  
        self.prior_sigma = prior_sigma  

    def kl_div(self, mu_q, rho_q, mu_p, sigma_p):
        sigma_q = torch.log1p(torch.exp(rho_q))  # Turn rho into sigma for variational posterior

        mu_p = torch.ones_like(mu_q) * mu_p  # The tensor of size mu_q's size with mu_p in every cell  
        sigma_p = torch.ones_like(sigma_q) * sigma_p  # The tensor of size sigma_q's size with sigma_p in every cell

        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *(sigma_p**2)) - 0.5

        return kl.mean()


    def forward(self, inputs, deterministic=False):
        if not deterministic:
            weight = self.weight.draw_sample()
            bias = self.bias.draw_sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        kl = self.kl_div(self.weight_mu, self.weight_rho, 
                              self.prior_mu, self.prior_sigma)  # Prior mu and sigma don't change
        kl += self.kl_div(self.bias_mu, self.bias_rho, 
                              self.prior_mu, self.prior_sigma)  # Prior mu and sigma don't change

        return F.linear(inputs, weight, bias), kl
