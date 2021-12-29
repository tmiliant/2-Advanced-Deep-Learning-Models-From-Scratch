import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Gaussian:
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu.to(device)
        self.rho = rho.to(device)  # Define the parameter with which the standard deviation is reparameterized.
  
    def draw_sample(self):
        epsilon = torch.distributions.Normal(0, 1).sample(self.rho.size()).to(device)
        return torch.log1p(torch.exp(self.rho)) * epsilon + self.mu

