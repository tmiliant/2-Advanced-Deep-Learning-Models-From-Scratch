import torch
import torch.nn as nn
import bayesian_linear.BayesianLinear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BayesianNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256, batch_size=256, 
            likelihood_variance=0.1, NUM_BATCHES=1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.batch_size = batch_size
        self.likelihood_variance = likelihood_variance
        self.NUM_BATCHES = NUM_BATCHES

        self.l1 = BayesianLinear(input_dim, hidden_size)
        self.l2 = BayesianLinear(hidden_size, hidden_size)
        self.l3 = BayesianLinear(hidden_size, output_dim)
    
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, deterministic=False):
        x = x.view(-1, self.input_dim)

        # INPUT => HL1
        x, kl1 = self.l1(x, deterministic=deterministic)
        x = self.activation(x)

        # HL1  => HL2
        x, kl2 = self.l2(x, deterministic=deterministic)
        x = self.activation(x)

        # HL2 => OUTPUT
        x, kl3 = self.l3(x, deterministic=deterministic)

        kl = kl1 + kl2 + kl3

        return x, kl
   
    def sample_elbo(self, inputs, target, samples=5):
        outputs = torch.zeros(samples, target.shape[0], target.shape[1]).to(device)   
        target = target.repeat(samples, 1, 1)

        for i in range(samples):
            outputs[i], kl = self.forward(inputs)  # kl is same for all samples during same grad step
           
        if self.output_transformation == identity:
            outputs = torch.tanh(outputs)
        
        negative_log_likelihood = -torch.sum( torch.mean(torch.distributions.Normal(outputs, 
                                            self.likelihood_variance).log_prob(target).to(device), axis=0) )

        loss = kl / self.NUM_BATCHES + negative_log_likelihood
        return loss
