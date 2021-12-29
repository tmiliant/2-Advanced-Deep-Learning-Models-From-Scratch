import torch
import torch.nn as nn
import torch.nn.functional as F

class MC_Dropout_Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256, mask_prob=0.2):
        super(MC_Dropout_Network, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_prob = mask_prob
 
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_dim)
        
        self.activation = nn.ReLU(inplace = True)

    def forward(self, x):
        
        x = x.view(-1, self.input_dim)
       
        x = self.layer1(x)
        x = self.activation(x) 
        x = F.dropout(x, p=self.mask_prob, training=True)
       
        x = self.layer2(x)
        x = self.activation(x) 
        x = F.dropout(x, p=self.mask_prob, training=True)
        
        x = self.layer3(x)

        return x

    def gaussian_predictive_dist(self, obs, no_passes):
        predictions = self.forward(obs).unsqueeze(0)
        for i in range(1, no_passes):   
            predictions = torch.cat((predictions, self.forward(obs).unsqueeze(0), 0))
        predictions = np.array(predictions)
        mean = torch.mean(predictions, 0)
        std = torch.std(predictions, 0)
        return mean, std
