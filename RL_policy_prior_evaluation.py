from garage.torch.distributions import TanhNormal
from abc import ABC, abstractmethod
from sac import Policy
from beha_MC_Dropout import MC_Dropout_Network
from beha_MFVI import BayesianNetwork 
import os, os.path
import torch
import os
import click
import gym
import numpy as np
import torch
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.optim as optim
import time



# behavioral prior has same architecture as online policy
from sac import Policy 

os.environ['MUJOCO_PY_MJKEY_PATH']='/data/ziz/iliant/mjkey.txt'
os.environ['MUJOCO_PY_MUJOCO_PATH']='/data/ziz/iliant/mujoco200'
import d4rl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_train_test_data(env_name, path_to_dataset, test_size=0.05, random_state=42, default=True):
    e = gym.make(env_name)
    e.reset()
    data = e.get_dataset(path_to_dataset)

    observations = torch_ify(data['observations']).to(device)
    actions = torch_ify(data['actions']).to(device)
        
    if default:  # train on whole dataset to clone prior
        return (observations, actions)

    else:  # keep test data for evaluating to get best model for clone
        train_x, test_x, train_y, test_y = train_test_split(observations,
                        actions, test_size=test_size, random_state=random_state)

        # first train, then test
        return (train_x, train_y, test_x, test_y)

def mse(preds_samples, targets, noise_var=0.5):  # llk variance should be passed into this
    preds_mean = preds_samples.mean(0)
    mse_of_mean = torch.mean(
        torch.mean((targets - preds_mean)**2, 0), 0
    )
    return mse_of_mean

def expected_mse(preds_samples, targets, noise_var=0.5):  # llk variance should be passed into this
    targets = targets.repeat(preds_samples.shape[0], 1, 1)
    #targets = torch.tile(targets, (preds_samples.shape[0], 1, 1))  # repeat targets on dimension of preds_samples
    exp_mse = torch.mean(
        torch.mean(torch.mean((targets - preds_samples)**2, 0), 0), 0
    )
    return exp_mse


class IPolicyPrior(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_log_probs(self, state_batch, action_batch):
        return

class Ensemble(IPolicyPrior):  # implements policy network from sac.py
    def __init__(self, load_dirr, n_models=10, n_samples=5):
        super(Ensemble, self).__init__() 

        self.load_dirr = load_dirr
        self.policies = []
        self.n_models = n_models
        self.n_samples = n_samples
        
        print(torch.cuda.is_available())

        for idx in range(self.n_models):
            self.policies.append(Policy(state_dim=17, action_dim=6))

        for idx, p in enumerate(self.policies):
            if not torch.cuda.is_available():
                p.load_state_dict(torch.load(self.load_dirr + f'net_{idx}.pt', map_location=torch.device('cpu')))
            else:
                p.load_state_dict(torch.load(self.load_dirr + f'net_{idx}.pt'))
                p.to(device)

    def get_evaluator_sample(self, state_batch, action_batch, sampling=True, mean_tanh=False):
        if sampling:
            pred_samples_ensemble = torch.empty(self.n_samples, state_batch.shape[0], action_batch.shape[1])
            for det_idx in range(self.n_models):
                mean_batch, logstd_batch = self.policies[det_idx].stable_network_forward(state_batch)
                std_batch = torch.exp(logstd_batch)  
                if mean_tanh:
                     pred_samples = torch.distributions.Normal(torch.tanh(mean_batch), std_batch).sample([self.n_samples])
                else:
                     pred_samples = torch.tanh(torch.distributions.Normal(mean_batch, std_batch).sample([self.n_samples]))
                pred_samples_ensemble = torch.cat((pred_samples_ensemble, pred_samples), 0)
            pred_samples_ensemble = pred_samples_ensemble[self.n_samples:]
        else:
            pred_samples_ensemble = torch.empty(self.n_models, state_batch.shape[0], action_batch.shape[1])
            for det_idx in range(self.n_models):
                mean_batch, _ = self.policies[det_idx].stable_network_forward(state_batch)
                assert mean_tanh == True
                pred_samples_ensemble[det_idx] = torch.tanh(mean_batch)

        return pred_samples_ensemble

    def eval_ensemble(self, state_batch, mean_tanh):
        stds = []
        means = []
        # Iterate through NNs
        for p in self.policies:
            mean_batch, logstd_batch = p.stable_network_forward(state_batch)
            std_batch = torch.exp(logstd_batch)
            
            if mean_tanh:
                means.append(torch.tanh(mean_batch))
            else:
                means.append(mean_batch)
            
            stds.append(std_batch)

        means = torch.stack(means)
        stds = torch.stack(stds)

        # implementing Lakshminarayanan
        mean = torch.mean(means, dim=0)
        std = (torch.var(means, dim=0) + torch.mean(stds, dim=0) ** 2) ** 0.5
        return mean, std

    def compute_log_probs(self, state_batch, action_batch, mean_tanh=False, debug=False):
        mean_batch, std_batch = self.eval_ensemble(state_batch, mean_tanh=mean_tanh)
        if not debug:
            if mean_tanh:  # each individual mean was tanh-ed, so mean_batch is in [-1,1] and we do regular normal
                return torch.sum(torch.distributions.Normal(mean_batch,
                                                std_batch).log_prob(action_batch), axis=1).reshape(state_batch.shape[0], 1).to(device)
            else:  # untanhed output, so we take do tanh normal
                # returns vector of length state_batch.shape[0]
                return TanhNormal(mean_batch, 
                                    std_batch).log_prob(action_batch).reshape(state_batch.shape[0], 1).to(device)  
        else:
            return mean_batch, std_batch

    def rollout_offline_policy(self, e):
        rets = []
        for ep in range(10):
            o = e.reset()
            action_batch_temp, _ = self.policies[0].stable_network_forward(torch_ify(o).reshape(1,-1))
            d = False
            t = 0
            ret = 0
            horizon = 1000
            while t < horizon and d is False:
                with torch.no_grad():
                    mean, std = self.compute_log_probs(state_batch=torch_ify(o).reshape(1,-1).to(device),
                                                       action_batch=action_batch_temp.to(device), mean_tanh=False, debug=True)
                    a = np_ify(mean)
                o, r, d, _ = e.step(a)
                ret = ret + r
                t = t + 1
            rets.append(ret)
        ret = np.mean(rets)
        print(rets)  # for debug
        print(ret)  # for debug



# TODO: check llk implicit variance in this; maybe 1/2 since 1/(2var) = 1?
class MC_Dropout(IPolicyPrior):
    def __init__(self, load_dirr, n_models=10, n_samples=5):
        super(MC_Dropout, self).__init__()

        self.load_dirr = load_dirr
        self.policies = []
        self.n_models = n_models
        self.n_samples = n_samples

        for idx in range(self.n_models):
            self.policies.append(MC_Dropout_Network(input_dim=17, output_dim=6))

        for idx, p in enumerate(self.policies):
            if not torch.cuda.is_available():
                p.load_state_dict(torch.load(self.load_dirr + f'net_{idx}.pt', map_location=torch.device('cpu')))
            else:
                p.load_state_dict(torch.load(self.load_dirr + f'net_{idx}.pt'))
                p.to(device)
      
    def get_evaluator_sample(self, state_batch, action_batch):
        bnn_samples = torch.empty(self.n_samples, state_batch.shape[0], action_batch.shape[1])
        for bnn_idx in range(self.n_models) :
            temp = torch.zeros(self.n_samples, state_batch.shape[0], action_batch.shape[1]) 
            for sample_idx in range(self.n_samples):
                temp[sample_idx] = self.policies[bnn_idx].forward(state_batch) 
            bnn_samples = torch.cat((bnn_samples, temp), 0)
        bnn_samples = bnn_samples[self.n_samples:]
        return bnn_samples 

    def compute_log_probs(self, state_batch, action_batch, debug=False):
        bnn_samples = self.get_evaluator_sample(state_batch, action_batch).to(device)  # mc dropout tanh-es output in forward function
        mean_batch = torch.mean(bnn_samples, axis=0)  
        std_batch = torch.std(bnn_samples, axis=0)
        
        if not debug:
            return torch.sum(torch.distributions.Normal(mean_batch, 
                                    std_batch).log_prob(action_batch), axis=1).reshape(state_batch.shape[0], 1)
        else:
            return mean_batch, std_batch

    def rollout_offline_policy(self, e):
        rets = []
        for ep in range(10):
            o = e.reset()
            action_batch_temp = self.policies[0].forward(torch_ify(o).reshape(1,-1))
            d = False
            t = 0
            ret = 0
            horizon = 1000
            while t < horizon and d is False:
                with torch.no_grad():
                    mean, std = self.compute_log_probs(state_batch=torch_ify(o).reshape(1,-1), action_batch=action_batch_temp, debug=True)
                    a = np_ify(mean)
                o, r, d, _ = e.step(a)
                ret = ret + r
                t = t + 1
            rets.append(ret)
        ret = np.mean(rets)
        print(rets)  # for debug
        print(ret)  # for debug



class MFVI(IPolicyPrior):
    def __init__(self, load_dirr, n_models=10, n_samples=2):
        super(MFVI, self).__init__() 

        self.load_dirr = load_dirr
        self.policies = []
        self.n_models = n_models
        self.n_samples = n_samples

        for idx in range(n_models):
            self.policies.append(BayesianNetwork(input_dim=17, output_dim=6))

        for idx, p in enumerate(self.policies):
            if not torch.cuda.is_available():
                p.load_state_dict(torch.load(self.load_dirr + f'net_{idx}.pt', map_location=torch.device('cpu')))
            else:
                p.load_state_dict(torch.load(self.load_dirr + f'net_{idx}.pt'))
                p.to(device) 

    def get_evaluator_sample(self, state_batch, action_batch):
        bnn_samples = torch.empty(self.n_samples, state_batch.shape[0], action_batch.shape[1]).to(device)
        for bnn_idx in range(self.n_models):
            temp = torch.zeros(self.n_samples, state_batch.shape[0], action_batch.shape[1]).to(device) 
            for sample_idx in range(self.n_samples):
                temp[sample_idx], _  = self.policies[bnn_idx].forward(state_batch) 
            bnn_samples = torch.cat((bnn_samples, temp), 0)
        bnn_samples = bnn_samples[self.n_samples:]
        return bnn_samples 

    def compute_log_probs(self, state_batch, action_batch, debug=False):
        bnn_samples = self.get_evaluator_sample(state_batch, action_batch)  # mfvi tanhes output in forward function
        mean_batch = torch.mean(bnn_samples, axis=0)
        std_batch = torch.std(bnn_samples, axis=0)
        
        if not debug:
            return torch.sum(torch.distributions.Normal(mean_batch, 
                                        std_batch).log_prob(action_batch), axis=1).reshape(state_batch.shape[0], 1)
        else:
            return mean_batch, std_batch


    def rollout_offline_policy(self, e):
        rets = []
        for ep in range(10):
            o = e.reset()
            action_batch_temp, _  = self.policies[0].forward(torch_ify(o).reshape(1,-1))
            d = False
            t = 0
            ret = 0
            horizon = 1000
            while t < horizon and d is False:
                with torch.no_grad():
                    mean, std = self.compute_log_probs(state_batch=torch_ify(o).reshape(1,-1), action_batch=action_batch_temp, debug=True)
                    a = np_ify(mean)
                o, r, d, _ = e.step(a)
                ret = ret + r
                t = t + 1
            rets.append(ret)
        ret = np.mean(rets)
        print(rets)  # for debug
        print(ret)  # for debug



def factory_method(load_dirr, policy_type):
    if policy_type == "MFVI":
        return MFVI(load_dirr=load_dirr)  

    elif policy_type == "MC_Dropout":
        return MC_Dropout(load_dirr=load_dirr)  

    elif policy_type == "ensemble":
        return Ensemble(load_dirr=load_dirr)


class GaussianMSELoss(torch.nn.Module):  # do not use this for evaluation
    # do not use this for evaluation because it uses predicted stds rather than llk variance
    def __init__(self):
        super(GaussianMSELoss, self).__init__()

    def forward(self, mu, target, logstd, logvar_loss=False):  # do not use predictive variance from the model
        logvar = 2 * logstd
        inv_var = (-logvar).exp()
        if logvar_loss:
            return (logvar + (target - mu) ** 2 * inv_var).mean()
        else:
            return ((target - mu) ** 2).mean()


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return from_numpy(np_array_or_other)
    else:
        return np_array_or_other
def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return get_numpy(tensor_or_other)
    else:
        return tensor_or_other


# MAIN =========================================================
@click.command(help='')
@click.option('--eval_metric', type=str, default="exp_mse")
@click.option('--env_name', type=str, default="halfcheetah-medium-expert-v0")
@click.option('--path_to_dataset', type=str, default="/data/ziz/iliant/datasets/halfcheetah_medium_expert.hdf5")
@click.option('--load_dirr', type=str, default="/data/ziz/iliant/KL/robust-offline-rl/behavioral_prior/MC_ensemble_final/lr_0.0004_wd_1e-06_bs_256/")
@click.option('--torch_seed', type=int, default=0)
@click.option('--policy_type', type=str, default='MC_Dropout')

def main(eval_metric, env_name, path_to_dataset, load_dirr, torch_seed,
        policy_type):

    print(policy_type)
    print(load_dirr)

    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    random.seed(torch_seed)

    _ , _ , observations, actions = get_train_test_data(env_name, path_to_dataset,
                                                        default=False)

    obs_dim = len(observations[0])
    act_dim = len(actions[0])
 

    policy = factory_method(load_dirr=load_dirr, policy_type=policy_type)
    samples_policy = policy.get_evaluator_sample(observations, actions)
    
    e = gym.make(env_name)
    print("Performance on the online rollouts... \n")
    policy.rollout_offline_policy(e)

    if eval_metric == "exp_mse":
        print("Expected mse for mfvi is: ")
        print(expected_mse(samples_policy, actions).item())
    elif eval_metric == "mse_of_mean":
        print("MSE of mean for mfvi is: ")
        print(mse(samples_policy, actions).item()) 
        
   
if __name__ == '__main__':
    main()
