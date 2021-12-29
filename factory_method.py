from Bayesian_Neural_Network.bayesian_network import BayesianNetwork
from Monte_Carlo_Dropout.mcd import MC_Dropout_Network 


def factory_method(model, input_dim, output_dim):
    if model == 'mc dropout':
        return MC_Dropout_Network(input_dim, output_dim) 

    elif model == 'bayes by backprop':
        return BayesianNetwork(input_dim, output_dim)

    else:
        raise ValueError('This is not one of the expected models.')
