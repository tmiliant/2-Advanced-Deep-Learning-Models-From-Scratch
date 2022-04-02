# 2-Advanced-Deep-Learning-Models-From-Scratch

I implemented the algorithms presented in the following two papers https://arxiv.org/abs/1506.02142, https://arxiv.org/abs/1505.05424 from scratch, using PyTorch, and coded in the Vim editor.
I also implemented a factory method that allows to decide, at runtime, which of the models to use. Finally, I included a ~400LOC file with relatively complex code that evaluates prior policies learned using models such as Bayesian Network (with Mean Field Variational Inference), Ensembles of Neural Networks with Early Stopping, and Monte Carlo Dropout. Predictive distributions are also obtained for these models, for example for MC Dropout mask is not only applied during training, but during prediction as well.
