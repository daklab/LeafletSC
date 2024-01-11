# factor_model.py 
# maintainer: Karin Isaev 
# date: 2024-01-11

# purpose:  
#   - define a probabilistic Bayesian model using a Beta-Dirichlet factorization to infer cell states driven by splicing differences across cells
#   - fit the model using Stochastic Variational Inference (SVI)
#   - sample from the guide (posterior)
#   - extract the latent variables

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
import torch
import matplotlib.pyplot as plt

def my_log_prob(y_sparse, total_counts_sparse, pred):
    
    """
    Compute the log probability of observed data under a binomial distribution.

    This function calculates the log probabilities for non-zero elements in a sparse
    tensor representation of observed data. It assumes the data follows a binomial 
    distribution parameterized by the predictions (`pred`).

    .log_prob(y_values) calculates the log probability of observing y_values successes given the binomial distribution specified.

    Parameters:
    y_sparse (torch.Tensor): A sparse tensor representing observed junction count data.
    total_counts_sparse (torch.Tensor): A sparse tensor representing the total intron cluster counts 
                                        to which each junction belongs to (i.e., the 'n' in a binomial distribution).
    pred (torch.Tensor): A dense tensor of predicted probabilities (i.e., the 'p' in a binomial distribution).

    Returns:
    torch.Tensor: The sum of log probabilities for all observed non-zero data points.
    """
        
    # Extract non-zero elements and their indices from y and total_counts
    y_indices = y_sparse._indices() #y_indices[0] is the row index, y_indices[1] is the column index
    y_values = y_sparse._values()

    total_counts_indices = total_counts_sparse._indices()
    total_counts_values = total_counts_sparse._values()

    # Ensure that y and total_counts align on the same indices
    if not torch.equal(y_indices, total_counts_indices):
        raise ValueError("The indices of y_indices and total_counts_indices do not match.")

    # Compute log probabilities at these indices
    log_probs = dist.Binomial(total_counts_values, pred[y_indices[0], y_indices[1]]).log_prob(y_values)
    # Sum the log probabilities
    return log_probs.sum()


def model(y, total_counts, K):

    """
    Define a probabilistic Bayesian model using a Beta-Dirichlet factorization.

    The model assumes the observed data (junction and intron cluster counts) are generated from a mixture of 
    beta distributions, with latent variables and priors modeling various aspects of the data. 

    By using a mixture of these Beta-distributed psi variables, the model accounts for the possibility that the 
    observed data might be influenced by several different factors or cell states, each 
    with its own characteristic distribution of values.

    Parameters:
    y (torch.Tensor): A tensor representing observed data (junction counts).
    total_counts (torch.Tensor): A tensor representing the total counts for each observation (total intron cluster counts).
    K (int): The number of factors representing cell states.

    Latent Variables and Priors:
    - a, b: Parameters for the Beta distribution, modeling the average behavior per junction, sampled from a Gamma distribution.
    - psi: The unknown probabilities for each factor and junction, sampled from a Beta distribution.
    - pi: Category probabilities for each factor, sampled from a Dirichlet distribution.
    - conc: Concentration parameter for the Dirichlet distribution, sampled from a Gamma distribution.

    Likelihood:
    Modeled by a Binomial distribution, with the likelihood of observations conditioned on the latent variables.

    Model Specifications:
    Prior distributions are defined for each latent variable, and the likelihood is computed using a custom log probability function.

    Returns:
    None: This function contributes to the Pyro model's trace and does not return any value.
    """

    N, P = y.shape

    a = pyro.sample("a", dist.Gamma(1., 1.).expand([P]).to_event(1))
    b = pyro.sample("b", dist.Gamma(1., 1.).expand([P]).to_event(1))

    psi_dist = dist.Beta(a, b).expand([K, P]).to_event(2)
    psi = pyro.sample("psi", psi_dist)

    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(K) / K))
    conc = pyro.sample("conc", dist.Gamma(1, 1))

    with pyro.plate('data1', N):
        assign_dist = dist.Dirichlet(pi * conc)
        assign = pyro.sample("assign", assign_dist)

    pred = torch.mm(assign, psi)

    # Use custom log probability
    log_prob = my_log_prob(y, total_counts, pred) # double check that I am not double summing here 
    #pyro.factor("obs", log_prob.sum()) <- i think this one is wrong 
    pyro.factor("obs", log_prob) # this should be correct 

def fit(y, total_counts, K, guide, patience=5, min_delta=0.01, lr=0.05, num_epochs=500):
    
    """
    Fit a probabilistic model using Stochastic Variational Inference (SVI).

    This function performs optimization to fit the model to the data. It uses the SVI approach
    with the Adam optimizer, and includes features like early stopping to prevent overfitting.

    Parameters:
    y (torch.Tensor): A tensor representing observed junction counts.
    total_counts (torch.Tensor): A tensor representing the observed intron cluster counts.
    K (int): The number of components in the mixture model.
    guide (function): A guide function for the SVI (variational distribution).
    patience (int, optional): The number of epochs to wait without improvement before stopping early. Default is 5.
    min_delta (float, optional): Minimum change in the loss to qualify as an improvement. Default is 0.01.
    lr (float, optional): Learning rate for the Adam optimizer. Default is 0.05.
    num_epochs (int, optional): Maximum number of epochs for training. Default is 500.

    Returns:
    list: A list of loss values for each epoch during the optimization process.

    The function iteratively updates the model parameters using the SVI step method.
    It monitors the training loss, and if the loss does not improve by at least `min_delta`
    for a `patience` number of consecutive epochs, the training will stop early. This helps
    in preventing overfitting. The function returns the history of loss values for each epoch.
    """    
    
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    losses = []
    best_loss = float('inf')
    epochs_since_improvement = 0

    for j in range(num_epochs):
        loss = svi.step(y, total_counts, K)
        losses.append(loss)

        # Check for improvement
        if best_loss - loss > min_delta:
            best_loss = loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Early stopping check
        if epochs_since_improvement >= patience:
            print(f"Stopping early at epoch {j}")
            break

        if j % 10 == 0:
            print(f"Epoch {j}, Loss: {loss}")

        #torch.cuda.empty_cache()  # Attempt to free unused memory
    return losses

def main(y, total_counts, K=50, loss_plot=True):

    """
    Main function to fit the Bayesian model.

    Parameters:
    y (torch.Tensor): A tensor representing observed junction counts.
    total_counts (torch.Tensor): A tensor representing the observed intron cluster counts.
    K (int, optional): The number of components in the mixture model. Default is 50.
    """

    # Define the guide
    guide = AutoDiagonalNormal(model)

    # Fit the model
    losses = fit(y, total_counts, K, guide, patience=5, min_delta=0.01, lr=0.05, num_epochs=500)
    if loss_plot:
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    # Sample from the guide (posterior)
    sampled_guide = guide()
    guide_trace = pyro.poutine.trace(guide).get_trace(y, total_counts, K)

    # Extract the latent variables 
    latent_vars = {name: node["value"].detach().cpu().numpy() for name, node in guide_trace.nodes.items() if node["type"] == "sample"}

    print("Done! Returning losses, sampled_guide, and latent_vars.")
    return losses, sampled_guide, latent_vars

if __name__ == "__main__":
    main()