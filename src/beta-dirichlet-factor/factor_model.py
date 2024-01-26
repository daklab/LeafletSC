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
import random
import datetime

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
    #print(log_probs.sum())
    # Sum the log probabilities
    return log_probs.sum()


def model(y, total_counts, K, use_global_prior=True):

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
    use_global_prior (bool, optional): Whether to use a global prior for psi. Default is True.

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

    if use_global_prior:

        # Global priors for PSI
        
        # sample K values for a and b
        a = pyro.sample("a", dist.Gamma(1., 1.).expand([K]).to_event(1)) # watch out for these .to_event(1), if using multiple a,bs (one for each K, need to include this so they are treated independently)
        b = pyro.sample("b", dist.Gamma(1., 1.).expand([K]).to_event(1))

        # with a common prior, capturing the idea that some junctions might be 
        # consistently used across different factors.
        # Capture shared + unique behavior of each junction

        # some junctions will be more globally used (more consitutive regardless of splicing)
        # a and b for every junction shared across factors 

        # Sample PSI for each junction
        with pyro.plate('junctions', P):
            psi = pyro.sample("psi", dist.Beta(a, b).to_event(1))
        
        psi = psi.T # shape is K,P

    # original code check, check if there was one a globally and b globally 

    else:
        # Independent junction-specific parameters without global influence
        a = pyro.sample("a", dist.Gamma(1., 1.).expand([P]).to_event(1))
        b = pyro.sample("b", dist.Gamma(1., 1.).expand([P]).to_event(1))# per junction to model average behaviour
        # in this setup,the behavior of each junction is modeled independently of others
        psi_dist = dist.Beta(a, b).expand([K, P]).to_event(2) 
        psi = pyro.sample("psi", psi_dist) # shape is K,P

    # Sampling pi values and conc for each factor
    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(K) / K)) # shape is K, represents the base probabilities for each of the K factors via uniform prior (initially all factors are equally likely)
    conc = pyro.sample("conc", dist.Gamma(1, 1)) # value scales the pi vector (higher conc makes the sampled probs more uniform, a lower conc allows more variability, leading to probability vectors that might be skewed towards certain factors).

    # Sampling the assignment of each cell to a factor
    with pyro.plate('data2', N):
        assign_dist = dist.Dirichlet(pi * conc)
        assign = pyro.sample("assign", assign_dist)

    # Compute the predicted probabilities for each cell  (mixture of beta distributions)
    pred = torch.mm(assign, psi)

    #print("a shape:", a.shape, "b shape:", b.shape)
    #print("psi shape:", psi.shape, "pred shape:", pred.shape)
    # Use custom log probability function to compute the likelihood of observations
    log_prob = my_log_prob(y, total_counts, pred) 
    pyro.factor("obs", log_prob) 

def fit(y, total_counts, K, use_global_prior, guide, patience=5, min_delta=0.01, lr=0.05, num_epochs=500):
    
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
        #print(j)
        # Pass use_global_prior to the model in the SVI step
        loss = svi.step(y, total_counts, K, use_global_prior)
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
    return losses

def main(y, total_counts, num_initializations=5, seeds=None, file_prefix=None, use_global_prior=True, save_to_file=True, K=50, loss_plot=True, lr = 0.05, num_epochs=100):

    """
    Main function to fit the Bayesian model.

    Parameters:
    y (torch.Tensor): A tensor representing observed junction counts.
    total_counts (torch.Tensor): A tensor representing the observed intron cluster counts.
    K (int, optional): The number of components in the mixture model. Default is 50.
    num_initializations (int, optional): Number of random initializations. Default is 5.
    seeds (list, optional): List of seeds for random initializations. Default is None, which will generate random seeds.
    """

    # If seeds are not provided, create a list of random seeds
    if seeds is None:
        seeds = [random.randint(1, 10000) for _ in range(num_initializations)]

    all_results = []

    if use_global_prior:
        print("Using global prior for psi, junctions are shared across factors")
    else:
        print("Not using global prior for psi, junctions are independent across factors")
        
    for i, seed in enumerate(seeds):
        print(f"Initialization {i+1} with seed {seed}")

        # Set the seed
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Define the guide
        print("Define the guide")
        guide = AutoDiagonalNormal(model)

        # Fit the model
        print("Fit the model")
        losses = fit(y, total_counts, K, use_global_prior, guide, patience=5, min_delta=0.01, lr=lr, num_epochs=num_epochs)
        if loss_plot:
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss Plot for Initialization {i+1}")
            plt.show()

        # Sample from the guide (posterior)
        print("Sample from the guide (posterior)")
        sampled_guide = guide()
        guide_trace = pyro.poutine.trace(guide).get_trace(y, total_counts, K, use_global_prior)

        # Extract the latent variables 
        print("Extract the latent variables")
        latent_vars = {name: node["value"].detach().cpu().numpy() for name, node in guide_trace.nodes.items() if node["type"] == "sample"}
        all_results.append({
            'seed': seed,
            'losses': losses,
            'sampled_guide': sampled_guide,
            'latent_vars': latent_vars
        })

    print("All initializations complete. Returning results.")
    print("------------------------------------------------")

    if save_to_file:
        print("Saving results to file")

        # add date time K to file name
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if file_prefix is None:
            file_name = f"results_{date}_{K}_{num_initializations}_factors.pt"
        else:
            file_name = f"{file_prefix}_{date}_{K}_{num_initializations}_factors.pt"
        torch.save(all_results, file_name)
        print(f"Results saved to {file_name}")
        print("------------------------------------------------")

    return all_results

if __name__ == "__main__":
    main()