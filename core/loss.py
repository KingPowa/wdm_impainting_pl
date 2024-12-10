import torch
import numpy as np
import torch.nn.functional as F

from collections.abc import Collection
from typing import Tuple

def elbo_loss(inputs, outputs, subset_params, mopoe_params, prior_mean, prior_logvar, beta):
    """
    Compute the ELBO loss as per the formula in the image.
    
    Args:
    - inputs: List containing inputs per each modality
    - outputs: outputs of the network
    - subset_params: List of dictionaries, each containing `mean` and `logvar` 
                     (log variance) for each subset of modalities.
    - mopoe_params: Dictionary containing `mean` and `logvar` (log variance) 
                    for the MOPOE posterior.
    - prior_mean: Mean of the prior distribution for z.
    - prior_logvar: Log variance of the prior distribution for z.
    - beta: how much weight has the KL
    
    Returns:
    - elbo: The ELBO loss (negative ELBO, for minimization).
    """
    # Unpack the mopoe mean and logvar
    mopoe_mean, mopoe_logvar = mopoe_params
    
    # Reconstruction loss: sum over all modalities
    recon_loss = calc_ll(inputs, outputs)
    
    # KL divergence between MOPOE posterior and the prior
    # kl_div_mopoe = kl_divergence_logvar(mopoe_mean, mopoe_logvar, prior_mean, prior_logvar)
    # kl_div_poe = kl_div_mopoe

    kl_div_poe = 0
    
    # KL divergence between each POE subset posterior and the prior
    kl_div_poe += kullback_divergence_subsets(subset_params, prior_mean, prior_logvar)
    
    # Since the ELBO is maximized, we return the negative of the sum (to minimize)
    elbo = recon_loss + beta*kl_div_poe
    return {"loss": elbo, "recon_err": recon_loss, "kullback": kl_div_poe}

def kullback_divergence_subsets(subset_params: Collection[Tuple[torch.Tensor, torch.Tensor]], prior_mu: torch.Tensor, prior_logvar: torch.Tensor):
    weight = 1/subset_params[0].shape[0]
    kl = kl_divergence_logvar(subset_params[0], subset_params[1], prior_mu, prior_logvar).mean(1).sum() # Mean over batches, sum over modalities and dimensions

    return kl*weight

import torch

def kl_divergence_logvar(mean_q: torch.Tensor, logvar_q: torch.Tensor, mean_p: torch.Tensor, logvar_p: torch.Tensor):
    """
    Calculate the Kullback-Leibler (KL) divergence between two Gaussian distributions
    with log variances provided as input.
    
    Parameters:
    -----------
    mean_q : torch.Tensor
        Mean of the first distribution (q).
    logvar_q : torch.Tensor
        Log variance of the first distribution (q).
    mean_p : torch.Tensor
        Mean of the second distribution (p).
    logvar_p : torch.Tensor
        Log variance of the second distribution (p).

    Returns:
    --------
    kl_div : torch.Tensor
        KL divergence between the two distributions.
    """
    # Calculate the exponential of the log variances to get variances
    var_q = torch.exp(logvar_q)  # σ_q^2
    var_p = torch.exp(logvar_p)  # σ_p^2
    
    # Calculate the KL divergence term by term
    log_var_ratio = logvar_p - logvar_q  # log(σ_p^2 / σ_q^2)
    var_ratio = var_q / var_p            # (σ_q^2 / σ_p^2)
    mean_diff_squared = (mean_q - mean_p).pow(2) / var_p  # (μ_q - μ_p)^2 / σ_p^2
    
    # Combine the terms to compute the KL divergence
    kl_div = 0.5 * (log_var_ratio + var_ratio + mean_diff_squared - 1)
    
    return kl_div

def calc_ll(x: Collection[torch.Tensor], outputs: Collection[torch.Tensor]):
    """
    Calculate the log-likelihood for each subset of modalities.
    
    Args:
    - x (list of torch.Tensor): A list of input data tensors, one for each subset.
    - subset_params (list of dict): Each dictionary contains 'mean' and 'logvar' (log variance) 
      for the predicted distribution for the corresponding subset.
    
    Returns:
    - ll (torch.Tensor): Log-likelihood for all subsets.
    """
    ll = 0  # Initialize log-likelihood to zero
    
    for input, output in zip(x, outputs):
        ll += F.mse_loss(output, input, reduction='sum')
        
    return ll

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


