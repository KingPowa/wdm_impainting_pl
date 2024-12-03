_authors = "https://github.com/SayantanKumar/ISBI_MoPoE/blob/main/multimodal_VAE.py"
_modified_by = "Francesco Sammarco"

import torch
import pytorch_lightning as pl
import hydra
import abc
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from typing import List, Tuple, Any, Dict
from collections.abc import Collection
from omegaconf import DictConfig, ListConfig
from ml.core.utils import calculate_output_shape
from ml.core.loss import elbo_loss
from itertools import combinations

class MVAE(pl.LightningModule): # @param n_latents : number of latent dimensions
    
    def __init__(self, config: DictConfig):
        
        super(MVAE, self).__init__()
        self.config = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Configuration of the latent
        #  dimensions
        self.n_latents = config.network_config.latent_dim
        self.cond_shape = config.network_config.cond_shape if "cond_shape" in config.network_config else None
        self.prior_mu, self.prior_var = [torch.Tensor([val]).to(device) for val in config.network_config.prior]

        # Modalities
        self.configurations = {mconf.modality: mconf for mconf in config.data_config}

        self.encoders = nn.ModuleList(
            [hydra.utils.instantiate(
                enc_cfg, 
                lat_dim = self.n_latents, 
                cond_shape = self.cond_shape, 
                shape = self.configurations[enc_cfg.modality].shape) 
            for enc_cfg in config.modalities.encoders]
        )
        self.decoders = nn.ModuleList(
            [hydra.utils.instantiate(
                dec_cfg, 
                lat_dim = self.n_latents, 
                cond_shape = self.cond_shape, 
                shape = self.configurations[dec_cfg.modality].shape,
                enc_output_shape=enc.output_shape) 
            for dec_cfg, enc in zip(config.modalities.decoders, self.encoders)]
        )
        self.poe = ProductOfExperts()
        self.moe = MixtureOfExperts()
        self._training = True

        # Subset of modalities
        self.Xk = self.__calculate_subsets(config.modalities.encoders)
    
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
            
        if self._training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
            
        else:
            return mu
        
    def forward(self, inputs, age_cond):
        subset_params, mopoe_params = self.infer(inputs, age_cond)
        outputs = self.decode(mopoe_params, age_cond)
            
        return {"outputs": outputs, "subsets_posteriors": subset_params, "mopoe_posterior": mopoe_params}
    
    def decode(self, posterior_mopoe: Collection[torch.Tensor], age_cond):

        # Use the mopoe_params to sample z and decode it
        mu, logvar = posterior_mopoe
        z = self.reparametrize(mu, logvar)
        #z_cond = torch.cat((z, age_cond), dim= -1)
        outputs = torch.stack([decode(z, age_cond) for decode in self.decoders], dim=1) # (B, D, H, W)
        return outputs
        
    def infer(self, inputs, age_cond): 

        # Create inferred mean and log-variance per modality
        (mu, logvar) = [], []
        for i, encoder in enumerate(self.encoders):
            input_mu, input_logvar = encoder(inputs[:,i,...], age_cond) # Modality is stacked on dim 1, suppose (B, D, H, W)
            mu.append(input_mu) # Append (B, D)
            logvar.append(input_logvar)
        (mu, logvar) = (torch.stack(mu), torch.stack(logvar)) # tuple with size (D, B, dim)

        # When training, compute per each combination of modalities the posterior q(z|x)
        mu_out = []
        logvar_out = []
        if self._training:
            for x_k in self.Xk:
                mu_k: torch.Tensor = mu[x_k] # (M, B, dim) where M is the number of experts picked by th subset (len(x_k))
                logvar_k: torch.Tensor = logvar[x_k]
                mu_k, logvar_k = self.poe(mu_k, logvar_k) # (1, dim) 
                mu_out.append(mu_k)
                logvar_out.append(logvar_k)

            (mu_out, logvar_out) = (torch.stack(mu_out), torch.stack(logvar_out)) # Stacking on 0 -> num of modalities subsets (COMB x dim)     

            moe_mu, moe_logvar = self.moe(mu_out, logvar_out) # (1 x DIM)

            return [(mu_out, logvar_out), (moe_mu, moe_logvar)]
        
        # If not traning, skip creating the distributions
        else:
            for x_k in self.Xk:
                mu_k: torch.Tensor = mu[x_k]
                logvar_k: torch.Tensor = logvar[x_k]
                mu_k, logvar_k = self.poe(mu_k, logvar_k)    
                mu_out.append(mu_k)
                logvar_out.append(logvar_k)

            (mu_out, logvar_out) = (torch.stack(mu_out), torch.stack(logvar_out)) # Stacking on 0 -> num of modalities subsets     

            moe_mu, moe_logvar = self.moe(mu_out, logvar_out)

            return [None, (moe_mu, moe_logvar)]
        
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.__step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.__step(batch, batch_idx, stage="val")
    
    def __step(self, batch, batch_idx, stage):
        inputs, age_cond = batch
        output = self.forward(inputs, age_cond)
        loss = self.calc_loss(batch, output)
        for loss_n, loss_val in loss.items():
            self.log(
                f"{stage}_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def configure_optimizers(self):
        # You can define the optimizer here
        optimizer = Adam(self.parameters(), lr=self.config.train_config.learning_rate)
        return optimizer
    
    def calc_loss(self, batch, model_output):
        inputs, age_cond = batch
        outputs = model_output["outputs"]
        loss = elbo_loss(inputs=inputs, outputs=outputs, 
                         subset_params=model_output["subsets_posteriors"], mopoe_params=model_output["mopoe_posterior"], 
                         prior_mean=self.prior_mu, prior_logvar=self.prior_var.log(), beta=self.config.train_config.beta_kl)
        return loss
    
    def __calculate_subsets(self, enc_cfgs: ListConfig):
        self.n_modalities = len(enc_cfgs)
        xs = list(range(0, self.n_modalities))
        tmp = [list(combinations(xs, n+1)) for n in range(len(xs))]
        subset_list = [list(item) for sublist in tmp for item in sublist]
        return subset_list

# ----------------------------------------------------------------------------------------------------------------------------------

class MLPLayer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, activation: str = None):
        super(MLPLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.fc(x))
    
class FCLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, filter_dim: int, stride: int, padding: int = 1, activation: str = None):
        super(FCLayer, self).__init__()
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=(filter_dim, filter_dim), stride=stride, padding=padding)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.fc(x))
    
class UpsampleFCLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, filter_dim: int, stride: int, padding: int = 1, activation: str = None):
        super(UpsampleFCLayer, self).__init__()
        self.fc = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(filter_dim, filter_dim), stride=stride, padding=padding, output_padding=padding)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.fc(x))
    
class BaseAdapter():

    @abc.abstractmethod
    def encoder(self, in_channels: int, act: str = None) -> nn.Module:
        raise NotImplementedError("Abstract method")
    
    @abc.abstractmethod
    def decoder(self, out_channels: int, act: str = None) -> nn.Module:
        raise NotImplementedError("Abstract method")
            
# ----------------------------------------------------------------------------------------------------------------------------------

class BaseModalityCoder(nn.Module):

    def __init__(self, modality: str, lat_dim: int, cond_shape: Collection[int], shape: Collection[int]):
        super(BaseModalityCoder, self).__init__()
        self.modality = modality
        self.covariate = cond_shape is not None
        self.shape = shape
        self.lat_dim = lat_dim
        self.cond_shape = cond_shape

    def set_modality(self, modality: str):
        self.modality = modality

# ----------------------------------------------------------------------------------------------------------------------------------

class BaseModalityEncoder(BaseModalityCoder):
        
    def create_output_layers(self, out_shape: int):
        self.mu = MLPLayer(out_shape, self.lat_dim)
        self.var = MLPLayer(out_shape, self.lat_dim)
    
    @abc.abstractmethod
    def preproc(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract method")
    
    @abc.abstractmethod
    def join_cov(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract method")

    def forward(self, x, c):
        x = self.preproc(x)
        x = self.join_cov(x, c)
        for layer in self.layers:
            x = layer(x)

        return self.mu(x), self.var(x)  


class ModalityMLPEncoder(BaseModalityEncoder):

    def __init__(self, n_neurons: Collection[int], activations: Collection[str], *args, **kwargs):
        super(ModalityMLPEncoder, self).__init__(*args, **kwargs)
        curr_shape = self.lat_dim + self.cond_shape if self.covariate else self.lat_dim
        mlp_layers : List = []

        for fc_dim, fc_act in zip(n_neurons, activations):
            mlp_layers.append(MLPLayer(curr_shape, fc_dim, fc_act))
            curr_shape = fc_dim

        self.layers = nn.Sequential(*mlp_layers)
        self.create_output_layers(curr_shape)

        self.output_shape = self.lat_dim
    
    def preproc(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
    def join_cov(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([x,c], dim=-1) if self.covariate else x
    
class ModalityCNNEncoder(BaseModalityEncoder):

    def __init__(self, channels: Collection[int], filters: Collection[int], activations: Collection[str], strides: Collection[int] = None,  *args, **kwargs):
        super(ModalityCNNEncoder, self).__init__(*args, **kwargs)
        if strides is None:
            strides = [1 for _ in range(len(channels))]
        curr_n_channels = self.shape[0] if not self.covariate else self.shape[0]+self.cond_shape
        curr_shape = self.shape
        fc_layers : List = []

        for channel, filter, stride, act in zip(channels, filters, strides, activations):
            fc_layers.append(FCLayer(curr_n_channels, channel, filter, stride, 1, act))
            curr_n_channels = channel
            curr_shape = calculate_output_shape(curr_shape, filter, stride, 1, 1, channel)
        fc_layers.append(nn.Flatten())

        self.layers = nn.Sequential(*fc_layers)
        self.create_output_layers(np.array(curr_shape).prod())

        self.output_shape = curr_shape
    
    def preproc(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def join_cov(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.covariate:
            # Reshape the normalized value and gender to 1xWxL tensors
            # Assuming W and L can be inferred from the tensor's shape
            B, W, L = x.shape[0], x.shape[2], x.shape[3]  # Assuming tensor shape is BxCxWxL
            normalized_tensor = c[:, 0].detach().view(B, -1, 1, 1).repeat(1, 1, W, L)
            gender_tensor = c[:, 1].detach().view(B, -1, 1, 1).repeat(1, 1, W, L)
            return torch.cat((x, normalized_tensor, gender_tensor), dim=1)
        return x

# ----------------------------------------------------------------------------------------------------------------------------------

class BaseModalityDecoder(BaseModalityCoder):

    def forward(self, z: torch.Tensor, c: torch.Tensor):
        z = torch.cat((z, c), dim=-1) if self.covariate else z
        for layer in self.layers:
            z = layer(z)

        z = self.out(z)
        return z.view(z.shape[0], *self.shape)
    
class ModalityMLPDecoder(BaseModalityDecoder):

    def __init__(self, enc_output_shape: Collection[int], n_neurons: Collection[int], activations: Collection[str], *args, **kwargs):
        super(ModalityMLPDecoder, self).__init__(*args, **kwargs)
        curr_shape = self.lat_dim + self.cond_shape if self.covariate else self.lat_dim
        mlp_layers : List = []

        for fc_dim, fc_act in zip(n_neurons, activations):
            mlp_layers.append(MLPLayer(curr_shape, fc_dim, fc_act))
            curr_shape = fc_dim

        self.layers = nn.Sequential(*mlp_layers)
        self.out = MLPLayer(curr_shape, np.array(self.shape).prod(), "sigmoid")

class ModalityCNNDecoder(BaseModalityDecoder):

    def __init__(self, enc_output_shape: Collection[int], channels: Collection[int], filters: Collection[int], activations: Collection[str],  strides: Collection[int] = None, *args, **kwargs):
        super(ModalityCNNDecoder, self).__init__(*args, **kwargs)

        if strides is None:
            strides = [1 for _ in range(len(channels))]
        input_dim = self.lat_dim + self.cond_shape if self.covariate else self.lat_dim
        curr_n_channels = enc_output_shape[0]
        fc_layers : List = []

        fc_layers.append(MLPLayer(input_dim=input_dim, output_dim=np.array(enc_output_shape).prod()))
        fc_layers.append(nn.Unflatten(1, tuple(enc_output_shape)))

        for channel, filter, stride, act in zip(channels, filters, strides, activations):
            fc_layers.append(UpsampleFCLayer(curr_n_channels, channel, filter, stride, 1, act))
            curr_n_channels = channel

        self.layers = nn.Sequential(*fc_layers)
        self.out = UpsampleFCLayer(curr_n_channels, self.shape[0], filter, stride, 1, "sigmoid")

# ----------------------------------------------------------------------------------------------------------------------------------


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.
    Implementation from: https://github.com/thomassutter/MoPoE

    Args:
    mus (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvars (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mus, logvars):

        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        weights = (1/num_components) * torch.ones(num_components).to(mus[0].device)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k-1])
            if k == num_components-1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples*weights[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples

        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])

        return mu_sel, logvar_sel

# ----------------------------------------------------------------------------------------------------------------------------------

# In case someone wonders... https://discuss.pytorch.org/t/is-there-any-different-between-torch-sigmoid-and-torch-nn-functional-sigmoid/995
class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)
      
class Sigmoid(nn.Module):
    def forward(self, x):
        return F.sigmoid(x)
    
class Relu(nn.Module):
    def forward(self, x):
        return F.relu(x)
    
class Tanh(nn.Module):
    def forward(self, x):
        return F.tanh(x)

def get_activation(activation: str) -> nn.Module:
    if activation is None: return nn.Identity()
    if activation == "relu": return Relu()
    if activation == "sigmoid": return Sigmoid()