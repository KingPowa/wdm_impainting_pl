import torch
from typing import Tuple
from collections.abc import Collection
from datetime import datetime
import numpy as np
from noise import pnoise2
from scipy.ndimage import gaussian_filter

def get_timestamp():
    return datetime.timestamp(datetime.now())

def calculate_output_shape(input_shape: Collection[int], 
                           kernel_size: int | Tuple[int, int], 
                           stride: int | Tuple[int, int]=1, 
                           padding: int | Tuple[int, int]=0, 
                           dilation: int =1, out_channels: int =None):
    """
    Calculate the output shape after a convolutional layer.
    
    Args:
        input_shape (tuple): The shape of the input image (batch_size, channels, height, width) or (channels, height, width).
        kernel_size (int or tuple): The size of the convolutional kernel.
        stride (int or tuple): The stride of the convolution. Default is 1.
        padding (int or tuple): The padding applied to the input. Default is 0.
        dilation (int): The dilation rate of the kernel. Default is 1.
        out_channels (int): The number of output channels (number of filters). Required for the number of channels in the output.
    
    Returns:
        tuple: The shape of the output image.
    """
    
    if len(input_shape) == 4:  # Batch size present
        batch_size, in_channels, height, width = input_shape
    elif len(input_shape) == 3:  # No batch size
        in_channels, height, width = input_shape
        batch_size = None
    else:
        raise ValueError("Invalid input shape. Must be of length 3 or 4.")
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # Output height and width calculation
    output_height = (height + 2 * padding[0] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
    output_width = (width + 2 * padding[1] - dilation * (kernel_size[1] - 1) - 1) // stride[1] + 1
    
    if out_channels is None:
        out_channels = in_channels  # If not provided, keep the same number of channels
    
    # Return the output shape
    if batch_size is not None:
        return (batch_size, out_channels, output_height, output_width)
    else:
        return (out_channels, output_height, output_width)
    

def calculate_transpose_conv_output_shape(input_shape, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, out_channels=None):
    """
    Calculate the output shape after a transposed convolutional (ConvTranspose) layer.
    
    Args:
        input_shape (tuple): The shape of the input image (batch_size, channels, height, width) or (channels, height, width).
        kernel_size (int or tuple): The size of the convolutional kernel.
        stride (int or tuple): The stride of the convolution. Default is 1.
        padding (int or tuple): The padding applied to the input. Default is 0.
        output_padding (int or tuple): Additional size added to the output shape. Default is 0.
        dilation (int): The dilation rate of the kernel. Default is 1.
        out_channels (int): The number of output channels (number of filters). Required for the number of channels in the output.
    
    Returns:
        tuple: The shape of the output image.
    """
    
    if len(input_shape) == 4:  # Batch size present
        batch_size, in_channels, height, width = input_shape
    elif len(input_shape) == 3:  # No batch size
        in_channels, height, width = input_shape
        batch_size = None
    else:
        raise ValueError("Invalid input shape. Must be of length 3 or 4.")
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    
    # Output height and width calculation for ConvTranspose
    output_height = (height - 1) * stride[0] - 2 * padding[0] + dilation * (kernel_size[0] - 1) + 1 + output_padding[0]
    output_width = (width - 1) * stride[1] - 2 * padding[1] + dilation * (kernel_size[1] - 1) + 1 + output_padding[1]
    
    if out_channels is None:
        out_channels = in_channels  # If not provided, keep the same number of channels
    
    # Return the output shape
    if batch_size is not None:
        return (batch_size, out_channels, output_height, output_width)
    else:
        return (out_channels, output_height, output_width)
    
def reparametrize(mu: torch.Tensor, var: torch.Tensor, log=False):

    std = var.mul(0.5).exp_() if log else var.mul(0.5)
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)

from scipy.ndimage import gaussian_filter

def generate_perlin_mask_with_contour_smoothing(image_size, scale=100, octaves=4, persistence=0.5, lacunarity=2.0, threshold=0.0, sigma=2.0, seed=123):
    """
    Generate a binary mask with smooth contours using Perlin noise.

    Parameters:
        image_size (tuple): Size of the image (height, width).
        scale (float): Scale of the noise (higher values zoom out the noise pattern).
        octaves (int): Number of noise layers blended together for detail.
        persistence (float): Controls amplitude of octaves (higher = more detail).
        lacunarity (float): Controls frequency of octaves (higher = more detail).
        threshold (float): Threshold to create a binary mask.
        sigma (float): Standard deviation for Gaussian smoothing.
        seed (int): seed controlling perlin noise generation, for deterministically generate it.

    Returns:
        np.ndarray: Binary mask (0 or 1) with smooth contours.
    """
    height, width = image_size
    noise_grid = np.zeros((height, width))
    
    # Generate Perlin noise
    for y in range(height):
        for x in range(width):
            noise_value = pnoise2(x / scale, 
                                  y / scale, 
                                  octaves=octaves, 
                                  persistence=persistence, 
                                  lacunarity=lacunarity, 
                                  repeatx=width, 
                                  repeaty=height, 
                                  base=seed)  # Seed for reproducibility
            noise_grid[y, x] = noise_value
    
    # Normalize noise values to range 0-1
    noise_grid = (noise_grid - noise_grid.min()) / (noise_grid.max() - noise_grid.min())
    
    # Apply Gaussian smoothing to the noise grid
    smoothed_noise = gaussian_filter(noise_grid, sigma=sigma)
    
    # Apply threshold to create a binary mask
    binary_mask = (smoothed_noise > threshold).astype(np.uint8)
    
    return binary_mask
