import os
import random
import numpy as np
import torch
from dlsia.core.networks import tunet
from torch import nn
import matplotlib.pyplot as plt
import argparse
from dataclasses import MISSING, fields

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def create_parser(cls):
    """Dynamically create an ArgumentParser from a dataclass."""
    parser = argparse.ArgumentParser(description="Auto-generate parser from dataclass")

    for f in fields(cls):
        arg_name = f"--{f.name}"  # Always treat fields as optional

        kwargs = {}
        # Ensure default values are correctly handled
        if f.default is not MISSING:
            kwargs["default"] = f.default
        elif f.default_factory is not MISSING:
            kwargs["default"] = f.default_factory()
        else:
            kwargs["required"] = True  # Only mark required if no default exists

        # Handle booleans separately with store_true/store_false
        if f.type is bool:
            if f.default is False:
                kwargs.pop("default", None)
                kwargs["action"] = "store_true"
            else:
                kwargs.pop("default", None)
                kwargs["action"] = "store_false"
        else:
            kwargs["type"] = f.type

        parser.add_argument(arg_name, **kwargs)

    return parser


def parse_arguments(Config):
    """Parse command-line arguments and initialize configuration."""
    parser = create_parser(Config)
    args = parser.parse_args()
    return Config(**vars(args))


class TUNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_shape = (config.input_size, config.input_size)
        self.in_channels = 1
        self.out_channels = 1
        self.depth = config.depth
        self.base_channels = config.base_channels
        self.growth_rate = config.growth_rate
        self.hidden_rate = config.hidden_rate
        self.model = tunet.TUNet(
            image_shape=self.image_shape,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            depth=self.depth,
            base_channels=self.base_channels,
            growth_rate=self.growth_rate,
            hidden_rate=self.hidden_rate
        )
        
    def forward(self, x):
        output = self.model(x)
        return output


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def denoise(wave, amplitude_model, phase_model):
    with torch.no_grad():
        amplitude = torch.abs(wave).unsqueeze(0).unsqueeze(0)
        phase = torch.angle(wave).unsqueeze(0).unsqueeze(0)

        if amplitude_model is not None:
            amplitude_denoised = amplitude_model(amplitude).squeeze(0).squeeze(0)
        else:
            amplitude_denoised = amplitude.squeeze(0).squeeze(0)

        if phase_model is not None:
            phase_denoised = phase_model(phase).squeeze(0).squeeze(0)
        else:
            phase_denoised = phase.squeeze(0).squeeze(0)

        wave_denoised = amplitude_denoised * torch.exp(1j * phase_denoised)

    return wave_denoised

def visualize(data, title=''):
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    if np.iscomplexobj(data):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        plt.suptitle(title)
        
        # Calculate 90% range for amplitude
        amp_data = np.abs(data)
        amp_vmin, amp_vmax = np.percentile(amp_data, [0.5, 99.5])
        
        # Calculate 90% range for phase
        phase_data = np.angle(data)
        phase_vmin, phase_vmax = np.percentile(phase_data, [0.5, 99.5])
        
        # Plot amplitude
        im1 = ax1.imshow(amp_data, cmap='viridis', vmin=amp_vmin, vmax=amp_vmax)
        ax1.set_title('Amplitude')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1)

        # Plot phase
        im2 = ax2.imshow(phase_data, cmap='twilight', vmin=phase_vmin, vmax=phase_vmax)
        ax2.set_title('Phase')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()
    else:
        # Fallback for real data
        data_vmin, data_vmax = np.percentile(data, [0, 100])
        plt.figure(figsize=(10, 8))
        plt.title(title)
        plt.imshow(data, cmap='viridis', vmin=data_vmin, vmax=data_vmax)
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

def scatter2D(data):
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=20)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Scatter Plot of Positions')
    plt.grid(True, alpha=0.3)
    plt.show()

