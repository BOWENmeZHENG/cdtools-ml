import os
import random
import numpy as np
import torch
from dlsia.core.networks import tunet
from torch import nn
import matplotlib.pyplot as plt
import argparse
from dataclasses import MISSING, fields

__all__ = ['seed_everything', 'create_parser', 'parse_arguments', 'TUNetModel', 'denoise', 'visualize', 'scatter2D']

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
    def __init__(self, image_shape):
        super().__init__()
        self.model = tunet.TUNet(
            image_shape=image_shape,
            in_channels=1,
            out_channels=1,
            depth=3,
            base_channels=16,
            growth_rate=2,
            hidden_rate=1
        )
        
    def forward(self, x):
        output = self.model(x)
        return output

# def load_model(model_config, model_path, freeze=False):
#     if t.cuda.is_available():
#         device = t.device("cuda")
#     else:
#         device = t.device("cpu")

#     model = TUNetModel(model_config)
#     checkpoint = t.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint)
#     model.to(device)
#     model.eval() 

#     for param in model.parameters():
#         param.requires_grad = not freeze
#     return model

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

