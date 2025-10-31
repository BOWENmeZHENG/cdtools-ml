import torch
import torch.nn as nn
from dlsia.core.networks import tunet

__all__ = ['complex_to_2ch', 'ch2_to_complex', 'pad_to_multiple', 'ComplexUNet', 'denoise_obj']

def complex_to_2ch(x: torch.Tensor) -> torch.Tensor:
    """
    x: complex [1,H,W]
    returns: real [1, 2, H, W]
    """
    xr = torch.view_as_real(x)        # [1,H,W,2]
    xr = xr.permute(0,3,1,2)          # [1,2,H,W]
    return xr.contiguous()

def ch2_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    x: real [1, 2, H, W]
    returns: complex [1, H, W]
    """
    x = x.permute(0,2,3,1).contiguous()          # [1,H,W,2]
    return torch.view_as_complex(x)

def pad_to_multiple(x, multiple=16):
    H, W = x.shape[-2:]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h)), pad_h, pad_w

class ComplexUNet(nn.Module):
    """
    Wraps a real UNet to map complex [1,H,W] -> complex [1,H,W] without batches.
    """
    def __init__(self, image_shape: tuple):
        super().__init__()
        self.net = tunet.TUNet(
            image_shape=image_shape,
            in_channels=2,
            out_channels=2,
            depth=4,
            base_channels=64,
            growth_rate=2,
            hidden_rate=1
        )
    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        x_2ch = complex_to_2ch(x_complex)       # [1,2,H,W]
        y_2ch = self.net(x_2ch)                 # [1,2,H,W]
        y_cplx = ch2_to_complex(y_2ch)          # [1,H,W] complex
        return y_cplx

def denoise_obj(obj, model):
    obj = obj.unsqueeze(0)  # Shape: (1, H_obj, W_obj)
    obj_denoised = model(obj)
    return obj_denoised.squeeze(0)  # Back to (H_obj, W_obj)

# def denoise_obj(Object, amplitude_model, phase_model):
#     # Data shape (H_obj, W_obj)
#     def denoise(obj, amplitude_model, phase_model):
#         obj = obj.unsqueeze(0)  # Shape: (1, H_obj, W_obj) - add channel dim
#         amplitude = t.abs(obj)  # Shape: (1, H_obj, W_obj) - add channel dim
#         phase = t.angle(obj)    # Shape: (1, H_obj, W_obj) - add channel dim
#         if amplitude_model is not None:
#             amplitude_model.eval()
#             amplitude_denoised = amplitude_model(amplitude)
#             amplitude_denoised = amplitude_denoised.squeeze(0)  # Back to (H_obj, W_obj)
#         else:
#             amplitude_denoised = amplitude.squeeze(0) # Back to (H_obj, W_obj)

#         if phase_model is not None:
#             phase_model.eval()
#             phase_denoised = phase_model(phase)
#             phase_denoised = phase_denoised.squeeze(0)  # Back to (H_obj, W_obj)
#         else:
#             phase_denoised = phase.squeeze(0) # Back to (H_obj, W_obj)

#         wave_denoised = amplitude_denoised * t.exp(1j * phase_denoised)

#         return wave_denoised

#     Object_denoised = denoise(Object, amplitude_model, phase_model)
#     return Object_denoised