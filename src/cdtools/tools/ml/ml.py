import torch as t

__all__ = ['denoise_obj']


def denoise_obj(Object, amplitude_model, phase_model):
    # Data shape (H_obj, W_obj)
    def denoise(obj, amplitude_model, phase_model):
        obj = obj.unsqueeze(0)  # Shape: (1, H_obj, W_obj) - add channel dim
        amplitude = t.abs(obj)  # Shape: (1, H_obj, W_obj) - add channel dim
        phase = t.angle(obj)    # Shape: (1, H_obj, W_obj) - add channel dim
        if amplitude_model is not None:
            amplitude_model.eval()
            amplitude_denoised = amplitude_model(amplitude)
            amplitude_denoised = amplitude_denoised.squeeze(0)  # Back to (H_obj, W_obj)
        else:
            amplitude_denoised = amplitude.squeeze(0) # Back to (H_obj, W_obj)

        if phase_model is not None:
            phase_model.eval()
            phase_denoised = phase_model(phase)
            phase_denoised = phase_denoised.squeeze(0)  # Back to (H_obj, W_obj)
        else:
            phase_denoised = phase.squeeze(0) # Back to (H_obj, W_obj)

        wave_denoised = amplitude_denoised * t.exp(1j * phase_denoised)

        return wave_denoised

    Object_denoised = denoise(Object, amplitude_model, phase_model)
    return Object_denoised