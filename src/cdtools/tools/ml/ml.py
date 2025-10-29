import torch as t

__all__ = ['denoise_probe']

def denoise_probe(probe, amplitude_model, phase_model):
    # Data shape (N_modes, H, W)
    def denoise(wave, amplitude_model, phase_model):
        wave = wave.unsqueeze(1) # Shape: (N_modes, 1, H, W)
        amplitude = t.abs(wave) # Shape: (N_modes, 1, H, W)
        phase = t.angle(wave) # Shape: (N_modes, 1, H, W)
        # print("Denoising probe with shape:", phase.shape)
        if amplitude_model is not None:
            amplitude_model.eval()
            # amplitude_input = amplitude.unsqueeze(0).unsqueeze(0)
            amplitude_denoised = amplitude_model(amplitude) # Shape: (N_modes, 1, H, W)
            # Remove batch and channel dimensions: [1, 1, H, W] -> [H, W]
            amplitude_denoised = amplitude_denoised.squeeze(1)
        else:
            amplitude_denoised = amplitude.squeeze(1)

        if phase_model is not None:
            phase_model.eval()
            # phase_input = phase.unsqueeze(0).unsqueeze(0)
            phase_denoised = phase_model(phase) # Shape: (N_modes, 1, H, W)
            # Remove batch and channel dimensions: [1, 1, H, W] -> [H, W]
            
            phase_denoised = phase_denoised.squeeze(1)
            # print("Denoised phase shape:", phase_denoised.shape)
        else:
            phase_denoised = phase.squeeze(1)

        wave_denoised = amplitude_denoised * t.exp(1j * phase_denoised)
        # print("Denoised wave shape:", wave_denoised.shape)
        return wave_denoised

    probe_denoised = denoise(probe, amplitude_model, phase_model)
    return probe_denoised