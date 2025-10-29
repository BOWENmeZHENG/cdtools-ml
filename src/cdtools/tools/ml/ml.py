import torch as t

__all__ = ['denoise_exit_wave']


def denoise_exit_wave(wavefields, amplitude_model, phase_model):
    # Data shape (batch_size, N_modes, H, W)
    def denoise(wave, amplitude_model, phase_model):
        wave = wave.unsqueeze(2)  # Shape: (batch_size, N_modes, 1, H, W) - add channel dim
        amplitude = t.abs(wave)  # Shape: (batch_size, N_modes, 1, H, W) - add channel dim
        phase = t.angle(wave)    # Shape: (batch_size, N_modes, 1, H, W) - add channel dim
        if amplitude_model is not None:
            amplitude_model.eval()
            amplitude = amplitude.reshape(-1, 1, wave.shape[3], wave.shape[4])  # Flatten batch and mode dims
            amplitude_denoised = amplitude_model(amplitude)
            amplitude_denoised = amplitude_denoised.reshape(wave.shape[0], wave.shape[1], 1, wave.shape[3], wave.shape[4])  # Restore original shape
            amplitude_denoised = amplitude_denoised.squeeze(2)  # Back to (batch_size, N_modes, H, W)
        else:
            amplitude_denoised = amplitude.squeeze(2) # Back to (batch_size, N_modes, H, W)

        if phase_model is not None:
            phase_model.eval()
            phase = phase.reshape(-1, 1, wave.shape[3], wave.shape[4])  # Flatten batch and mode dims
            phase_denoised = phase_model(phase)
            phase_denoised = phase_denoised.reshape(wave.shape[0], wave.shape[1], 1, wave.shape[3], wave.shape[4])  # Restore original shape
            phase_denoised = phase_denoised.squeeze(2)  # Back to (batch_size, N_modes, H, W)
        else:
            phase_denoised = phase.squeeze(2) # Back to (batch_size, N_modes, H, W)

        wave_denoised = amplitude_denoised * t.exp(1j * phase_denoised)

        return wave_denoised

    exit_wave_denoised = denoise(wavefields, amplitude_model, phase_model)
    return exit_wave_denoised