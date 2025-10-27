import torch as t

__all__ = ['denoise_exit_wave']


def denoise_exit_wave(wavefields, amplitude_model, phase_model):

    def denoise(wave, amplitude_model, phase_model):
        amplitude = t.abs(wave).unsqueeze(1)  # Shape: [10, 1, 512, 512] - add channel dim
        phase = t.angle(wave).unsqueeze(1)    # Shape: [10, 1, 512, 512] - add channel dim

        if amplitude_model is not None:
            amplitude_model.eval()
            amplitude_denoised = amplitude_model(amplitude).squeeze(1)  # Back to [10, 512, 512]
        else:
            amplitude_denoised = amplitude.squeeze(1)

        if phase_model is not None:
            phase_model.eval()
            phase_denoised = phase_model(phase).squeeze(1)  # Back to [10, 512, 512]
        else:
            phase_denoised = phase.squeeze(1)

        wave_denoised = amplitude_denoised * t.exp(1j * phase_denoised)

        return wave_denoised

    exit_wave_denoised = denoise(wavefields, amplitude_model, phase_model)
    return exit_wave_denoised