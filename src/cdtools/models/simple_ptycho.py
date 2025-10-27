import torch as t
from torch import nn
from cdtools.models import CDIModel
from cdtools import tools
from cdtools.tools import plotting as p
from dataclasses import dataclass
from dlsia.core.networks import tunet

__all__ = ['SimplePtycho']        

@dataclass
class ModelConfig:
    input_size: int = 512
    depth: int = 3
    base_channels: int = 16
    growth_rate: int = 2
    hidden_rate: int = 1

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

def load_model(model_path, freeze=False, device='cuda'):
    config = ModelConfig()
    model = TUNetModel(config)
    checkpoint = t.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval() 

    for param in model.parameters():
        param.requires_grad = not freeze
    return model

class SimplePtycho(CDIModel):
    """A simple ptychography model to demonstrate the structure of a model
    """
    def __init__(
            self,
            wavelength,
            probe_basis,
            probe_guess,
            obj_guess,
            min_translation = [0,0],
    ):

        # We initialize the superclass
        super().__init__()

        # We register all the constants, like wavelength, as buffers. This
        # lets the model hook into some nice pytorch features, like using
        # model.to, and broadcasting the model state across multiple GPUs
        self.register_buffer('wavelength', t.as_tensor(wavelength))
        self.register_buffer('min_translation', t.as_tensor(min_translation))
        self.register_buffer('probe_basis', t.as_tensor(probe_basis))

        # We cast the probe and object to 64-bit complex tensors
        probe_guess = t.as_tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.as_tensor(obj_guess, dtype=t.complex64)

        # We rescale the probe here so it learns at the same rate as the
        # object when using optimizers, like Adam, which set the stepsize
        # to a fixed maximum
        self.register_buffer('probe_norm', t.max(t.abs(probe_guess)))

        # And we store the probe and object guesses as parameters, so
        # they can get optimized by pytorch
        self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess)


    @classmethod
    def from_dataset(cls, dataset, amplitude_model_path=None, phase_model_path=None,
                     ml_epochs=[], propagation_distance=None, freeze=False):
        # We get the key geometry information from the dataset
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # Then, we generate the probe geometry
        ewg = tools.initializers.exit_wave_geometry
        probe_basis =  ewg(det_basis, det_shape, wavelength, distance)

        # Next generate the object geometry from the probe geometry and
        # the translations
        (indices, translations), patterns = dataset[:]
        pix_translations = tools.interactions.translations_to_pixel(
            probe_basis,
            translations,
        )
        obj_size, min_translation = tools.initializers.calc_object_setup(
            det_shape,
            pix_translations,
        )

        # Finally, initialize the probe and object using this information
        probe = tools.initializers.SHARP_style_probe(dataset, propagation_distance=propagation_distance)
        # probe = tools.initializers.gaussian_probe(
        #         dataset,
        #         probe_basis,
        #         det_shape, 
        #         1.0
        #     )
        obj = t.ones(obj_size).to(dtype=t.complex64)

        # Create the model instance
        model = cls(
            wavelength,
            probe_basis,
            probe,
            obj,
            min_translation=min_translation
        )
        
        # Load and store the ML models if paths are provided
        if amplitude_model_path is not None: # and phase_model_path is not None:
            model.amplitude_model = load_model(amplitude_model_path, freeze=freeze, device='cuda')
        if phase_model_path is not None:
            model.phase_model = load_model(phase_model_path, freeze=freeze, device='cuda')
        model.ml_epochs = ml_epochs
        return model



    def interaction(self, index, translations):
        
        # We map from real-space to pixel-space units
        pix_trans = tools.interactions.translations_to_pixel(
            self.probe_basis,
            translations)
        pix_trans -= self.min_translation
        
        # This function extracts the appropriate window from the object and
        # multiplies the object and probe functions
        return tools.interactions.ptycho_2D_round(
            self.probe_norm * self.probe,
            self.obj,
            pix_trans)

    # New code here
    def ml(self, wavefields, amplitude_model, phase_model):
        return tools.ml.denoise_exit_wave(wavefields, amplitude_model, phase_model)

    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)

    def measurement(self, wavefields):
        return tools.measurements.intensity(wavefields)

    def loss(self, real_data, sim_data):
        return tools.losses.amplitude_mse(real_data, sim_data)


    # This lists all the plots to display on a call to model.inspect()
    plot_list = [
        ('Probe Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe, fig=fig, basis=self.probe_basis)),
        ('Probe Phase',
         lambda self, fig: p.plot_phase(self.probe, fig=fig, basis=self.probe_basis)),
        ('Object Amplitude',
         lambda self, fig: p.plot_amplitude(self.obj, fig=fig, basis=self.probe_basis)),
        ('Object Phase',
         lambda self, fig: p.plot_phase(self.obj, fig=fig, basis=self.probe_basis))
    ]
    
    def save_results(self, dataset):
        # This will save out everything needed to recreate the object
        # in the same state, but it's not the best formatted. 
        base_results = super().save_results()

        # So we also save out the main results in a more useable format
        probe_basis = self.probe_basis.detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        wavelength = self.wavelength.cpu().numpy()

        results = {
            'probe_basis': probe_basis,
            'probe': probe,
            'obj': obj,
            'wavelength': wavelength,
        }

        return {**base_results, **results}
