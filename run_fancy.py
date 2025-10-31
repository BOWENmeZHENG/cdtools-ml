import cdtools
from cdtools.tools.misc import visualize, scatter2D, parse_arguments, seed_everything
import numpy as np
import random
import os
import torch
from torch import nn
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from dlsia.core.networks import tunet

import warnings
from dataclasses import dataclass, field
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Config:
    SEED: int = 42
    ITER_TRAIN_INIT: int = 10 #vary this
    LR: float = 0.005
    LR_ML: float = 0.001
    BS: int = 50
    ITERATIONS: int = 200
    PROP_DIST: float = 5e-6
    OVERSAMPLING: int = 1
    N_MODES : int = 2
    USE_ML: bool = False
    SCHEDULER: bool = True
    PLOT_FREQ: int = 50
    SHOW_PLOTS: bool = True
    SAVE_PLOTS: bool = False
    DATA: str = 'NS_241017025_ccdframes_30_0'

    SAVE_TRAIN_DATA: bool = False
    SAVE_EPOCHS: list = field(default_factory=lambda: [10, 50, 100, 200])
    DEVICE: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

def main():
    config = parse_arguments(Config)
    seed_everything(config.SEED)
    timestamp = datetime.now().strftime('%m%d_%H%M')

    if config.SAVE_PLOTS or config.SAVE_TRAIN_DATA:
        ml_str = f"ml_{config.ITER_TRAIN_INIT}_lr_ml_{config.LR_ML}" if config.USE_ML else "no_ml"
        results_dir = f'results/{timestamp}_{ml_str}_nm_{config.N_MODES}_lr_{config.LR}_scheduler_{config.SCHEDULER}_bs_{config.BS}_pd_{config.PROP_DIST}_s_{config.SEED}'
        os.makedirs(results_dir, exist_ok=True)
        with open(f'{results_dir}/settings.txt', 'w') as f:
            for key, value in config.__dict__.items():
                f.write(f"{key}: {value}\n")

    start_time = time.time()
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(f'real_data/{config.DATA}.cxi')

    model = cdtools.models.FancyPtychoML.from_dataset(
                dataset,
                n_modes=config.N_MODES, 
                oversampling=config.OVERSAMPLING, 
                probe_support_radius=None,
                propagation_distance=config.PROP_DIST, 
            )

    model.to(device=config.DEVICE)

    print(model.obj_size)
    dataset.get_as(device=config.DEVICE)

    ptycho_params = [model.obj, model.probe]
    ptycho_optimizer = torch.optim.Adam(ptycho_params, lr=config.LR)

    model_init = TUNetModel(image_shape=model.obj_size).to(device=config.DEVICE)
    ml_optimizer = torch.optim.Adam(model_init.parameters(), lr=config.LR_ML) if config.USE_ML else None

    losses = []

    print(f"Ptycho parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Ptycho LR: {config.LR}")
    if config.USE_ML:
        print(f"ML parameters: {sum(p.numel() for p in model_init.parameters()):,}")
        print(f"ML LR: {config.LR_ML}")

    if config.SAVE_TRAIN_DATA:
        os.makedirs('train_data', exist_ok=True)
        model.save_exit_wave_epochs = config.SAVE_EPOCHS

    for i, loss in enumerate(model.Adam_optimize(config.ITERATIONS, dataset,
                            ptycho_optimizer=ptycho_optimizer,
                            ml_optimizer=ml_optimizer,
                            batch_size=config.BS,
                            schedule=config.SCHEDULER)):
        losses.append(loss)
        if (i + 1) % config.PLOT_FREQ == 0:
            print(i + 1, loss)
            obj_cpu = model.obj.detach().cpu().numpy()
            probe_cpu = model.probe.detach().cpu().numpy()

            plt.figure(figsize=(12, 10))
            
            obj_amp_vmin, obj_amp_vmax = np.percentile(np.abs(obj_cpu), [0.5, 99.5])
            obj_phase_vmin, obj_phase_vmax = np.percentile(np.angle(obj_cpu), [0.5, 99.5])
            probe_amp_vmin, probe_amp_vmax = np.percentile(np.abs(probe_cpu), [0, 100])
            probe_phase_vmin, probe_phase_vmax = np.percentile(np.angle(probe_cpu), [0, 100])
            
            plt.subplot(2, 2, 1)
            plt.imshow(np.abs(obj_cpu), cmap='viridis', vmin=obj_amp_vmin, vmax=obj_amp_vmax)
            plt.title(f'Object Amplitude - Iteration {i + 1}')
            plt.colorbar()
            
            plt.subplot(2, 2, 2)
            plt.imshow(np.angle(obj_cpu), cmap='twilight', vmin=obj_phase_vmin, vmax=obj_phase_vmax)
            plt.title(f'Object Phase - Iteration {i + 1}')
            plt.colorbar()


            # crop center 250x350
            # h, w = obj_cpu.shape[:2]
            # ch, cw = 250, 350
            # sh = max((h - ch) // 2, 0)
            # sw = max((w - cw) // 2, 0)
            # cropped = obj_cpu[sh:sh + min(ch, h), sw:sw + min(cw, w)]

            # amp = np.abs(cropped)
            # phase = np.angle(cropped)
            # amp_vmin, amp_vmax = np.percentile(amp, [0.5, 99.5])
            # phase_vmin, phase_vmax = np.percentile(phase, [0.5, 99.5])
            # plt.subplot(2, 2, 1)
            # plt.imshow(amp, cmap='viridis', vmin=amp_vmin, vmax=amp_vmax)
            # plt.title(f'Object Amplitude - Iteration {i + 1}')
            # plt.colorbar()
            
            # plt.subplot(2, 2, 2)
            # plt.imshow(phase, cmap='twilight', vmin=phase_vmin, vmax=phase_vmax)
            # plt.title(f'Object Phase - Iteration {i + 1}')
            # plt.colorbar()

            plt.subplot(2, 2, 3)
            plt.imshow(np.abs(probe_cpu[0]), cmap='viridis', vmin=probe_amp_vmin, vmax=probe_amp_vmax)
            plt.title(f'Probe Amplitude - Iteration {i + 1}')
            plt.colorbar()

            plt.subplot(2, 2, 4)
            plt.imshow(np.angle(probe_cpu[0]), cmap='twilight', vmin=probe_phase_vmin, vmax=probe_phase_vmax)
            plt.title(f'Probe Phase - Iteration {i + 1}')
            plt.colorbar()

            plt.tight_layout()
            if config.SHOW_PLOTS:
                plt.show()
            if config.SAVE_PLOTS:
                plt.savefig(f'{results_dir}/iteration_{i + 1}.png')
                plt.close()

    # Plot and save loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log') 

    if config.SHOW_PLOTS:
        plt.show()
    if config.SAVE_PLOTS:
        plt.savefig(f'{results_dir}/loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        np.save(f'{results_dir}/losses.npy', np.array(losses))


    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Final loss: {losses[-1]:.6f}")

    if config.SAVE_TRAIN_DATA:
        saved = np.array(model.exit_wave_list)
        print(f"Saved exit waves shape: {saved.shape}")
        np.save(f'train_data/exit_waves_{config.DATA}_{config.SAVE_EPOCHS}.npy', saved)

if __name__ == "__main__":
    main()