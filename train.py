import numpy as np
import torch
from utils import TUNetModel, seed_everything, visualize
import csv
import os
import time
from datetime import datetime

SEED = 43
DATA = 'combined_exit_waves'
MODEL_TYPE = 'phase'  # 'amp' or 'phase'
EPOCHS = 50
BATCHSIZE = 16
LR = 1e-3
DEPTH = 3
BASE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_complex = np.load(f'train_data/{DATA}.npy')
print(f"Data shape: {data_complex.shape}")
data = np.abs(data_complex) if MODEL_TYPE == 'amp' else np.angle(data_complex)
print(f"Data shape: {data.shape}")
data_tensor = torch.from_numpy(data).unsqueeze(1).to(DEVICE)  # shape (N, 1, H, W)

# Model config
class ModelConfig:
	input_size = 512
	depth = DEPTH
	base_channels = BASE
	growth_rate = 2
	hidden_rate = 1

seed_everything(SEED)

config = ModelConfig()
model = TUNetModel(config).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

timestamp = datetime.now().strftime('%m%d_%H%M')
experiment_name = f'{timestamp}_{MODEL_TYPE}_{DATA}_d{DEPTH}_bc{BASE}_bs{BATCHSIZE}_lr{LR}_s{SEED}'
experiment_dir = os.path.join('models', experiment_name)
os.makedirs(experiment_dir, exist_ok=True)
print(f"Experiment directory: {experiment_dir}")

# Save experiment configuration
config_filename = os.path.join(experiment_dir, 'config.txt')
if not os.path.exists(config_filename):
	with open(config_filename, 'w') as config_file:
		config_file.write(f"Experiment Configuration\n")
		config_file.write(f"========================\n")
		config_file.write(f"Name: {DATA}\n")
		config_file.write(f"Model Type: {MODEL_TYPE}\n")
		config_file.write(f"Depth: {DEPTH}\n")
		config_file.write(f"Base Channels: {BASE}\n")
		config_file.write(f"Batch Size: {BATCHSIZE}\n")
		config_file.write(f"Learning Rate: {LR}\n")
		config_file.write(f"Epochs: {EPOCHS}\n")
		config_file.write(f"Seed: {SEED}\n")
		config_file.write(f"Device: {DEVICE}\n")
		config_file.write(f"Total Parameters: {total_params:,}\n")

# Save epoch loss to CSV
csv_filename = os.path.join(experiment_dir, f'training_loss.csv')

# Write header if file doesn't exist
if not os.path.exists(csv_filename):
	with open(csv_filename, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['epoch', 'loss'])

N = data_tensor.shape[0]

# Training loop
start_time = time.time()
for epoch in range(EPOCHS):
	model.train()
	perm = torch.randperm(N)
	epoch_loss = 0.0
	for i in range(0, N, BATCHSIZE):
		idx = perm[i:i+BATCHSIZE]
		noisy = data_tensor[idx]
		clean = data_tensor[idx]  
		output = model(noisy)
		loss = criterion(output, clean)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item() * noisy.size(0)
	epoch_loss /= N
	
	# Append epoch loss
	with open(csv_filename, 'a', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow([epoch + 1, epoch_loss])
	print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.6f}")
	
	# Save the trained model
	if epoch >= 10 and (epoch + 1) % 2 == 0:  
		saved_model_path = os.path.join(experiment_dir, f'model_epoch_{epoch+1}.pth')
		torch.save(model.state_dict(), saved_model_path)
		print(f"Model saved as {saved_model_path}")

end_time = time.time()
print(f"Total time for {EPOCHS} epochs: {(end_time - start_time) / 60:.2f} minutes")