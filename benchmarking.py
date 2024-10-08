import os
import time
import torch
import torch.nn.functional as F
from model import DelphiConfig, Delphi
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import get_batch, get_p2i

# Configuration
out_dir = 'Delphi-2M'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'
seed = 1337
vocab_size = 1269
batch_size = 128
iterations = 1
max_age = 85 * 365.25

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = {'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
conf = DelphiConfig(**checkpoint['model_args'])
model = Delphi(conf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)

# Function to generate synthetic input with padding and end tokens
def generate_synthetic_input(seq_length, batch_size, vocab_size, max_age):
    # Token-Daten erzeugen, initialisiere mit 0 (Padding)
    d0 = torch.zeros(batch_size, seq_length, dtype=torch.long).to(device)
    
    # Altersdaten erzeugen
    d1 = torch.rand(batch_size, seq_length).to(device) * max_age

    # Beispielhafter Token-Bereich für Tokens außer dem Ende-Token
    non_end_token_limit = vocab_size - 2  # Reserviere Platz für Token 0 und 1
    
    # Zufällige Tokens zwischen 0 und (vocab_size - 2) außer dem End-Token (1)
    random_tokens = torch.randint(2, non_end_token_limit, (batch_size, seq_length - 1), dtype=torch.long).to(device)
    
    # In die Sequenzen einfügen, End-Token (1) am Ende jeder Sequenz
    d0[:, :-1] = random_tokens  # Fülle die Sequenzen mit zufälligen Tokens
    d0[:, -1] = 1  # End-Token am Ende jeder Sequenz
    
    # Sortiere `d1` (Alterswerte) aufsteigend und wende die gleiche Sortierung auf `d0` an
    o = d1.argsort(1)
    d0 = d0.gather(1, o)
    d1 = d1.gather(1, o)
    
    return d0, d1

# Benchmarking function
def benchmark(model, start_length, seq_length, use_kvcache, iterations):
    token_data, age_data = generate_synthetic_input(start_length, batch_size, vocab_size, max_age=max_age) 
    times = []
    for _ in range(iterations):
        start_time = time.time()
        with torch.no_grad():
            _,_,_,_,times = model.generate(token_data, age_data, max_new_tokens = seq_length, max_age=max_age, no_repeat=False, use_kvcache=use_kvcache)
        end_time = time.time()
    return times

# Define sequence lengths
seq_length = 50
start_length = 5

# Run benchmarks
results_naive = []
results_cache = []

print(f"Benchmarking for sequence length: {seq_length}")
naive_times = benchmark(model, start_length, seq_length, use_kvcache=False, iterations=iterations)
cache_times = benchmark(model, start_length, seq_length, use_kvcache=True, iterations=iterations)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot([i for i in range(seq_length) if i % 20 == 0], naive_times, label='Naive (No Cache)', marker='o')
plt.plot([i for i in range(seq_length) if i % 20 == 0], cache_times, label='With Cache', marker='o')
plt.xlabel('Sequence Length')
plt.ylabel('Time to generate the next token (seconds)')
plt.title('Benchmarking Sampling Speed with and without KV Cache')
plt.legend()
plt.grid(True)
# Save the plot to a file
plt.savefig('benchmarking_results.png')