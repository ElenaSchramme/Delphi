import os
import pickle
import torch
import torch.nn.functional as F
from model import DelphiConfig, Delphi
from tqdm import tqdm
import pandas as pd
import numpy as np
import textwrap
from utils import get_batch, get_p2i

dataset_subset_size = 128

out_dir = 'Delphi-2M'
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype ='float32' #'bfloat16' # 'float32' or 'bfloat16' or 'float16'
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = {'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if torch.cuda.is_available():
    checkpoint = torch.load(ckpt_path, map_location=device)
else:
    checkpoint = torch.load(ckpt_path, map_location='cpu')
conf = DelphiConfig(**checkpoint['model_args'])
model = Delphi(conf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)

checkpoint['model_args']

train = np.fromfile('data/ukb_simulated_data/train.bin', dtype=np.uint32).reshape(-1,3)
val = np.fromfile('data/ukb_simulated_data/val.bin', dtype=np.uint32).reshape(-1,3)

train_p2i = get_p2i(train)
val_p2i = get_p2i(val)

w = np.where(np.isin(sorted(list(set(train[:,0]))), train[train[:,2]==1268,0]))[0]
v = np.where(~np.isin(sorted(list(set(train[:,0]))), train[train[:,2]==1268,0]))[0]

d = get_batch(range(0,val_p2i.shape[0]-1,1), val, val_p2i,  
              select='smart_random', block_size=64, 
              device=device, padding='random')

age = 60
n_samples = 1024 * 8

d0 = torch.zeros((n_samples, 48), dtype=torch.int)
d1 = torch.zeros((n_samples, 48)) - 10000.

w = np.where((d[1].cpu().detach().numpy() <= age * 365.25) * (d[3].cpu().detach().numpy() >= age * 365.25))
u = np.unique(w[0])

d0 = d[0][u[:n_samples]].clone().detach()
d1 = d[1][u[:n_samples]].clone().detach()

d0[d1>age*365.25] = 0
d1[d1>age*365.25] = -10000.

if age > 0:
    d0 = F.pad(d0, (0,1), 'constant', 1)
    d1 = F.pad(d1, (0,1), 'constant', age*365.25)

o = d1.argsort(1)
d0 = d0.gather(1, o)[:,-48:]
d1 = d1.gather(1, o)[:,-48:]

batch_size = 128
oo = []
model.to(device)
print("Starting the generation process...")
with torch.no_grad():
    for dd in tqdm(zip(*map(lambda x: torch.split(x, batch_size), (d0,d1))), total=len(d0)//batch_size + 1):
        mm = model.generate(dd[0].to(device), dd[1].to(device), max_age=80*365.25, no_repeat=True, use_kvcache=True)
        print(f"Processed batch of size {len(dd[0])}")
        oo += [(mm[0],mm[1])]