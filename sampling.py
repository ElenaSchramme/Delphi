import os
import pickle
import torch
import torch.nn.functional
from model import DelphiConfig, Delphi
from tqdm import tqdm
import pandas as pd
import numpy as np
import textwrap
from utils import get_batch, get_p2i

import matplotlib.pyplot as plt
#config InlineBackend.figure_format='retina'

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'axes.grid': True,
                     'grid.linestyle': ':',
                     'axes.spines.bottom': False,
          'axes.spines.left': False,
          'axes.spines.right': False,
          'axes.spines.top': False})
plt.rcParams['figure.dpi'] = 72
plt.rcParams['pdf.fonttype'] = 42

#Green
light_male = '#BAEBE3'
normal_male = '#0FB8A1'
dark_male = '#00574A'


#Purple
light_female = '#DEC7FF'
normal_female = '#8520F1'
dark_female = '#7A00BF'


delphi_labels = pd.read_csv('delphi_labels_chapters_colours_icd.csv')
labels = pd.read_csv("data/ukb_simulated_data/labels.csv", header=None, sep="\t")

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

females = train[np.isin(train[:,0], train[train[:,2]==1,0])]
males = train[np.isin(train[:,0], train[train[:,2]==2,0])]
n_females = (train[:,2]==1).sum()
n_males = (train[:,2]==2).sum()

n_males = - np.cumsum(np.histogram(np.maximum(40,np.round(males[np.where(males[:-1,0]!=males[1:,0])[0],1]/365))+1, np.arange(100))[0]) #males[:,2]==males[:,2].max(),1])
n_males = n_males - n_males[-1]
n_females = - np.cumsum(np.histogram(np.maximum(40,np.round(females[np.where(females[:-1,0]!=females[1:,0])[0],1]/365)+1), np.arange(100))[0]) #males[:,2]==males[:,2].max(),1])
n_females = n_females - n_females[-1]

incidence_k_g = []
for k in range(len(labels)):
    h_f,x = np.histogram(females[females[:,2]==k-1,1]/365.25, np.arange(100))
    h_m,x = np.histogram(males[males[:,2]==k-1,1]/365.25, np.arange(100))
    incidence_k_g.append([h_f/n_females,h_m/n_males])

incidence_k_g = np.array(incidence_k_g)

diseases_of_interest = [np.where(labels[0].str.startswith(x))[0][0] for x in ['A41','B01','C25','C50','G30','E10','F32','I21','J45','Death',]]

## Load large chunk of data
d = get_batch(range(dataset_subset_size), val, val_p2i,  
              select='smart_random', block_size=64, 
              device=device, padding='random')

has_gender = np.array([2 in x or 3 in x for x in d[0]])
is_male = np.array([3 in x for x in d[0]])
is_female = np.array([2 in x for x in d[0]])

w = np.zeros((d[0].shape[0], model.config.vocab_size))
for i,row in enumerate(d[0]):
    for j in row:
        w[i, int(j)]=1

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
    d0 = torch.nn.functional.pad(d0, (0,1), 'constant', 1)
    d1 = torch.nn.functional.pad(d1, (0,1), 'constant', age*365.25)

o = d1.argsort(1)
d0 = d0.gather(1, o)[:,-48:]
d1 = d1.gather(1, o)[:,-48:]

batch_size = 128
oo = []
model.to(device)
with torch.no_grad():
    for dd in tqdm(zip(*map(lambda x: torch.split(x, batch_size), (d0,d1))), total=len(d0)//batch_size + 1):
        mm = model.generate(dd[0].to(device), dd[1].to(device), max_age=80*365.25, no_repeat=True)
        oo += [(mm[0],mm[1])]