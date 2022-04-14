#!/usr/env/bin python3

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',type=str,required=True,help='pth file path')
parser.add_argument('--model', action='store',choices=('model_v1','model_v2','mlp'),default='default',help='choice of model')
parser.add_argument('--ldim',action='store',type=int,default=10,help='latent dim')
parser.add_argument('-o','--outdir',action='store',type=str,required=True,help='Path to Output directory')
parser.add_argument('--device',action='store',type=int,default=0,help='device')
parser.add_argument('-n','--NtoShow',action='store',type=int,required=True,help='How many images do you want?')
args=parser.parse_args()

if torch.cuda.is_available() and args.device >=0: torch.cuda.set_device(args.device)
if not os.path.exists(args.outdir): os.makedirs(args.outdir)

sys.path.append('../model')
if args.model == 'model_v1':
	import vae_model_v1 as MyModel
elif args.model == 'model_v2':
	import vae_model_v2 as MyModel
elif args.model == 'mlp':
	import mlp_vae_v1 as MyModel
print(args.model,"loaded.")

latent_dim = args.ldim

model = MyModel.conv_VAE(latent_dim)
model.load_state_dict(torch.load(args.infile,map_location='cpu'))
model.cuda()
print(args.infile,' state dict loaded')

z_input = np.random.normal(size=(args.NtoShow,latent_dim))
z_batch = torch.cuda.FloatTensor(z_input)
z_batch = Variable(z_batch)
vis_batch = model.decode(z_batch)

data = np.array(vis_batch.data.cpu())

from tqdm import tqdm
print("Visualising")
for i in tqdm(range(0,len(data))):
	plt.imshow(data[i,0,:,:],'gray')
	plt.savefig(args.outdir+'/'+str(i)+'.png')


