#!/usr/bin/env python
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
import torchvision.utils as vutils
import h5py
import sys, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trained',action='store',type=str,required=True,help='pth file path')
parser.add_argument('-i','--infile',action='store',type=str,required=True,help='h5py Data file')
parser.add_argument('--model', action='store',choices=('model_v1','model_v2','mlp','conv_vade','mlp_vade'),default='default',help='choice of model')
parser.add_argument('--ldim',action='store',type=int,default=10,help='latent dim')
parser.add_argument('--nClusters',action='store',type=int,default=10,help='nclusters')
parser.add_argument('-o','--outdir',action='store',type=str,required=True,help='Path to Output directory')
parser.add_argument('--device',action='store',type=int,default=0,help='device')
args=parser.parse_args()

if torch.cuda.is_available() and args.device >=0: torch.cuda.set_device(args.device)
if not os.path.exists(args.outdir): os.makedirs(args.outdir)

file_list = args.infile.split(',')

imageList=[]
labelList=[]
for file_path in file_list:
	print('Loading file'+file_path)
	dataset = h5py.File(file_path,'r',libver='latest',swmr=True)
	FimageList=[]
	FlabelList=[]
	for gName,group in dataset.items():
		for dName,data in group.items():
			if dName == 'images':
				FimageList.append(data)
			elif dName == 'labels':
				FlabelList.append(data)

	if len(FimageList) >= 2:
		print("More than 2 gropus in File")
		image_concat = []
		for i in range(0,len(FimageList)):
			image_concat.append(FimageList[i][:])
		imageList.append(np.concatenate(image_concat))
		label_concat = []
		for i in range(0,len(FlabelList)):
			label_concat.append(FlabelList[i][:])
		labelList.append(np.concatenate(label_concat))
	else:
		imageList.append(FimageList[0][:])
		labelList.append(FlabelList[0][:])
imageList = np.concatenate(imageList)
labelList = np.concatenate(labelList)
print('input image shape : ',imageList.shape)
print('input label shape : ',labelList.shape)
ds = TensorDataset(torch.tensor(imageList),torch.tensor(labelList))

sys.path.append('../model')
if args.model == 'model_v1':
	import vae_model_v1 as MyModel
elif args.model == 'model_v2':
	import vae_model_v2 as MyModel
elif args.model == 'mlp':
	import mlp_vae_v1 as MyModel
elif args.model == 'conv_vade':
	import vade_conv as MyModel
elif args.model == 'mlp_vade':
	import vade_mlp as MyModel
print(args.model,' loaded.')

latent_dim = args.ldim
nclusters = args.nClusters

if 'vade' in args.model:
	model=MyModel.MyModel(latent_dim=latent_dim,nClusters=nclusters)
else:
	model=MyModel.MyModel(latent_dim)

model.load_state_dict(torch.load(args.trained,map_location='cpu'))
model.cuda()
print(args.trained,' state dict loaded')

length = [int(len(ds)*0.7),int(len(ds)*0.2)]
length.append(len(ds)-sum(length))

trnSet,valSet,tstSet=torch.utils.data.random_split(ds,length)

test_loader = DataLoader(tstSet,batch_size=length[-1],shuffle=False)

test, test_la = iter(test_loader).next()
test = test.clone().detach().cuda()

if 'mlp' in args.model:
	test=test.view(-1,784)

recon_e = []
kl_div = []

from tqdm import tqdm
model.eval()
print("Calculating Errors")
for i,(data,_) in enumerate(test_loader):
	#if 'mlp' in args.model:
		#data = data.view(-1,784)
	data = data.cuda()
	data = Variable(data,volatile=True)

	recon , mu , logvar = model(data)

#	if 'mlp' in args.model:
#		recon , mu , logvar = model(data.view(-1,784))
#	else:
#		recon , mu , logvar = model(data)
	#print(logvar.shape)	
	#print(logvar[0].reshape([1,10]).shape)
	#print(model.KLD(mu,logvar))
	for j in tqdm(range(0,len(data))):
		if 'vade' in args.model:
			kld = model.KLD(mu[j].reshape([1,args.ldim]),logvar[j].reshape([1,args.ldim]))
		else:
			kld = model.KLD(mu[j],logvar[j])
		
		re = model.RE(recon[j],data[j])
		#kld = model.KLD(mu[j],logvar[j])

		re = re.detach().cpu().numpy()
		kld = kld.detach().cpu().numpy()

		recon_e.append(re)
		kl_div.append(kld)
	
import pandas as pd
df_label = pd.DataFrame(data=test_la,columns=['Label'])
df_RE = pd.DataFrame(data=recon_e,columns=['RE'])
df_KLD = pd.DataFrame(data=kl_div,columns=['KLD'])

df_err = pd.merge(df_RE,df_KLD,left_index=True,right_index=True)
df = pd.merge(df_label,df_err,left_index=True,right_index=True)

df = df.sort_values(by=['Label'],ascending=True)
df = df.reset_index(drop=True)
df.to_csv(args.outdir+'/TestSetErrorInfo.csv')
