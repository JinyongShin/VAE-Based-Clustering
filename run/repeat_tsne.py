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
parser.add_argument('--model', action='store',choices=('model_v1','model_v2','mlp'),default='default',help='choice of model')
parser.add_argument('--ldim',action='store',type=int,default=10,help='latent dim')
parser.add_argument('-o','--outdir',action='store',type=str,required=True,help='Path to Output directory')
parser.add_argument('--device',action='store',type=int,default=0,help='device')
args=parser.parse_args()

if torch.cuda.is_available() and args.device >=0: torch.cuda.set_device(args.device)
#if not os.path.exists(args.outdir): os.makedirs(args.outdir)

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
print(args.model,"loaded.")

latent_dim = args.ldim
model = MyModel.conv_VAE(latent_dim)
model.load_state_dict(torch.load(args.trained,map_location='cpu'))
model.cuda()
print(args.trained,' state dict loaded')

length = [int(len(ds)*0.7),int(len(ds)*0.2)]
length.append(len(ds)-sum(length))

trnSet,valSet,tstSet=torch.utils.data.random_split(ds,length)

test_loader = DataLoader(tstSet,batch_size=length[-1],shuffle=False)

test, test_la = iter(test_loader).next()
test = test.clone().detach().cuda()

if args.model =='mlp':
	test=test.view(-1,784)

enc1,enc2 = model.encode(test)

z = model.reparametrize(enc1,enc2)

z_test = z.cpu()

z_test=z_test.detach().numpy()

import matplotlib.pyplot as plt

n_components = 2
tsne = TSNE(n_components=n_components)
tsneArr = tsne.fit_transform(z_test)

c_lst = [plt.cm.nipy_spectral(a) for a in np.linspace(0.0, 1.0, len(np.unique(test_la)))]

plt.figure(figsize=(10,10))
for i in range(0,len(np.unique(test_la))):
	print('Class ',i,'x :',tsneArr[test_la==i,0])
	print('Class ',i,'y :',tsneArr[test_la==i,1])
	plt.scatter(tsneArr[test_la==i,0],tsneArr[test_la==i,1],label=i,color=c_lst[i])
plt.legend(loc='best')
plt.title('Manifold 2D')
#plt.savefig(args.outdir + '/manifold2D.png')
plt.savefig(args.outdir)
