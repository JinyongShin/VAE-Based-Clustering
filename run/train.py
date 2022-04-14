#!/usr/bin/env python

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
import torchvision.utils as vutils

import argparse
import sys, os
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--infile',action='store',type=str,required=True,help='input data file')
parser.add_argument('--epoch', action='store', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch', action='store', type=int, default=100, help='Batch size')
parser.add_argument('--ldim', action='store', type=int, default=10, help='latent dim')
parser.add_argument('--nClusters',action='store',type=int,default=10,help='number of clusters')
parser.add_argument('--pre',action='store',type=int,default=10,help='GMM Pre-training Epoch')
parser.add_argument('-o', '--outdir', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--shuffle', action='store', type=bool, default=True, help='Shuffle batches for each epochs')
parser.add_argument('--model', action='store', choices=('model_v1','model_v2','mlp','conv_vade','mlp_vade'), default='default', help='choice of model')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')

args =parser.parse_args()
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)
if not os.path.exists(args.outdir): os.makedirs(args.outdir)

#Build Model
sys.path.append("../model")
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

latent_dim = args.ldim
nclusters = args.nClusters

if 'vade' in args.model:
	net=MyModel.MyModel(latent_dim=latent_dim,nClusters=nclusters)
	pretrain=True
else:
	net=MyModel.MyModel(latent_dim)
	pretrain=False

model = net.cuda()
print(model)

#RE(Reconstruction Error) + KL-D(Kullback-Leibler Divergence) Loss summed over all elements and batch
#def loss_function(recon_x,x,mu,log_var):
#	BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(-1,784), x.view(-1,784), size_average=False)
#	#Appendix B from VAE paper : https://arxiv.org/abs/1312.6114
#	KLD = -0.5 * torch.sum(1+log_var - mu.pow(2) - log_var.exp())
#	return BCE + KLD 
##BCE+KLD vs BCE+3*KLD ? ???


def train(epoch):
	model.train()
	train_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):
		data = Variable(data)
		if torch.cuda.is_available():
			data = data.cuda()
		
		optimizer.zero_grad()

		recon_batch, mu, logvar = model(data)

		loss = model.loss_function(recon_batch, data, mu, logvar)
		loss.backward()
		train_loss += loss.data
		#print(loss.data)
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.data / len(data)))
	print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

	if epoch % 5 == 0:
		torch.save(model.state_dict(),args.outdir+'/model_%d.pth' % (epoch))
	return (train_loss/len(train_loader.dataset)).cpu()

def test(epoch):
	model.eval()
	test_loss = 0
	for i, (data, _) in enumerate(test_loader):
		if torch.cuda.is_available():
			data = data.cuda()
		data = Variable(data, volatile=True)
		recon_batch, mu, logvar = model(data)
		test_loss += model.loss_function(recon_batch, data, mu, logvar).data
		if i == 0:
			n = min(data.size(0), 16)
			comparison = torch.cat([data[:n],
								  recon_batch.view(batch_size, 1, 28, 28)[:n]])
	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss.cpu()

epochs = args.epoch
batch_size = args.batch

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available else {}

optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
		#print("More than 2 gropus in File")
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
length = [int(len(ds)*0.7),int(len(ds)*0.2)]
length.append(len(ds)-sum(length))

trnSet,valSet,tstSet=torch.utils.data.random_split(ds,length)

#train Loader
train_loader = DataLoader(trnSet, batch_size=args.batch, shuffle=args.shuffle, **kwargs)
#test Loader
test_loader = DataLoader(valSet, batch_size=args.batch, shuffle=False, **kwargs)

if pretrain == True:
	if not os.path.exists(args.outdir + '/pretrained.pth'):
		model.pre_train(train_loader,args.outdir,args.pre)
	else:
		model.load_state_dict(torch.load(args.outdir+'/pretrained.pth'))

train_loss_arr = []
test_loss_arr = []
for epoch in range(1,epochs+1):
	train_loss_arr.append(train(epoch))
	test_loss_arr.append(test(epoch))

import pandas as pd

df_train = pd.DataFrame(data=train_loss_arr, columns=['train_loss'])
df_test = pd.DataFrame(data=test_loss_arr, columns=['val_loss'])

df = pd.merge(df_train,df_test,left_index=True,right_index=True)
df.to_csv(args.outdir+'/loss.csv')

#import matplotlib.pyplot as plt
#
#plt.plot(train_loss_arr,label='Loss(train)')
#plt.plot(test_loss_arr,label='Loss(test)')
#plt.legend(loc='upper right')
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.ylim(120,150)
#plt.grid()
#plt.savefig('LossCurve_model2.png')
#plt.close()

