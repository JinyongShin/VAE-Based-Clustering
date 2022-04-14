#!/usr/bin/env python
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable 
import torchvision.utils as vutils
import itertools

import os
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

nc = 1
ndf = 64
ngf = 64

class MyModel(nn.Module):
	def __init__(self,latent_dim,nClusters):
		super(MyModel,self).__init__()
		
		self.latent_dim = latent_dim
		self.nClusters = nClusters
		self.pi_=nn.Parameter(torch.FloatTensor(self.nClusters,).fill_(1)/self.nClusters,requires_grad=True)
		self.mu_c=nn.Parameter(torch.FloatTensor(self.nClusters,self.latent_dim).fill_(0),requires_grad=True)
		self.log_var_c=nn.Parameter(torch.FloatTensor(self.nClusters,self.latent_dim).fill_(0),requires_grad=True)

		self.mu = nn.Linear(256,latent_dim)
		self.logvar = nn.Linear(256,latent_dim)

		self.fc4 = nn.Linear(latent_dim, 256)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		self.encoder = nn.Sequential(
			nn.Linear(784,512),
			nn.ReLU(),
			nn.BatchNorm1d(512),
			nn.Dropout(0.5),

			nn.Linear(512,256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Dropout(0.5),
		)

		self.decoder = nn.Sequential(
			nn.Linear(256,512),
			nn.ReLU(),
			nn.BatchNorm1d(512),
			nn.Dropout(0.5),

			nn.Linear(512,784),
			nn.Sigmoid(),
		)

	def encode(self, x):
		h= self.encoder(x)
		return self.mu(h), self.logvar(h)

	def decode(self, z):
		h3 = self.relu(self.fc4(z))
		return self.decoder(h3)

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = Variable(std.data.new(std.size()).normal_())
		  # num = np.array([[ 1.096506	,  0.3686553 , -0.43172026,  1.27677995,  1.26733758,
		  #		  1.30626082,  0.14179629,	0.58619505, -0.76423112,  2.67965817]], dtype=np.float32)
		  # num = np.repeat(num, mu.size()[0], axis=0)
		  # eps = Variable(torch.from_numpy(num))
		return eps.mul(std).add_(mu)
	
#	 def reparametrize(self, mu, logvar):
#		 std = logvar.mul(0.5).exp_()
#		 if torch.cuda.is_available():
#			 eps = torch.cuda.FloatTensor(std.size()).normal_()
#		 else:
#			 eps = torch.FloatTensor(std.size()).normal_()
#		 eps = Variable(eps)
#		 return eps.mul(std).add_(mu)
	
	def gaussian_pdf_log(self,x,mu,log_sigma2):
		return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))
	
	def gaussian_pdfs_log(self,x,mus,log_sigma2s):
	#def gaussian_pdfs_log(x,mus,log_sigma2s):
		G=[]
		for c in range(self.nClusters):
			G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
		return torch.cat(G,1)
	
	def forward(self, x):
		# print("x", x.size())
		mu, logvar = self.encode(x.view(-1,784))
		#decoded = self.decode(mu)
		z = self.reparameterize(mu,logvar)
		#z = torch.randn_like(mu)*torch.exp(logvar/2)+mu
		decoded = self.decode(z)
		# print("decoded", decoded.size())
		return decoded, mu, logvar

	def pre_train(self,dataloader,outdir,pre_epoch=10):
		loss=nn.MSELoss()
		optimizer = optim.Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))
		for epoch in tqdm(range(pre_epoch)):
			L=0
			for i, (data,y) in enumerate(dataloader):
				x=data.cuda()
				x = x.view(-1,784)
				z,_ = self.encode(x)
				x_ = self.decode(z)

				Loss = loss(x,x_)
				L += Loss.detach().cpu().numpy()

				optimizer.zero_grad()
				Loss.backward()
				optimizer.step()
		#print('epoch : ',epoch,'Loss : ',L)
		self.logvar.load_state_dict(self.mu.state_dict())

		Z=[]
		Y=[]
		with torch.no_grad():
			for i, (data,_) in enumerate(dataloader):
				data = data.view(-1,784).cuda()

				z1, z2 = self.encode(data)
				assert nn.functional.mse_loss(z1,z2)==0
				Z.append(z1)
				Y.append(y)

		Z=torch.cat(Z,0).detach().cpu().numpy()
		Y=torch.cat(Y,0).detach().numpy()

		gmm = GaussianMixture(n_components=self.nClusters,covariance_type='diag')

		pre = gmm.fit_predict(Z)

		self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
		self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
		self.log_var_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())

		torch.save(self.state_dict(), outdir + '/pretrained.pth')

	def RE(self,recon_x,x):
		return torch.nn.functional.binary_cross_entropy(recon_x.view(-1,784),x.view(-1,784),size_average=False)

	def KLD(self,mu,log_var):
		det=1e-10

		pi=self.pi_
		log_var_c = self.log_var_c
		mu_c = self.mu_c

		z = torch.randn_like(mu) * torch.exp(log_var/2) + mu

		yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_var_c))+det
		yita_c = yita_c/(yita_c.sum(1).view(-1,1))
		loss = 0.5*torch.mean(torch.sum(yita_c*torch.sum(log_var_c.unsqueeze(0)+
													torch.exp(log_var.unsqueeze(1)-log_var_c.unsqueeze(0))+
													(mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_var_c.unsqueeze(0)),2),1))
		loss -= torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+log_var,1))
		return loss

	def loss_function(self,recon_x,x,mu,log_var):
		return self.RE(recon_x,x)+self.KLD(mu,log_var)
