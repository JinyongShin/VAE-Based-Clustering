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
		
		self.encoder = nn.Sequential(
			# input is (nc) x 28 x 28
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 14 x 14
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 7 x 7
			nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 4 x 4
			nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
			# nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.Sigmoid()
		)
		
		self.decoder = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(		1024, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,		nc, 4, 2, 1, bias=False),
			# nn.BatchNorm2d(ngf),
			# nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			# nn.ConvTranspose2d(	 ngf,	   nc, 4, 2, 1, bias=False),
			# nn.Tanh()
			nn.Sigmoid()
			# state size. (nc) x 64 x 64
		)

		self.fc1 = nn.Linear(1024, 512)
		self.fc21 = nn.Linear(512, latent_dim)
		self.fc22 = nn.Linear(512, latent_dim)

		self.fc3 = nn.Linear(latent_dim, 512)
		self.fc4 = nn.Linear(512, 1024)

		self.lrelu = nn.LeakyReLU()
		self.relu = nn.ReLU()
		# self.sigmoid = nn.Sigmoid()
	def encode(self, x):
		conv = self.encoder(x);
		# print("encode conv", conv.size())
		h1 = self.fc1(conv.view(-1, 1024))
		# print("encode h1", h1.size())
		return self.fc21(h1), self.fc22(h1)

	def decode(self, z):
		h3 = self.relu(self.fc3(z))
		deconv_input = self.fc4(h3)
		# print("deconv_input", deconv_input.size())
		deconv_input = deconv_input.view(-1,1024,1,1)
		# print("deconv_input", deconv_input.size())
		return self.decoder(deconv_input)

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
		mu, logvar = self.encode(x)
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
				z,_ = self.encode(x)
				x_ = self.decode(z)

				Loss = loss(x,x_)
				L += Loss.detach().cpu().numpy()

				optimizer.zero_grad()
				Loss.backward()
				optimizer.step()
		#print('epoch : ',epoch,'Loss : ',L)
		self.fc22.load_state_dict(self.fc21.state_dict())

		Z=[]
		Y=[]
		with torch.no_grad():
			for i, (data,_) in enumerate(dataloader):
				data = data.cuda()

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
