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


nc = 1
ndf = 64
ngf = 64
class MyModel(nn.Module):
	def __init__(self,latent_dim):
		super(MyModel,self).__init__()
		
		self.latent_dim = latent_dim
		
		self.mu = nn.Linear(256,latent_dim)
		self.logvar = nn.Linear(256,latent_dim)
		
		self.fc4 = nn.Linear(latent_dim, 256)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		self.encoder = nn.Sequential(
			nn.Linear(784,512),
			nn.ReLU(),
			nn.Linear(512,256),
			nn.ReLU(),
		)

		self.decoder = nn.Sequential(
			nn.Linear(256,512),
			nn.ReLU(),
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
		if torch.cuda.is_available():
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def forward(self, x):
		# print("x", x.size())
		mu, logvar = self.encode(x.view(-1,784))
		# print("mu, logvar", mu.size(), logvar.size())
		z = self.reparameterize(mu, logvar)
		# print("z", z.size())
		decoded = self.decode(z)
		# print("decoded", decoded.size())
		return decoded, mu, logvar

	def RE(self,reco_x,x):
		return torch.nn.functional.binary_cross_entropy(reco_x.view(-1,784),x.view(-1,784),size_average=False)
	def KLD(self,mu,log_var):
		return -0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp())
	def loss_function(self,reco_x,x,mu,log_var):
		return self.RE(reco_x,x) + self.KLD(mu,log_var)

