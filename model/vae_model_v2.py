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
		
		self.encoder = nn.Sequential(
			# input is (nc) x 28 x 28
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 14 x 14
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout2d(0.5),
			# state size. (ndf*2) x 7 x 7
			nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout2d(0.5),
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

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		if torch.cuda.is_available():
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def forward(self, x):
		# print("x", x.size())
		mu, logvar = self.encode(x)
		# print("mu, logvar", mu.size(), logvar.size())
		z = self.reparametrize(mu, logvar)
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

