#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','--indir',action='store',type=str,required=True,help='input dir')
args =parser.parse_args()

df = pd.read_csv(args.indir + '/TestSetErrorInfo.csv')

#hist_la = []
#hist_re = []
#hist_kld = []
#
#for i in range(0,len(np.unique(df['Label'].to_numpy()))):
#	hist_la.append(i)
#	hist_re.append(df['RE'][df['Label']==i].to_numpy())
#	hist_kld.append(df['KLD'][df['Label']==i].to_numpy())
#
#plt.hist(hist_re,label=hist_la,stacked=True)
#plt.legend()
#plt.yscale('log')
#plt.show()
#plt.close()
#
#plt.hist(hist_kld,label=hist_la,stacked=True)
#plt.show()
#plt.close()

n_la=[]
n_re=[]
n_kld=[]

an_la=['A1','A2','A3']
an_re=[]
an_kld=[]

for i in range(0,len(np.unique(df['Label'].to_numpy()))):
	if i <= 9:
		n_la.append(str(i))
		n_re.append(df['RE'][df['Label']==i].to_numpy())
		n_kld.append(df['KLD'][df['Label']==i].to_numpy())

	else:
		an_re.append(df['RE'][df['Label']==i].to_numpy())
		an_kld.append(df['KLD'][df['Label']==i].to_numpy())

c_lst = [plt.cm.Set3(a) for a in np.linspace(0.0, 1.0, len(n_la))]

#print(n_re)
#print(n_kld)

an_re_c = np.concatenate(an_re)

plt.hist(n_re,label=n_la,color=c_lst,range=(0,500),bins=10,stacked=True)
plt.hist(an_re_c,label='Anomaly',histtype='step',color='r',linewidth=2,range=(0,500),bins=10)
#plt.yscale('log')
plt.legend()
plt.xlabel('Reconstruction Error')
plt.ylabel('Number of events / bin')
plt.show()
#plt.savefig(args.indir + '/stacked_re.png')
plt.close()

#an_kld_c = np.concatenate(an_kld)
#plt.hist(n_kld,label=n_la,color=c_lst,range=(0,40),bins=8,stacked=True)
#plt.hist(an_kld_c,label='Anomaly',histtype='step',color='r',linewidth=2,range=(0,40),bins=8)
#plt.yscale('log')
#plt.legend()
#plt.xlabel('KL-Divergence')
#plt.ylabel('Number of events / bin')
##plt.show()
#plt.savefig(args.indir + '/stacked_kld2.png')
#plt.close()
#
#n_re = np.concatenate(n_re)
#n_kld = np.concatenate(n_kld)
#an_re = np.concatenate(an_re)
#an_kld = np.concatenate(an_kld)

#plt.hist(n_kld,label='normal',color='royalblue',range=(0,500),bins=10)
#plt.hist(an_kld,label='Anormaly',histtype='step',color='r',linewidth=2,range=(0,500),bins=10)
#plt.legend()
#plt.yscale('log')
##plt.title('KL-Divergence')
#plt.xlabel('KL-Divergence')
#plt.ylabel('Number of events / bin')
#plt.show()
##plt.savefig(args.indir + '/kld_hist2.png')
#plt.close()

#plt.hist(n_re,label='normal',color='royalblue',range=(0,500),bins=10)
#plt.hist(an_re,label='Anomaly',histtype='step',color='r',linewidth=2,range=(0,500),bins=10)
#plt.legend()
#plt.yscale('log')
##plt.title('Reconstruction Error')
#plt.xlabel('Reconstruction Error')
#plt.ylabel('Number of events / bin')
##plt.show()
#plt.savefig(args.indir + '/re_hist.png')
#plt.close()
