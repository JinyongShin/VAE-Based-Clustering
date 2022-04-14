#!/usr/bin/env python

import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i','--infile',action='store',type=str,required=True,help='input directory')
args =parser.parse_args()

df = pd.read_csv(args.infile + '/loss.csv')

plt.plot(df['train_loss'][1:-1],label='Train')
plt.plot(df['val_loss'][1:-1],label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.ylim(80,150)
plt.grid()
plt.legend()
plt.savefig(args.infile + '/LossCurve.png')

