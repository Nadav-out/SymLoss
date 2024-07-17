import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
import einops

import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import sys, os, time, datetime, json, argparse
from symm_loss_defs import *



#parse cmd arguments

parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jobid", help="job id", dest="jobid")
parser.add_argument("-c", "--jsonfile", help="json file", dest="jsonfile")
parser.add_argument("-d", "--defaultjsonfile", help="default json file", default ="n", dest="defaultjsonfile")
parser.add_argument("-o", "--outdir", help="output directory", default = "./storage",dest="outdir")
parser.add_argument("--plotsoutdir", help="plots directory", default = "./plots",dest="plotsoutdir")
parser.add_argument("-s", "--savenet", help="save net in pkl?", default = "y",dest="savenet")
parser.add_argument("-p", "--saveplots", help="save plots?", default = "n",dest="saveplots")
parser.add_argument("--seeddata",help = "set seed for data generation", default = "", dest = "seeddata")
parser.add_argument("--seedtrain",help = "set seed for trainning", default = "", dest = "seedtrain")

args = parser.parse_args()
jsonfile = args.jsonfile
defaultjsonfile = args.defaultjsonfile
savenet = args.savenet
saveplots = args.saveplots
outdir = args.outdir
plotsoutdir = args.plotsoutdir
seed_data = args.seeddata
seed_train = args.seedtrain

#get argumnets from json file
with open(jsonfile, 'r') as js:
    config = json.load(js)

if defaultjsonfile!="n":
    with open(defaultjsonfile, 'r') as djs:
        config_default = json.load(djs)
    for d_key in config_default.keys():
        if not (d_key in config.keys()):
            config[d_key] = config_default[d_key]
        

symm_net = symm_net_train()

if args.seeddata.isnumeric():
    seed_data = int(seed_data)
else:
    seed_data = int(torch.round(torch.rand(1)*10000))
if args.seedtrain.isnumeric():
    seed_train = int(seed_train)
else:
    seed_train = int(torch.round(torch.rand(1)*10000))
print(f"data seed: {seed_data}, train seed: {seed_train}")


# symm_net.prepare_dataset(seed = seed_data, N = 10000)
# symm_net.set_model(init = "rand",equiv="False",rand="False", hidden_size =15, n_hidden_layers = 7)
# symm_net.run_training(train_loader = symm_net.train_loader,nepochs = 5000,lam_vec = [0.1,0.01,0.0],seed = seed_train, lr = 5e-3)

broken_symm = config["broken_symm"]
spurions = config["spurions"]
if broken_symm == "True" or broken_symm == True:
    print("broken symmetry")
    print(f"spurions:{spurions}")
    symm_net.prepare_dataset(seed = seed_data, N = config["N"], batch_size = config["batch_size"], broken_symm = config["broken_symm"], spurions = config["spurions"], true_func = Lorentz_myfun_broken)
else:
    symm_net.prepare_dataset(seed = seed_data, N = config["N"], batch_size = config["batch_size"], broken_symm = config["broken_symm"], spurions = config["spurions"])

symm_net.set_model(init = config["init"],equiv=config["equiv"],rand=config["rand"], freeze =config["freeze"],skip = config["skip"],hidden_size =config["hidden_size"], n_hidden_layers = config["n_hidden_layers"],activation = config["activation"])
symm_net.run_training(train_loader = symm_net.train_loader,nepochs = config["nepochs"],lam_vec = config["lam_vec"],seed = seed_train, lr = config["lr"],symm_norm = config["symm_norm"])


anet =  analysis_trained()
anet.get_trained(symm_net)
anet.title()

if not (savenet=="n" or (savenet==False) or (savenet=="False")):
    anet.save_trained(outdir = outdir)

if not (saveplots=="n" or (saveplots==False) or (saveplots=="False")):
    anet.plot_losses(save = "True",outdir = plotsoutdir)
    anet.pred_plot(save = "True",outdir = plotsoutdir)

