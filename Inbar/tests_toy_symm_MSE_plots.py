import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
import einops

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import copy
import pickle
import os
import glob
from symm_MSE_loss_defs import *
#from top_symm_loss_defs import *
from decimal import Decimal
import glob
import matplotlib.cm as cm
import matplotlib.colors as mcolors

###### defs ########
storage_dir="/pscratch/sd/i/inbarsav/SymmLoss/storage"
# color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]
cmap = cm.get_cmap('cool')
lambdas = [0.0,0.1,1.0,10.0,100.0]
vmin=0
vmax=len(lambdas)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
colors_dict = {0.0: 'gray'}
colors_dict.update({lambdas[i]: scalar_map.to_rgba(i) for i in np.arange(len(lambdas))})
colors_dict.update({0.0: 'lightgray'})
color_vec = ["black","blue","magenta","pink"]

def set_plot_params(fontsize = 14,lable_size_major = 14,lable_size_minor = 12,legend_size = 14,font_family="serif"):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'xtick.labelsize': lable_size_major})
    plt.rcParams.update({'ytick.labelsize': lable_size_major})
    plt.rcParams.update({'legend.fontsize': legend_size})
    plt.rcParams.update({'font.family': font_family})
    # plt.rcParams.update({'axes.labelweight': 'bold'})

  

def Lorentz_myfun(input):
        m2 = torch.einsum("... i, ij, ...j -> ...",input, torch.diag(torch.tensor([1.00,-1.00,-1.00,-1.00])).to(devicef), input)
        out = m2**2+15*m2
        return out.unsqueeze(1).to(devicef)

def Lorentz_myfun_broken(input,spurions = [torch.tensor([0,0,0,0])]):
    metric_tensor = torch.diag(torch.tensor([1.00,-1.00,-1.00,-1.00])).to(devicef)
    m2 = torch.einsum("... i, ij, ...j -> ...",input, metric_tensor, input)
    breaking_scalars = [torch.einsum("... i, ij, ...j -> ...",spurion.to(devicef), metric_tensor, input) for spurion in spurions]
    coeffs = [20.9983, -23.2444, 3.0459, 12.7176, -17.4378, 1.4378, 10.1877,15.8890, -11.5178,  -4.3926]
    coeffs_2 = [-0.8431,   5.7529,  19.0048,   3.2927, -14.9460,   5.6997,  -5.9202, -10.5052, 2.6883, 16.5809]
    symm_out = m2**2+15*m2
    out = symm_out
    for i in range(len(breaking_scalars)):
        out += coeffs[i%len(coeffs)]*breaking_scalars[i]+coeffs_2[i%len(coeffs_2)]*breaking_scalars[i]**2
    return out.unsqueeze(1).to(devicef)


def pred_plot_ext(analysis,data,model = "last",transformed_spurions = "True",save = False, outdir = "./",filename = ""):
        inputs = analysis.train_data.to(devicef)
        plt.clf()
        fig = {}
        if model == "last":
            models = analysis.models
            ext = ""
        elif model== "symm":
            models = analysis.models_best_symm
            ext = "_best_symm"
        elif model == "MSE":
            models = analysis.models_best_MSE
            ext = "_best_MSE"
        elif model == "tot":
            models = analysis.models_best_tot
            ext = "_best_tot"
            
        
        
        if analysis.input_spurions == "True" or analysis.input_spurions==True:
            
            lens_spurions = [torch.numel(sp) for sp in analysis.spurions]
            len_spurions = sum(lens_spurions)
            
            truth_data = data[:,0:-len_spurions]
            
            
            if (transformed_spurions == "True" or transformed_spurions == True):
                print(f"now transformed_spurions = {transformed_spurions}")
                d = data.shape[-1]-len_spurions
                sum_length = np.concatenate((np.array([0]),np.cumsum(lens_spurions)))
                trans_spurions = [data[0,(d+sum_length[i]):(d+sum_length[i+1])] for i in range(len(sum_length)-1)]
                print(trans_spurions)
            else:
                trans_spurions = analysis.spurions
                expand_spurions = (torch.cat(trans_spurions)).expand(data.shape[0],len_spurions)
                data = torch.cat((truth_data.to(devicef),expand_spurions.to(devicef)),dim = -1)
        else:
            truth_data = data
            trans_spurions = analysis.spurions
            
            
        if analysis.broken_symm == "True" or analysis.broken_symm == True:
            truth_new = Lorentz_myfun_broken(truth_data,spurions = trans_spurions)
        else:
            truth_new = Lorentz_myfun(truth_data)
            
            
        for lam_val in analysis.models.keys():
            plt.clf()
            fig[lam_val] = plt.figure()
            plt.scatter(analysis.train_labels.cpu().squeeze(),models[lam_val](inputs).detach().cpu().squeeze(),label = rf"$\lambda$ = {lam_val} training data")
            plt.scatter(truth_new.cpu().squeeze(),models[lam_val](data).detach().cpu().squeeze(),label = rf"$\lambda$ = {lam_val} new data",color = "pink",alpha = 0.2)
            plt.scatter(analysis.train_labels.cpu().squeeze(),truth_new.cpu().squeeze(),label = rf"$\lambda$ = {lam_val} new data vs old labels",color = "purple",alpha = 0.05)

            plt.plot(truth_new.cpu().squeeze(),truth_new.cpu().squeeze(),color = "black")
            plt.legend()
            plt.xlabel("truth")
            plt.ylabel("pred")
            err = ((truth_new.cpu().squeeze()-models[lam_val](data).detach().cpu().squeeze())**2).mean()
            #err = '%.4E' % Decimal("f{err}")
            err = "{:.4e}".format(err)
            mse = ((analysis.train_labels.cpu().squeeze()-models[lam_val](inputs).detach().cpu().squeeze())**2).mean()#analysis.train_loss_lam[lam_val][-1]
            mse = "{:.4e}".format(mse)
            
            loss_full = SymmLoss(gens_list=gens_Lorentz, model = models[lam_val])
            
            symm = loss_full(inputs)#analysis.symm_loss_lam[lam_val][-1]
            symm = "{:.4e}".format(symm)
            text = f"lam = {lam_val}, var = {err} MSE = {mse}, symm = {symm}"
            
            if analysis.broken_symm == "True" or analysis.broken_symm == True:
                loss_unbroken = SymmLoss(gens_list=gens_Lorentz[[0,1,-1]], model = models[lam_val])
                symm_unbroken  = loss_unbroken(inputs)
                text = f"{text} unbroken symm = {symm_unbroken}"
                
            print(text)
            text = f"var = {err}, MSE = {mse}, symm = {symm}"
            plt.text(-9, -10,text)
            
            if save==True or save=="True":
                if filename =="":
                    file = f"plot_pred{ext}_lam_{lam_val}_{analysis.filename}"
                else:
                    file = filename
                fig[lam_val].show()
                fig[lam_val].savefig(f"{outdir}/plot_pred{ext}_lam_{lam_val}_{file}_{analysis.filename}.pdf")
            plt.show()
                #plt.close(fig[lam_val])

def performance_plot_ext(analysis,beta_range = torch.linspace(0,1,100),beta_dir = torch.tensor([1.0,0.0,0.0]),theta_range = torch.linspace(0,2*np.pi,100),theta_dir =torch.tensor([0.0,0.0,1.0]),model = "last",transformed_spurions = "True",save = False, outdir = "./plots",filename = "",relative = False):
        
    ######### initialize #################   
    trans_new_data = torch.zeros(len(beta_range))
    rot_data = torch.zeros(len(theta_range))
    data = analysis.train_data.to(devicef)
    inputs = analysis.train_data.to(devicef)
    ext = ""
    if analysis.input_spurions == "True" or analysis.input_spurions==True:
        lens_spurions = [torch.numel(sp) for sp in analysis.spurions]
        len_spurions = sum(lens_spurions)
        data = data[:,0:-len_spurions]
        if transformed_spurions == "False" or transformed_spurions == False:
            ext = "_same_frame"
        else:
            ext = "_trans_frame"

    if model == "last":
        models = analysis.models
        ext = ext
    elif model== "symm":
        models = analysis.models_best_symm
        ext = ext+"_best_symm"
    elif model == "MSE":
        models = analysis.models_best_MSE
        ext = ext+"_best_MSE"
    elif model == "tot":
        models = analysis.models_best_tot
        ext = ext+"_best_tot"

    #####################################

    plt.clf()
    fig = plt.figure()
    err = {}
    # color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]
    model_data_trans = []
    truth_new = []

    for i,beta in enumerate(beta_range):
        trans_new_data = Lorentz_Trans(data = data,beta = beta,beta_dir = beta_dir)
        

        if analysis.broken_symm == "True" or analysis.broken_symm == True:
            
            if transformed_spurions == "True" or transformed_spurions == True:
                spurions = [Lorentz_Trans(data = spurion.to(devicef), beta = beta, beta_dir = beta_dir.to(devicef)) for spurion in analysis.spurions]
            else:
                spurions = analysis.spurions
                
            truth_new.append(Lorentz_myfun_broken(trans_new_data ,spurions = spurions))
            
            if analysis.input_spurions == "True" or analysis.input_spurions==True:
                expand_spurions = (torch.cat(spurions)).expand(trans_new_data.shape[0],len_spurions)
                model_data_trans.append(torch.cat((trans_new_data.to(devicef),expand_spurions.to(devicef)),dim = -1))
            else:
                model_data_trans.append(trans_new_data.to(devicef))
                
            
        else:
            truth_new.append(Lorentz_myfun(trans_new_data))



    for i,lam_val in enumerate(analysis.models.keys()):
        err[lam_val] = torch.zeros_like(beta_range) 
        mse = ((analysis.train_labels.cpu().squeeze()-models[lam_val](inputs).detach().cpu().squeeze())**2).mean()#analysis.train_loss_lam[lam_val][-1]
        mse = "{:.1e}".format(mse)
        loss_full = SymmLoss(gens_list=gens_Lorentz, model = models[lam_val])
        symm = loss_full(inputs)
        symm = "{:.1e}".format(symm)
        label = rf"$\lambda$ = {lam_val}, train MSE = {mse}, symm = {symm}"

        if analysis.broken_symm == "True" or analysis.broken_symm == True:
            loss_unbroken = SymmLoss(gens_list=gens_Lorentz[[0,1,-1]], model = models[lam_val])
            symm_unbroken  = loss_unbroken(inputs)
            symm_unbroken = "{:.1e}".format(symm_unbroken)
            label = f"{label}, unbroken = {symm_unbroken}"
            
        for j,beta in enumerate(beta_range):
            if relative:
                err[lam_val][j] = (((truth_new[j].cpu().squeeze()-models[lam_val](model_data_trans[j]).detach().cpu().squeeze())/truth_new[j].cpu().squeeze())**2).mean()
            else:
                err[lam_val][j] = (((truth_new[j].cpu().squeeze()-models[lam_val](model_data_trans[j]).detach().cpu().squeeze()))**2).mean()

                
#       
        plt.semilogy(beta_range, err[lam_val],label = label, color = colors_dict[lam_val])
        
    plt.legend()
    plt.annotate(rf"$\hat\beta = {beta_dir}$",xy=(0.05,0.35),xycoords = "axes fraction")
    plt.xlabel(r"$\beta$")
    if relative:
        plt.ylabel("relative MSE")
    else:
        plt.ylabel("Mean Squared Error")
    text = analysis.title()
    plt.title(text)
        
    # if analysis.print_spurions == "True" or analysis.print_spurions == True:
    plt.annotate(analysis.spurions_for_print,xy=(0.05,0.4),xycoords = "axes fraction")
           
    if save==True or save=="True":
        if filename =="":
                file = filename
        else:
            file = f"_{filename}_"
        # fig.show()
        fig.savefig(f"{outdir}/performance_beta_{ext}{file}{analysis.filename}.pdf")

    else:
        fig.show()


            
        # for i,theta in enumerate(theta_range):
        #     rot_data[i] = rot(data = analysis.train_data.to(devicef),theta = theta,theta_dir = theta_dir)



def performance_plot_ext_theta(analysis,theta_range = torch.linspace(0,2*np.pi,100),theta_dir =torch.tensor([0.0,0.0,1.0]),model = "last",transformed_spurions = "True",save = False, outdir = "./plots",filename = "",relative = False):
        
    ######### initialize #################   
    trans_new_data = torch.zeros(len(theta_range))
    rot_data = torch.zeros(len(theta_range))
    data = analysis.train_data.to(devicef)
    inputs = analysis.train_data.to(devicef)
    ext = ""
    if analysis.input_spurions == "True" or analysis.input_spurions==True:
        lens_spurions = [torch.numel(sp) for sp in analysis.spurions]
        len_spurions = sum(lens_spurions)
        data = data[:,0:-len_spurions]
        if transformed_spurions == "False" or transformed_spurions == False:
            ext = "_same_frame"
        else:
            ext = "_trans_frame"

    if model == "last":
        models = analysis.models
        ext = ext
    elif model== "symm":
        models = analysis.models_best_symm
        ext = ext+"_best_symm"
    elif model == "MSE":
        models = analysis.models_best_MSE
        ext = ext+"_best_MSE"
    elif model == "tot":
        models = analysis.models_best_tot
        ext = ext+"_best_tot"

    #####################################

    plt.clf()
    fig = plt.figure()
    err = {}
    # color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]
    model_data_trans = []
    truth_new = []

    for i,theta in enumerate(theta_range):
        trans_new_data = rot(data = data,theta = theta,theta_dir = theta_dir)
        
        if analysis.broken_symm == "True" or analysis.broken_symm == True:
            
            if transformed_spurions == "True" or transformed_spurions == True:
                spurions = [Lorentz_Trans(data = spurion.to(devicef), beta = beta, beta_dir = beta_dir.to(devicef)) for spurion in analysis.spurions]
            else:
                spurions = analysis.spurions
                
            truth_new.append(Lorentz_myfun_broken(trans_new_data ,spurions = spurions))
            
            if analysis.input_spurions == "True" or analysis.input_spurions==True:
                expand_spurions = (torch.cat(spurions)).expand(trans_new_data.shape[0],len_spurions)
                model_data_trans.append(torch.cat((trans_new_data.to(devicef),expand_spurions.to(devicef)),dim = -1))
            else:
                model_data_trans.append(trans_new_data.to(devicef))
                
            
        else:
            truth_new.append(Lorentz_myfun(trans_new_data))
        
        
        
        
        
        
#         if analysis.input_spurions == "True" or analysis.input_spurions==True:
#             if transformed_spurions == "True" or transformed_spurions == True:
#                 spurions = [rot(data = spurion.to(devicef), theta = theta, theta_dir = theta_dir.to(devicef)) for spurion in analysis.spurions]
#             else:
#                 spurions = analysis.spurions

#             expand_spurions = (torch.cat(spurions)).expand(trans_new_data.shape[0],len_spurions)
#             model_data_trans.append(torch.cat((trans_new_data.to(devicef),expand_spurions.to(devicef)),dim = -1))

#         if analysis.broken_symm == "True" or analysis.broken_symm == True:
#             truth_new.append(Lorentz_myfun_broken(data ,spurions = spurions))
#         else:
#             truth_new.append(Lorentz_myfun(data))



    for i,lam_val in enumerate(analysis.models.keys()):
        err[lam_val] = torch.zeros_like(theta_range) 
        mse = ((analysis.train_labels.cpu().squeeze()-models[lam_val](inputs).detach().cpu().squeeze())**2).mean()#analysis.train_loss_lam[lam_val][-1]
        mse = "{:.1e}".format(mse)
        loss_full = SymmLoss(gens_list=gens_Lorentz, model = models[lam_val])
        symm = loss_full(inputs)
        symm = "{:.1e}".format(symm)
        label = rf"$\lambda$ = {lam_val}, train MSE = {mse}, symm = {symm}"

        if analysis.broken_symm == "True" or analysis.broken_symm == True:
            loss_unbroken = SymmLoss(gens_list=gens_Lorentz[[0,1,-1]], model = models[lam_val])
            symm_unbroken  = loss_unbroken(inputs)
            symm_unbroken = "{:.1e}".format(symm_unbroken)
            label = f"{label}, unbroken = {symm_unbroken}"
            
        for j,theta in enumerate(theta_range):
            if relative:
                err[lam_val][j] = (((truth_new[j].cpu().squeeze()-models[lam_val](model_data_trans[j]).detach().cpu().squeeze())/truth_new[j].cpu().squeeze())**2).mean()
            else:
                err[lam_val][j] = (((truth_new[j].cpu().squeeze()-models[lam_val](model_data_trans[j]).detach().cpu().squeeze()))**2).mean()

                
#       
        plt.semilogy(theta_range, err[lam_val],label = label, color = color_vec[i%len(color_vec)])
        
    plt.legend()
    plt.annotate(rf"$\hat\theta = {theta_dir}$",xy=(0.05,0.35),xycoords = "axes fraction")
    plt.xlabel(r"$\theta$")
    if relative:
        plt.ylabel("relative MSE")
    else:
        plt.ylabel("Mean Squared Error")
    text = analysis.title()
    plt.title(text)
        
    # if analysis.print_spurions == "True" or analysis.print_spurions == True:
    plt.annotate(analysis.spurions_for_print,xy=(0.05,0.4),xycoords = "axes fraction")
           
    if save==True or save=="True":
        if filename =="":
                file = filename
        else:
            file = f"_{filename}_"
        # fig.show()
        fig.savefig(f"{outdir}/performance_theta_{ext}{file}{analysis.filename}.pdf")

    else:
        fig.show()


            
        # for i,theta in enumerate(theta_range):
        #     rot_data[i] = rot(data = analysis.train_data.to(devicef),theta = theta,theta_dir = theta_dir)



def loss_res_fun(a,mag = 1.0,spurion_dir = [0.0,0.0,0.0,1.0],analysis_spurions = False, generators = gens_Lorentz):
    if analysis_spurions:
        spurions = a.spurions
    else:
        spurions = [mag*torch.tensor(spurion_dir)]
    mymodelLorentz = broken_model(dinput = 4, init = "eta",spurions = spurions)
    lossLorentz = SymmLoss(model = mymodelLorentz, gens_list=generators)
    loss_res = lossLorentz(input = a.train_data.to(devicef)).detach().cpu()
    return loss_res



def analyze(filename, model= "MSE",theta = 0.5,theta_dir = torch.tensor([0,0,1]),beta = 0.6,beta_dir = torch.tensor([1,0,0]),transformed_spurions = "True",plot = True):
    with open(f"./storage/{filename}.pkl","rb") as f:
        a = pickle.load(f)
    a.plot_losses()
    if plot=="True" or plot== True:
        if transformed_spurions == "False" or transformed_spurions == False:
            ext = "_same_frame"
        else:
            ext = "_trans_frame"

        trans_new_data = Lorentz_Trans(data = a.train_data.to(devicef),beta = beta,beta_dir = beta_dir)
        rot_data = rot(data = a.train_data.to(devicef),theta = theta,theta_dir = theta_dir)
        pred_plot_ext(a,rot_data,model = model, save = True, outdir = "./plots",filename = f"rot_{theta}{ext}",transformed_spurions = transformed_spurions)
        pred_plot_ext(a,trans_new_data,model = model,save = True, outdir = "./plots",filename = f"boost_{beta}{ext}",transformed_spurions = transformed_spurions)
    return a



def Lorentz_Trans(data,beta,beta_dir = torch.tensor([1,0,0])):
    gamma = 1/np.sqrt(1-beta**2)
    beta_dir = beta_dir/(torch.sqrt(torch.sum(beta_dir**2)))
    # LorentzBoost = torch.tensor([[gamma, -gamma*beta,  0, 0],[-gamma*beta, gamma, 0, 0],[0,0,1,0],[0,0,0,1]],dtype = torch.float32).to(devicef)
    LorentzBoost = torch.diag(torch.tensor([gamma,1,1,1],dtype = torch.float32)).to(devicef)
    for i in range(1,4):
        LorentzBoost[0,i] += -gamma*beta*beta_dir[i-1]
        LorentzBoost[i,0] += -gamma*beta*beta_dir[i-1]
        for j in range(1,4):
            LorentzBoost[i,j] += 0.5*(gamma-1)*beta_dir[i-1]*beta_dir[j-1]
            LorentzBoost[j,i] += 0.5*(gamma-1)*beta_dir[i-1]*beta_dir[j-1]
    d = 4
    data = einops.rearrange(data, '... (N d) -> ... N d',d = 4)
    trans_data = torch.einsum("ij,...j-> ...i",LorentzBoost,data).to(devicef)
    trans_data = einops.rearrange(trans_data, '... N d -> ... (N d)', d = 4)
    # print(LorentzBoost)
    return trans_data


def rot(data,theta,theta_dir = torch.tensor([0,0,1])):
    theta_dir = (theta_dir.to(devicef)/(torch.sqrt(torch.sum(theta_dir**2))))
    # rotate = torch.tensor([[1, 0,  0, 0],[0, np.cos(theta), np.sin(theta), 0],[0,-np.sin(theta),np.cos(theta),0],[0,0,0,1]],dtype = torch.float32).to(devicef)
    
    rotate = torch.diag(torch.tensor([1,1,1,1],dtype = torch.float32)).to(devicef)
    L_gens = gens_Lorentz[3::].to(devicef)
    e_L_gens = torch.einsum("i,ijk-> jk",theta_dir,L_gens).to(devicef)
    rotate = rotate+np.sin(theta)*e_L_gens+(1-np.cos(theta))*torch.matmul(e_L_gens,e_L_gens)
    
    data = einops.rearrange(data, '... (N d) -> ... N d',d = 4)
    trans_data = torch.einsum("ij,...j-> ...i",rotate,data).to(devicef)
    trans_data = einops.rearrange(trans_data, '... N d -> ... (N d)', d = 4)
    return trans_data



def performance_calc_ext(analysis,beta_range = torch.linspace(0,1,100),beta_dir = torch.tensor([1.0,0.0,0.0]),theta_range = torch.linspace(0,2*np.pi,100),theta_dir =torch.tensor([0.0,0.0,1.0]),model = "last",transformed_spurions = "True",save = False, outdir = "./plots",filename = "",relative = False,test_data = "train",N = 10000,norm = 1,dinput = 4, seed = 29487):
    err = {}
    err_analysis = []
    # color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]

    ######### initialize #################   
    trans_new_data = torch.zeros(len(beta_range))
    rot_data = torch.zeros(len(theta_range))
    if test_data=="train":
        data = analysis.train_data.to(devicef)
    else:
        np.random.seed(seed)
        torch.manual_seed(seed)
        data = ((torch.rand(N,dinput)-0.5)*norm).to(devicef)
        
    inputs = analysis.train_data.to(devicef)
    ext = ""
    if analysis.input_spurions == "True" or analysis.input_spurions==True:
        lens_spurions = [torch.numel(sp) for sp in analysis.spurions]
        len_spurions = sum(lens_spurions)
        data = data[:,0:-len_spurions]
        if transformed_spurions == "False" or transformed_spurions == False:
            ext = "_same_frame"
        else:
            ext = "_trans_frame"

    if model == "last":
        models = analysis.models
        ext = ext
    elif model== "symm":
        models = analysis.models_best_symm
        ext = ext+"_best_symm"
    elif model == "MSE":
        models = analysis.models_best_MSE
        ext = ext+"_best_MSE"
    elif model == "tot":
        models = analysis.models_best_tot
        ext = ext+"_best_tot"

    #####################################

    
    err = {}
    model_data_trans = []
    truth_new = []

    for i,beta in enumerate(beta_range):
        trans_new_data = Lorentz_Trans(data = data,beta = beta,beta_dir = beta_dir)


        if analysis.broken_symm == "True" or analysis.broken_symm == True:

            if transformed_spurions == "True" or transformed_spurions == True:
                spurions = [Lorentz_Trans(data = spurion.to(devicef), beta = beta, beta_dir = beta_dir.to(devicef)) for spurion in analysis.spurions]
            else:
                spurions = analysis.spurions

            truth_new.append(Lorentz_myfun_broken(trans_new_data ,spurions = spurions))

            if analysis.input_spurions == "True" or analysis.input_spurions==True:
                expand_spurions = (torch.cat(spurions)).expand(trans_new_data.shape[0],len_spurions)
                model_data_trans.append(torch.cat((trans_new_data.to(devicef),expand_spurions.to(devicef)),dim = -1))
            else:
                model_data_trans.append(trans_new_data.to(devicef))


        else:
            truth_new.append(Lorentz_myfun(trans_new_data))



    for i,lam_val in enumerate(analysis.models.keys()):
    
        err[lam_val] = torch.zeros_like(beta_range) 
        # mse = ((analysis.train_labels.cpu().squeeze()-models[lam_val](inputs).detach().cpu().squeeze())**2).mean()#analysis.train_loss_lam[lam_val][-1]
        # mse = "{:.1e}".format(mse)
        # loss_full = SymmLoss(gens_list=gens_Lorentz, model = models[lam_val])
        # symm = loss_full(inputs)
        # symm = "{:.1e}".format(symm)
        

#         if analysis.broken_symm == "True" or analysis.broken_symm == True:
#             loss_unbroken = SymmLoss(gens_list=gens_Lorentz[[0,1,-1]], model = models[lam_val])
#             symm_unbroken  = loss_unbroken(inputs)
#             symm_unbroken = "{:.1e}".format(symm_unbroken)
            #label = f"{label}, unbroken = {symm_unbroken}"

        for j,beta in enumerate(beta_range):
            if relative:
                err[lam_val][j] = (((truth_new[j].cpu().squeeze()-models[lam_val](model_data_trans[j]).detach().cpu().squeeze())/truth_new[j].cpu().squeeze())**2).mean()
            else:
                err[lam_val][j] = (((truth_new[j].cpu().squeeze()-models[lam_val](model_data_trans[j]).detach().cpu().squeeze()))**2).mean()

        #err_analysis[lam_val].append(err[lam_val])

    return err, beta_range
                
#  

            
        # for i,theta in enumerate(theta_range):
        #     rot_data[i] = rot(data = analysis.train_data.to(devicef),theta = theta,theta_dir = theta_dir)



def performance_plot_ext_many_clean(errs,lams,beta_range, title = "",save = True, outdir = "./plots",filename = ""):
    # color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]

    #plt.clf()
    fig = plt.figure()
    set_plot_params()
    err_mean = {}
    for i,lam_val in enumerate(lams):
        # err_mean[lam_val] = torch.zeros_like(beta_range)
        lam_val_label = lam_val if float(lam_val)<1.0 else round(lam_val)
        label = r"$\lambda$ ="+f" {lam_val_label}" if lam_val!=0 else "Baseline"
        # label = rf"$\lambda$ = {'{:.1e}'.format(lam_val)}"
        #print(label)
        #print(errs)
        if any(lam_val in err.keys() for err in errs):
            err_vec = torch.stack([err[lam_val] for err in errs if lam_val in err.keys()])
        
        # for err in errs:
        #     err_mean[lam_val]+=  err[lam_val] if lam_val in err.keys()
        #     #print(err_mean)
        # err_mean[lam_val] = (1/len(errs))*err_mean[lam_val]
            err_mean[lam_val] = err_vec.mean(dim = 0)
            plt.semilogy(beta_range, err_mean[lam_val],label = label, color = colors_dict[lam_val])

    plt.legend()
    #plt.annotate(rf"$\hat\beta = {beta_dir}$",xy=(0.05,0.35),xycoords = "axes fraction")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Mean Squared Error")
    plt.ylim([1e-7,1e2])
    
    text = title

    plt.title(text)
    set_plot_params()
    # if analysis.print_spurions == "True" or analysis.print_spurions == True:
    #plt.annotate(analysis.spurions_for_print,xy=(0.05,0.4),xycoords = "axes fraction")
           
    if save==True or save=="True":
        
        if filename =="":
                file = filename
        else:
            file = f"_{filename}_"
        # fig.show()
        print(f"saving {outdir}/performance_beta_mean{file}.pdf")
        fig.savefig(f"{outdir}/performance_beta_mean{file}.pdf")

    else:
        fig.show()
        
    return err_mean, beta_range



def performance_plot_all(files,beta_range = torch.linspace(0,1,100),beta_dir = torch.tensor([1.0,0.0,0.0]),theta_range = torch.linspace(0,2*np.pi,100),theta_dir =torch.tensor([0.0,0.0,1.0]),model = "last",transformed_spurions = "False",save = True, outdir = "./plots",filename = "",relative = False, lams = [0.0], N=10000,norm=1,dinput=4,seed = "rand", test_data = "train",beta_max = 0.95 ):
    errs = []
    # color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]
    count = 0
    if seed.isnumeric():
        seed_data = int(seed_data)
    else:
        seed_data = int(torch.round(torch.rand(1)*10000))
    for file in files:
        with open(file,"rb") as f:
            a = pickle.load(f)
            Nepochs = len(a.train_outputs[0.0]['Loss'])
            if (not (hasattr(a,"beta_max"))):
                    a.beta_max = 0.95
            if (Nepochs >= 999) and (hasattr(a,"clip_grads")) and (any(lam in lams for lam in a.train_outputs.keys())):
                if a.beta_max==beta_max:
                    err, beta_vec = performance_calc_ext(a,beta_range = beta_range, beta_dir = beta_dir,transformed_spurions = False,save = save, outdir = outdir, filename = filename, relative = False, N=N, norm = norm, dinput = dinput,seed = seed_data, test_data = test_data)
                    errs.append(err)
                    count+=1
    
    err_mean,beta_range =  performance_plot_ext_many_clean(errs = errs,lams = lams,beta_range = beta_range, filename=filename, save = save, outdir = outdir)
    print(f"average over {count} samples")
    return err_mean,beta_range
                

def performance_plot_symm_MSE(err_mean,beta_range = torch.linspace(0,1,100),beta_dir = torch.tensor([1.0,0.0,0.0]),theta_range = torch.linspace(0,2*np.pi,100),theta_dir =torch.tensor([0.0,0.0,1.0]),save = True, outdir = "./plots", title = "",filename = "",lams = [0.0], N=10000,norm=1,dinput=4,seed = "rand", test_data = "train" ):
    errs = []
    # color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]
    
 
    plt.clf()
    fig = plt.figure()
    err_mean = err_mean.copy()
    for i,lam_val in enumerate(lams):
        lam_val_label = lam_val if float(lam_val)<1.0 else round(lam_val)
        label = r"$\lambda$ ="+f" {lam_val_label}" if lam_val!=0 else "Baseline"
        #print(label)
        #plt.semilogy(beta_range, err_mean['symm'][lam_val],label = r"$\lambda_{dSymm}$ ="+f" {lam_val}", color = color_vec[i])
        #plt.semilogy(beta_range, err_mean['MSE'][lam_val],label = r"$\lambda_{GSymm}$ ="+f" {lam_val}", color = color_vec[i],ls = "--")
        plt.semilogy(beta_range, err_mean['dsymm'][lam_val],label = label, color = colors_dict[lam_val])
        plt.semilogy(beta_range, err_mean['Gsymm'][lam_val], color = colors_dict[lam_val],ls = "--")



    plt.legend()
    #legend with solid - symm, dashed - MSE
    # Create custom legend handles for solid and dashed black lines
    custom_handles = [
        lines.Line2D([0], [0], color='black', linestyle='-', label=r'$\delta$SEAL'),
        lines.Line2D([0], [0], color='black', linestyle='--', label='GSEAL')
    ]

    # Get the automatic legend handles and labels
    auto_handles, auto_labels = plt.gca().get_legend_handles_labels()

    # Combine automatic and custom legend entries
    combined_handles = auto_handles + custom_handles
    combined_labels = auto_labels + [handle.get_label() for handle in custom_handles]

    # Add the legend
    plt.legend(combined_handles, combined_labels)


    #plt.annotate(rf"$\hat\beta = {beta_dir}$",xy=(0.05,0.35),xycoords = "axes fraction")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Mean Squared Error")
    plt.ylim([1e-7,1e2])

    text = title
    plt.title(text)
    set_plot_params()
    # if analysis.print_spurions == "True" or analysis.print_spurions == True:
    #plt.annotate(analysis.spurions_for_print,xy=(0.05,0.4),xycoords = "axes fraction")

    if save==True or save=="True":

        if filename =="":
                file = filename
        else:
            file = f"_{filename}_"
        # fig.show()
        print(f"saving {outdir}/performance_beta_mean{file}.pdf")
        fig.savefig(f"{outdir}/performance_beta_mean{file}.pdf")

    else:
        fig.show()


#Performance as a funciton of lambda

def performance_calc_lambda(analysis, model = "last",transformed_spurions = "True",save = False, outdir = "./plots",filename = "",relative = False,test_data = "train",N = 10000,norm = 1,dinput = 4, seed = 29487,beta_max = 0.95):
    err = {}
    err_analysis = []
    # color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]

    ######### initialize #################   
   
    if test_data=="train":
        data = analysis.train_data.to(devicef)
    else:
        np.random.seed(seed)
        torch.manual_seed(seed)
        data = ((torch.rand(N,dinput)-0.5)*norm).to(devicef)
        
    ext = ""
    if analysis.input_spurions == "True" or analysis.input_spurions==True:
        lens_spurions = [torch.numel(sp) for sp in analysis.spurions]
        len_spurions = sum(lens_spurions)
        data = data[:,0:-len_spurions]
        if transformed_spurions == "False" or transformed_spurions == False:
            ext = "_same_frame"
        else:
            ext = "_trans_frame"

    if model == "last":
        models = analysis.models
        ext = ext
    elif model== "symm":
        models = analysis.models_best_symm
        ext = ext+"_best_symm"
    elif model == "MSE":
        models = analysis.models_best_MSE
        ext = ext+"_best_MSE"
    elif model == "tot":
        models = analysis.models_best_tot
        ext = ext+"_best_tot"

    #####################################

    
    err = {}
    symm = {}
    mse = {}
    # model_data_trans = []
    # truth_new = []

    # for i,beta in enumerate(beta_range):
    
    new_data = data.detach()
    trans_new_data = boost_3d(new_data, devicef,beta_max = beta_max)


    if analysis.broken_symm == "True" or analysis.broken_symm == True:

#         if transformed_spurions == "True" or transformed_spurions == True:
#             spurions = [Lorentz_Trans(data = spurion.to(devicef), beta = beta, beta_dir = beta_dir.to(devicef)) for spurion in analysis.spurions]
#         else:
#             spurions = analysis.spurions
        #spurions are currently not dealt with!
        spurions = analysis.spurions
    
        truth_new = Lorentz_myfun_broken(new_data ,spurions = spurions)
        truth_new_trans = Lorentz_myfun_broken(trans_new_data ,spurions = spurions)

        if analysis.input_spurions == "True" or analysis.input_spurions==True:
            expand_spurions = (torch.cat(spurions)).expand(new_data.shape[0],len_spurions)
            model_data = torch.cat((new_data.to(devicef),expand_spurions.to(devicef)),dim = -1)
            
            expand_spurions_trans = (torch.cat(spurions)).expand(trans_new_data.shape[0],len_spurions)
            model_data_trans = torch.cat((trans_new_data.to(devicef),expand_spurions.to(devicef)),dim = -1)
            
        else:
            model_data = new_data.to(devicef)
            model_data_trans = trans_new_data.to(devicef)


    else:
        truth_new = Lorentz_myfun(new_data)
        truth_new_trans = Lorentz_myfun(trans_new_data)



    for i,lam_val in enumerate(analysis.models.keys()):
        model = models[lam_val]
        err[lam_val] = (((truth_new.cpu().squeeze()-model(model_data).detach().cpu().squeeze()))**2).mean()
        
        loss_full = SymmLoss(gens_list=gens_Lorentz, model = model )
        symm[lam_val] = loss_full(model_data).data.cpu()
        
        
        outputs_boost = model(trans_new_data)
        mse[lam_val] = ((model(model_data).detach().cpu().squeeze()-model(model_data_trans).detach().cpu().squeeze())**2).mean()#analysis.train_loss_lam[lam_val][-1]
        
       
        

#         if analysis.broken_symm == "True" or analysis.broken_symm == True:
#             loss_unbroken = SymmLoss(gens_list=gens_Lorentz[[0,1,-1]], model = models[lam_val])
#             symm_unbroken  = loss_unbroken(inputs)
#             symm_unbroken = "{:.1e}".format(symm_unbroken)
            #label = f"{label}, unbroken = {symm_unbroken}"
    metrics = {"BCE": err, "Symm_Loss": symm, "MSE_Loss": mse }

    return metrics
                
#  

            
        # for i,theta in enumerate(theta_range):
        #     rot_data[i] = rot(data = analysis.train_data.to(devicef),theta = theta,theta_dir = theta_dir)



def performance_plot_lambda_many_clean(errs,lams = "all", title = "",save = True, outdir = "./plots",filename = "",lin_logx = "lin",lin_logy = "log", apply_symm = True, apply_MSE = True, err=True, symm=True, MSE = True, beta_max = ""):
    # color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]

    # plt.clf()
    fig = plt.figure()
    err_mean = {}
    symm_mean = {}
    mse_mean = {}
    err_std = {}
    symm_std = {}
    mse_std = {}
    
    if not errs:
        print("empty metrics")
        return err_mean, symm_mean, mse_mean
    
    else:
    
        if lams == "all":
            lams = errs[0]["BCE"].keys()

        for i,lam_val in enumerate(lams):
            err_mean[lam_val] = 0
            symm_mean[lam_val] = 0
            mse_mean[lam_val] = 0
            err_std[lam_val] = 0
            symm_std[lam_val] = 0
            mse_std[lam_val] = 0

            #print(label)
            err_vec = torch.tensor([err["BCE"][lam_val] for err in errs if lam_val in err["BCE"].keys()])
            symm_vec = torch.tensor([err["Symm_Loss"][lam_val] for err in errs if lam_val in err["Symm_Loss"].keys()])
            mse_vec = torch.tensor([err["MSE_Loss"][lam_val] for err in errs if lam_val in err["MSE_Loss"].keys()])
            # for err in errs:
            #     err_mean[lam_val]+=  err[lam_val]
            #     #print(err_mean)
            err_mean[lam_val] = err_vec.mean()
            err_std[lam_val] = err_vec.std()
            symm_mean[lam_val] = symm_vec.mean()
            symm_std[lam_val] = symm_vec.std()
            mse_mean[lam_val] = mse_vec.mean()
            mse_std[lam_val] = mse_vec.std()
        print(f"err = {err_mean}")
        print(f"symm = {symm_mean}")
        print(f"MSE = {mse_mean}")
        
       
        #plt.annotate(rf"$\hat\beta = {beta_dir}$",xy=(0.05,0.35),xycoords = "axes fraction")
        ax = plt.axes()
        if lin_logx=="log":
            ax.set_xscale("log")
        if lin_logy=="log":
            ax.set_yscale("log")
        
        if err:
            lam_vals = [lam for lam in lams if not(torch.isnan(err_mean[lam]))]
            plt.errorbar(lam_vals, [err_mean[lam] for lam in lam_vals], [err_std[lam] for lam in lam_vals],label = "task loss", color = "violet")
        if symm:
            lam_vals = [lam for lam in lams if not(torch.isnan(symm_mean[lam]))]
            plt.errorbar(lam_vals, [symm_mean[lam] for lam in lam_vals],[symm_std[lam] for lam in lam_vals],label = "dSymm", color = "pink")
        if MSE:
            lam_vals = [lam for lam in lams if not(torch.isnan(symm_mean[lam]))]
            MSE_label = "GSEAL" if beta_max=="" else "GSEAL "+r"$\beta_{max}$="+f"{beta_max}"
            plt.errorbar(lam_vals, [mse_mean[lam] for lam in lam_vals],[mse_std[lam] for lam in lam_vals] ,label = MSE_label, color = "blue")

        plt.xlabel(r"$\lambda$")
        if apply_symm == True:
            plt.xlabel(r"$\lambda_{\delta SEAL}$")
        elif apply_MSE == True:
            plt.xlabel(r"$\lambda_{GSEAL}$")

        plt.ylabel("Loss")
        # plt.ylim([1e-7,1e2])
        
        plt.legend()
        text = title
        plt.title(text)

        # if analysis.print_spurions == "True" or analysis.print_spurions == True:
        #plt.annotate(analysis.spurions_for_print,xy=(0.05,0.4),xycoords = "axes fraction")

        if save==True or save=="True":

            if filename =="":
                    file = filename
            else:
                file = f"_{filename}_"
            # fig.show()
            print(f"saving {outdir}/performance_lambda_mean{file}.pdf")
            fig.savefig(f"{outdir}/performance_lambda_mean{file}.pdf")

        else:
            fig.show()

        return err_mean, symm_mean, mse_mean
    

def performance_plot_lambda_all(files,model = "last",transformed_spurions = "False",save = True, outdir = "./plots",filename = "",relative = False, lams = "all", N=10000,norm=1,dinput=4,seed = "rand", test_data = "train",lin_logx = "lin",lin_logy = "log", apply_symm = True, apply_MSE = True,err = True,symm = True, MSE = True, betas = [0.95],betas_symm = [0.95] ):
    metrics= {"symm":{},"MSE":{}}
    metrics_mean= {"symm":{},"MSE":{}}
    for beta in betas_symm:
        metrics["symm"][beta]= [] 
        metrics_mean["symm"][beta] = [] 
    for beta in betas:
        metrics["MSE"][beta] = []
        metrics_mean["MSE"][beta] = [] 
                 
    count_symm = 0
    count_MSE = 0
    # color_vec = ["deepskyblue","blue","blueviolet","violet","magenta","deeppink","pink"]
    if seed.isnumeric():
        seed_data = int(seed_data)
    else:
        seed_data = int(torch.round(torch.rand(1)*10000))
    for file in files:
        with open(file,"rb") as f:
            a = pickle.load(f)
            Nepochs = len(a.train_outputs[0.0]['Loss'])
            if (Nepochs >= 999) and (hasattr(a,"clip_grads")) and ((a.apply_symm == apply_symm) or (a.apply_MSE == apply_MSE) or ((a.apply_MSE==False) and (a.apply_symm==False))):           
                if (not (hasattr(a,"beta_max"))):
                    a.beta_max = 0.95
                    #print(metric)
                if a.apply_symm:
                    # metric = {}
                    for beta in betas_symm:
                        metric = performance_calc_lambda(a,transformed_spurions = False,save = save, outdir = outdir, filename = filename, relative = False, N=N, norm = norm, dinput = dinput,seed = seed_data, test_data = test_data,beta_max = beta)
                        metrics["symm"][beta].append(metric)
                    count_symm+=1
                if a.apply_MSE:
                    if a.beta_max in betas:
                        metric = performance_calc_lambda(a,transformed_spurions = False,save = save, outdir = outdir, filename = filename, relative = False, N=N, norm = norm, dinput = dinput,seed = seed_data, test_data = test_data,beta_max = a.beta_max)
                        metrics["MSE"][a.beta_max].append(metric)
                        count_MSE+=1
    # print(metrics)
    if apply_MSE:
        for beta in betas:
            err_mean, symm_mean, mse_mean=  performance_plot_lambda_many_clean(errs = metrics["MSE"][beta],lams = lams,outdir = outdir,filename=filename+f"_MSE_beta_max_{beta}", save = save,lin_logx = lin_logx,lin_logy = lin_logy, apply_symm = False, apply_MSE = True, err = err, symm = symm, MSE = MSE, beta_max = f"{beta}")
            metrics_mean["MSE"][beta] = {"BCE": err_mean, "Symm_Loss": symm_mean, "MSE_Loss": mse_mean}
            print(f"beta max = {beta} Gsymm average over {count_MSE} samples")
    if apply_symm:
        for beta in betas_symm:
            err_mean, symm_mean, mse_mean=  performance_plot_lambda_many_clean(errs = metrics["symm"][beta],lams = lams,outdir = outdir, filename=filename+f"_symm_beta_max_{beta}", save = save,lin_logx = lin_logx,lin_logy = lin_logy, apply_symm = True, apply_MSE = False, err = err, symm = symm, MSE = MSE, beta_max = f"{beta}")
            metrics_mean["symm"][beta] = {"BCE": err_mean, "Symm_Loss": symm_mean, "MSE_Loss": mse_mean}
            print(f"dsymm average over {count_symm} samples")

    
    return metrics_mean,metrics


def get_files(storage_dir=None,Nepochs = 999,apply_symm = True,apply_MSE = True,beta_max = 0.95,broken = False,spurion_mag = 0.0):
    if storage_dir is None:
        storage_dir = storage_dir
    files = glob.glob(f"{storage_dir}/*toy*symm_*MSE_*broken_symm_spurion*{spurion_mag}]*")
    files_filtered =[]
    for file in files:
        with open(file,"rb") as f:
            a = pickle.load(f)
            #print a attributes
            # print(f"file = {file}")
            nepochs = len(a.train_outputs[0.0]['Loss'])
            if (nepochs >= 999) and (hasattr(a,"clip_grads")) and ((a.apply_symm == apply_symm or apply_MSE == a.apply_MSE) or ((a.apply_MSE==False) and (a.apply_symm==False))):
                # print(f"file = {file} has Nepochs = {nepochs}, apply_symm = {a.apply_symm} and apply_MSE = {a.apply_MSE}")
                if (not (hasattr(a,"beta_max"))):
                    a.beta_max = 0.95
                if (a.beta_max == beta_max) and (a.broken_symm == broken or a.broken_symm == f"{broken}"):
                    # print(f"file = {file} has broken = {a.broken_symm} and beta_max = {a.beta_max}=={beta_max}")
                    # if (broken and (a.spurion_mag == spurion_mag)) or (not broken):
                    files_filtered.append(file)
                    # print(f"file = {file}")
            # print(files_filtered)
    return files_filtered


def main():
    storage_dir_nersc="/pscratch/sd/i/inbarsav/SymmLoss/storage"
    Nepochs = 999
    models = ["dsymm","Gsymm"]
    apply_symm = [True,False]
    apply_MSE = [True,False]
    broken = True
    spurion_mag = [0.0,0.001]
    beta_max_vec = [0.1,0.5,0.95]#[0.001,0.1,0.5,0.95]#[0.95]
    files =[]
    beta_dir = torch.tensor([0.0,0.0,1.0])
    # lams = [0.0,0.1,1.0,10.0,100.0]
    lams = [0.0,0.1,1.0,10.0,100.0] #[0.0,0.1,100]
    err_mean ={ beta:{mod:{} for mod in models} for beta in beta_max_vec}
    outdir = "./results/toys"
    save = True
    filename_both = "toy_z_symm_MSE_new"
    for mag in spurion_mag:
        spur_name = f"spurion_mag_{mag}"
        for model in models:
            if model=="dsymm":
                apply_symm = True
                apply_MSE = False
                beta_maxs = [0.95]
                files = []  
            else:
                apply_symm = False 
                apply_MSE = True
                beta_maxs = beta_max_vec
        
            for beta_max in beta_maxs:
                files =[]
                print(f"searching files with Nepochs = {Nepochs}, apply_symm = {apply_symm}, apply_MSE = {apply_MSE}, broken = {broken}, spurion_mag = {mag}, beta_max = {beta_max}")
                files = get_files(storage_dir=storage_dir_nersc,Nepochs = Nepochs,apply_symm = apply_symm,apply_MSE = apply_MSE,broken = broken,spurion_mag = mag,beta_max=beta_max)
                # print(files)
                if files!=[]:
                    filename = f"toys_{model}_{spur_name}_beta_max_{beta_max}" if model=="Gsymm" else f"{model}_{spur_name}"
                    err_mean[beta_max][model],_ = performance_plot_all(files,beta_dir = beta_dir,outdir = outdir,filename = f"{filename}_clean_4",lams = lams, save = True,test_data = "rand",seed = "rand",beta_max = beta_max)
                    print(f"plotted spurion mag = {mag} beta max = {beta_max}")
        for beta_max in beta_max_vec:
            filename = f"dsymm_Gsymm_{spur_name}_beta_max_{beta_max}"
            performance_plot_symm_MSE({"Gsymm": err_mean[beta_max]["Gsymm"],"dsymm":err_mean[0.95]["dsymm"]}, filename = f"{filename}_clean_4",lams = lams, save = True, outdir=outdir)
                                      

                
    return 


if __name__ == "__main__":
    main()