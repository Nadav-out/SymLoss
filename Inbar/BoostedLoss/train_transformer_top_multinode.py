import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch.nn as nn
from argparse import ArgumentParser
import h5py as h5
# import hdf5plugin
import math
import time
import random
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import (
    DistributedSampler,
)  # Distribute data across multiple gpus
from torch.distributed import init_process_group, destroy_process_group

from SymmLoss import create_transformation_dict, symm_loss_scalar7

# where are things going wrong?
# torch.autograd.set_detect_anomaly(True)

# define the transfromer model
class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_size):
        super().__init__()

        self.embed = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
        )

        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        self.encoder_layer_3 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        
        
        self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
        )
    
    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)
        x = torch.mean(x*mask.unsqueeze(-1), axis=1)
        return self.classifier(x)

class JetDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file

        with h5.File(self.h5_file, 'r') as hf:
            self.length = hf['pid'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5.File(self.h5_file, 'r') as data:
                X = data["data"][idx,:,:7]
                Y = data["pid"][idx]
                W = data["weights"][idx]
                jet = data["jet"][idx,-4:]
                mask = np.all(np.abs(X) != 0, axis=1)
                X_jet = jet[None,:]
                X_cart = data["data"][idx,:,7:]
        
        return X.astype(np.float32), np.array(Y, dtype=np.float32), mask, W.astype(np.float32), X_cart.astype(np.float32), X_jet.astype(np.float32)

def to_cylindrical(four_vec, log=True):
    E = four_vec[:,:,0]
    px = four_vec[:,:,1]
    py = four_vec[:,:,2]
    pz = four_vec[:,:,3]
    pt = torch.sqrt(px*px + py*py)
    phi = torch.arctan2(py,px)
    eta = torch.arcsinh(pz/pt)

    if log:
        cylindrical_four_vec = torch.cat([
            torch.log(E.unsqueeze(-1)),
            torch.log(pt.unsqueeze(-1)), 
            eta.unsqueeze(-1),
            phi.unsqueeze(-1)
        ], axis=2)

        cylindrical_four_vec = torch.where(cylindrical_four_vec < -1e30, 0, cylindrical_four_vec)
    else:
        cylindrical_four_vec = torch.cat([E.unsqueeze(-1),pt.unsqueeze(-1), eta.unsqueeze(-1),phi.unsqueeze(-1)], axis=2)

    
    return torch.nan_to_num(cylindrical_four_vec)

def get_jet_relvars(four_vec, four_vec_cy, jet_four_vec_cy):
    
    pi = torch.tensor(math.pi)
    # log(E)
    log_E = torch.log(four_vec_cy[:,:,0])
    
    # log(pT)
    log_pt = torch.log(four_vec_cy[:,:,1])
    
    # log(E_const/E_jet)
    log_Er = torch.log((four_vec_cy[:,:,0]/jet_four_vec_cy[:,:,0]))

    # log(pt_const/pt_jet)
    log_ptr = torch.log((four_vec_cy[:,:,1]/jet_four_vec_cy[:,:,1]))

    # dEta
    dEta =  four_vec_cy[:,:,2] - jet_four_vec_cy[:,:,2]

    # dPhi
    dPhi =  four_vec_cy[:,:,3] - jet_four_vec_cy[:,:,3] 
    dPhi[dPhi>pi] -=  2*pi
    dPhi[dPhi<= - pi] +=  2*pi

    # dR
    dR = torch.sqrt(dEta**2 + dPhi**2)

    
    # order of features [dEta, dPhi, log_R_pt, log_Pt, log_R_E, log_E, dR]
    jet_features = torch.cat([
        dEta.unsqueeze(-1), 
        dPhi.unsqueeze(-1),
        log_ptr.unsqueeze(-1),
        log_pt.unsqueeze(-1),
        log_Er.unsqueeze(-1),
        log_E.unsqueeze(-1),
        dR.unsqueeze(-1)
    ], axis=2)

    zero_mask = (four_vec == 0.0).any(dim=-1, keepdim=True)
    zero_mask = zero_mask.expand_as(jet_features)
    jet_features[zero_mask] = 0.0
    
    return jet_features
    
def get_data(path, training=False):
    
    def shuffle(a ,b ,c, d, e):
        idx = np.random.permutation(len(a))
        return a[idx], b[idx], c[idx], d[idx], e[idx]
    

    data =  h5.File(path, 'r')
    print("Sucessfully opened h5 file...")

    X = data["data"]
    Y = data["pid"]
    w = data["weights"]
    X_jet = data["jet"][:,-4:]
    masks = np.all(np.abs(X) != 0, axis=2)
    X_jet = X_jet[:,None,:]

    print("Collected variables...")
    del data

    if training:
        X, Y, w, mask, X_jet = shuffle(X, Y, w, mask,X_jet)

    return X, Y, w, mask, X_jet 

def boost(data, device="cpu"):
    beta = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    # beta = np.repeat(beta, data.shape[1], axis=None)
    beta = beta.repeat(data.shape[1])
    gamma = (1-beta*beta)**(-0.5)

    beta = beta.to(device)
    gamma = gamma.to(device)

    E_b = gamma*(data[:,:,0].flatten()- beta* data[:,:,1].flatten() )
    px_b = gamma*(data[:,:,1].flatten() - beta* data[:,:,0].flatten())
    
    E_b = E_b.reshape(data.shape[0], data.shape[1])
    px_b = px_b.reshape(data.shape[0], data.shape[1])

    return  torch.cat([ E_b.unsqueeze(-1), px_b.unsqueeze(-1), data[:,:,2:]], axis = 2)

def boost_3d(data,jet_data, device="cpu", beta=None):

    # sample beta from sphere
    b1 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    b2 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    theta = 2 * np.pi * b1
    phi = np.arccos(1 - 2 * b2)
    
    beta_x = np.sin(phi) * np.cos(theta)
    beta_y = np.sin(phi) * np.sin(theta)
    beta_z = np.cos(phi)
    
    beta = torch.cat([beta_x.unsqueeze(-1),beta_y.unsqueeze(-1), beta_z.unsqueeze(-1)], axis=1)
    bf = torch.tensor(np.random.uniform(0, 0.95, size=(len(data),1)), dtype=torch.float32)
    bf = bf**(1/2)
    beta = beta*bf
    
    beta_norm = torch.norm(beta, dim=1) 

    # make sure we arent violating speed of light
    assert torch.all(beta_norm < 1)

    gamma = 1 / torch.sqrt(1 - (beta_norm)**2)

    beta_squared = (beta_norm)**2

    # make boost matrix
    L = torch.zeros((len(data), 4, 4)).to(device)
    L[:,0, 0] = gamma
    L[:,1:, 0] = L[:,0, 1:] = -gamma.unsqueeze(-1) * beta
    L[:, 1:, 1:] = torch.eye(3) + (gamma[...,None, None] - 1) * torch.einsum('bi,bj->bij', (beta, beta))/ beta_squared[...,None, None]
    
    assert torch.all (torch.linalg.det(L)) == True

    boosted_four_vector = torch.einsum('bij,bkj->bik', L.type(torch.float32), data.type(torch.float32)).permute(0, 2, 1) 
    boosted_jet_vector = torch.einsum('bij,bkj->bik', L.type(torch.float32), jet_data.type(torch.float32)).permute(0, 2, 1) 

    # Validate that energy values remain non-negative
    assert torch.all(boosted_four_vector[:, :, 0] >= 0), "Negative energy values detected in constituents!"
    assert torch.all(boosted_jet_vector[:, :, 0] >= 0), "Negative energy values detected in jets!"
    
    return boosted_four_vector, boosted_jet_vector


def sum_reduce(num, device):
    r''' Sum the tensor across the devices.
    '''
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def train_step(model, dataloader, cost, optimizer, epoch, device, penalty=None, boost_=boost, apply_penalty=False):
    model.train()
    running_loss = 0.0
    rbce = 0.0
    rmse = 0.0
    rsymm = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch: {epoch}", mininterval=10)):
        
        batch = [x.to(device) for x in batch]
        X, y, mask, weights, X_cartesian, jet_vec = batch
        
        # boosted variables
        X_boost, jet_boost  = boost_3d(X_cartesian, jet_vec, device=device)
        X_boost_cy = to_cylindrical(X_boost, log=False)
        jet_boost_cy = to_cylindrical(jet_boost, log=False)
        boost_jet_vars = get_jet_relvars(X_boost, X_boost_cy, jet_boost_cy)

        optimizer.zero_grad()  # Zero the gradients

        outputs = model(X, mask)  # Forward pass
        outputs_boost = model(boost_jet_vars, mask)

        # catch NaNs
        output_nan = torch.sum(torch.isnan(outputs_boost))

        if output_nan > 0:
            print(f"Nan found in output in batch: {batch_idx}, Nans: {output_nan}")
            outputs_boost = torch.nan_to_num(outputs_boost, nan=0.5)
        
        bce = cost(outputs.squeeze(), y) * weights
        mse = penalty(outputs.squeeze(), outputs_boost.squeeze()) * weights
        symm = symm_loss_scalar7(model, X, X_cartesian, jet_vec, mask=mask) * weights

        # if device ==0:
        #     print(outputs.squeeze())
       
        if apply_penalty:
            loss =  bce.mean() + mse.mean()
        else:
            loss = bce.mean()

        loss.backward()  # Backward pass
        
        #take care of unruly gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        
        optimizer.step()  # Update parameters

        running_loss += loss.item()
        rbce += bce.mean().item()
        rmse += mse.mean().item()
        rsymm += symm.mean().item()


    # running_loss /=len(dataloader)
    # rbce /= len(dataloader)
    # rmse /= len(dataloader)

    dist.barrier()
    distributed_batch = sum_reduce(len(dataloader), device=device).item()
    distributed_loss = sum_reduce(running_loss, device=device).item()/distributed_batch
    distributed_bce = sum_reduce(rbce, device=device).item()/distributed_batch
    distributed_mse = sum_reduce(rmse, device=device).item()/distributed_batch
    distributed_symm = sum_reduce(rsymm, device=device).item()/distributed_batch

    return distributed_loss, distributed_bce, distributed_mse, distributed_symm

def test_step(model, dataloader, cost, epoch, device, penalty=None, boost_=boost, apply_penalty=False):
    model.eval()
    running_loss = 0.0
    rbce = 0.0
    rmse = 0.0
    rsymm = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch: {epoch}", mininterval=10)):
        
            
        batch = [x.to(device) for x in batch]
        X, y, mask, weights, X_cartesian, jet_vec = batch
        
        # boosted variables
        X_boost, jet_boost  = boost_3d(X_cartesian, jet_vec, device=device)
        X_boost_cy = to_cylindrical(X_boost, log=False)
        jet_boost_cy = to_cylindrical(jet_boost, log=False)
        boost_jet_vars = get_jet_relvars(X_boost, X_boost_cy, jet_boost_cy)
        symm = symm_loss_scalar7(model, X, X_cartesian, jet_vec, mask=mask) * weights
    
        with torch.no_grad():
            outputs = model(X, mask)  # Forward pass
            outputs_boost = model(boost_jet_vars, mask)
            # catch NaNs
            output_nan = torch.sum(torch.isnan(outputs_boost))

            if output_nan > 0:
                print(f"Nan found in output in batch: {batch_idx}, Nans: {output_nan}")
                outputs_boost = torch.nan_to_num(outputs_boost, nan=0.5)
            
            bce = cost(outputs.squeeze(), y) * weights
            mse = penalty(outputs.squeeze(), outputs_boost.squeeze()) * weights
            
            
            if apply_penalty:
                loss =  bce.mean() + mse.mean()
            else:
                loss = bce.mean()

            running_loss += loss.item()
            rbce += bce.mean().item()
            rmse += mse.mean().item()
            rsymm += symm.mean().item()


    # running_loss /=len(dataloader)
    # rbce /= len(dataloader)
    # rmse /= len(dataloader)

    dist.barrier()
    distributed_batch = sum_reduce(len(dataloader), device=device).item()
    distributed_loss = sum_reduce(running_loss, device=device).item()/distributed_batch
    distributed_bce = sum_reduce(rbce, device=device).item()/distributed_batch
    distributed_mse = sum_reduce(rmse, device=device).item()/distributed_batch
    distributed_symm = sum_reduce(rsymm, device=device).item()/distributed_batch

    return distributed_loss, distributed_bce, distributed_mse, distributed_symm

def evaluate(model, loader, device):
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    pred = []
    boost_ = []
    true = []

    with torch.no_grad():  # Disable gradient calculation
        # for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        for batch in loader:

            X, y, mask, weights, X_cartesian, jet_vec = batch
            X = X.to(device)
            y = y.to(device)
            X_cartesian = X_cartesian.to(device)
            mask = mask.to(device)
            weights = weights.to(device)
            jet_vec = jet_vec.to(device)

            X_boost, jet_boost  = boost_3d(X_cartesian, jet_vec, device=device)
            X_boost_cy = to_cylindrical(X_boost, log=False)
            jet_boost_cy = to_cylindrical(jet_boost, log=False)
            boost_jet_vars = get_jet_relvars(X_boost, X_boost_cy, jet_boost_cy)

            outputs = model(X, mask)  # Forward pass
            outputs_boost = model(boost_jet_vars, mask)

            pred.append(outputs)
            boost_.append(outputs_boost)
            true.append(y)

    # model.to("cpu")
    
    return torch.cat(pred), torch.cat(boost_), torch.cat(true)

def train_model(model, train_loader, test_loader, loss, optimizer, train_sampler, num_epochs=100, device='cpu', global_rank=0,patience=5, penalty=None, output_dir="", boost_=boost, apply_penalty=False, save_tag=""):
    print(f"Process ID: {device}")
    pen_tag = "_mse" if apply_penalty else ""
    boost_tag = "3d" if boost_ is boost_3d else "1d"
    model_save = f"best_model_{boost_tag}_boost_{pen_tag}_{save_tag}.pt"

    print(f"Saving model as: {model_save}")

    losses = {
        "train_loss": [],
        "train_BCE": [],
        "train_MSE": [],
        "train_symm": [],
        "val_loss": [],
        "val_BCE": [],
        "val_MSE": [],
        "val_symm": []
    }
   
    tracker = {
        "bestValLoss": np.inf,
        "bestEpoch": 0
    }

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        train_losses = train_step(model, train_loader, loss, optimizer, epoch, device, penalty, boost_, apply_penalty)
        val_losses = test_step(model, test_loader, loss, epoch, device, penalty, boost_, apply_penalty)
        
        losses["train_loss"].append(train_losses[0])
        losses["train_BCE"].append(train_losses[1])
        losses["train_MSE"].append(train_losses[2])
        losses["train_symm"].append(train_losses[3])

        losses["val_loss"].append(val_losses[0])
        losses["val_BCE"].append(val_losses[1])
        losses["val_MSE"].append(val_losses[2])
        losses["val_symm"].append(val_losses[3])


        if device == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {losses["train_loss"][-1]:.4f}, Val Loss: {losses["val_loss"][-1]:.4f},  BCE: {losses["train_BCE"][-1]:.4f}, Val BCE: {losses["val_BCE"][-1]:.4f}, MSE: {losses["train_MSE"][-1]:.4f}, Val MSE: {losses["val_MSE"][-1]:.4f}, Symm: {losses["train_symm"][-1]:.4f}, Val Symm: {losses["val_symm"][-1]:.4f} ')

        if losses["val_loss"][-1] < tracker["bestValLoss"]:
                tracker["bestValLoss"] = losses["val_loss"][-1]
                tracker["bestEpoch"] = epoch
                
                dist.barrier()

                if global_rank==0:
                    torch.save(
                        model.module.state_dict(), f"{output_dir}/{model_save}"
                    )

                    # torch.save(
                    #     model, f"{output_dir}/{model_save}"
                    # )

        dist.barrier() # syncronise (top GPU is doing more work)

        # check the validation loss from each GPU:
        debug = False 
        if debug:
            print(f"Rank: {global_rank}, Device: {device}, Train Loss: {losses['train_loss'][-1]:.5f}, Validation Loss: {losses['val_loss'][-1]:.5f}")
            print(f"Rank: {global_rank}, Device: {device}, Best Loss: {tracker['bestValLoss']}, Best Epoch: {tracker['bestEpoch']}")
        # early stopping check
        if epoch - tracker["bestEpoch"] > patience:
            print(f"breaking on Rank: {global_rank}, device: {device}")
            break
        
    if global_rank==0:
        print(f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!")
    
        # save losses
        json.dump(losses, open(f"{output_dir}/training_{boost_tag}{pen_tag}_{save_tag}.json", "w"))

        

# Each process control a single gpu
def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8870"  # select any idle port on your machine

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(world_size, global_rank, rank, args=None):
    # ddp_setup(rank, world_size)

    # set random seeds
    np.random.seed(int(args.save_tag))
    torch.manual_seed(int(args.save_tag))
    random.seed(int(args.save_tag))


    # make dataset
    train_data = JetDataset(f"{args.data_dir}/train_atlas_symmetry.h5")
    test_data = JetDataset(f"{args.data_dir}/val_atlas_symmetry.h5")

    # distributed loader
    sampler_train = DistributedSampler(train_data, shuffle=True, num_replicas=world_size, rank=global_rank)
    sampler_test = DistributedSampler(test_data, shuffle=False, num_replicas=world_size, rank=global_rank)
    
    # make dataloader
    train_loader = DataLoader(
            train_data, 
            batch_size=256*4,
            shuffle=False,
            sampler=sampler_train,
            num_workers=os.cpu_count()//world_size,
            pin_memory=True,
            # prefetch_factor=4
            )
    test_loader = DataLoader(
        test_data,
        batch_size=256*4,
        shuffle=False,
        sampler=sampler_test,
        num_workers=os.cpu_count()//world_size,
        pin_memory=True,
        # prefetch_factor=4

        )

    

    # set up model
    model = Transformer(input_dim=7, embed_dim=256, hidden_size=128)
    model = DDP(model.to(rank), device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    if rank==0:
        d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {d} Rank: {global_rank}")

    # train
    BCE = nn.BCELoss(reduction='none')
    MSE = nn.MSELoss(reduction='none')
    boost_dict = {
        "1D": boost,
        "3D": boost_3d
    }
    train_model(
        model, 
        train_loader, 
        test_loader, 
        BCE, 
        optimizer, 
        train_sampler=sampler_train, 
        device=rank,
        global_rank=global_rank, 
        penalty=MSE, 
        output_dir=args.outdir, 
        boost_=boost_dict[args.boost_type],
        apply_penalty=args.apply_penalty,
        save_tag=args.save_tag
        )

    dist.barrier()

    if args.run_eval and global_rank == 0:
        
        pen_tag = "_mse" if args.apply_penalty else ""
        boost_tag = "3d" if args.boost_type == "3D" else "1d"
        
        model_save = f"best_model_{boost_tag}_boost_{pen_tag}_{args.save_tag}.pt"

        model2 = Transformer(input_dim=7, embed_dim=256, hidden_size=128)
        model2.load_state_dict(torch.load(f"{args.outdir}/{model_save}"))

        # model2 = torch.load(model_save)
        
        print("Evaluating Loaded Model...")
        preds, _, trues = evaluate(model2, test_loader, rank)
        
        plt.figure()
        plt.hist(preds[torch.where(trues==0)].cpu().flatten().numpy(), color="orange", histtype="step")
        plt.hist(preds[torch.where(trues==1)].cpu().flatten().numpy(), color="blue", histtype="step")
        plt.savefig("Evaluation_Separation_Load_full.png", dpi=100)
    
    dist.barrier()
    # destroy_process_group()

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        default="",
        help="Directory of training and validation data"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="outdir",
        default="",
        help="Directory to output best model",
    )
    parser.add_argument(
        "--save_tag",
        dest="save_tag",
        default="",
        help="Extra tag for checkpoint model",
    )
    parser.add_argument(
        "--apply_penalty",
        dest="apply_penalty",
        default=True,
        action="store_true"
    )
    parser.add_argument(
        "--boost_type",
        dest="boost_type",
        default="1D",
        choices=["1D", "3D"]
    )
    parser.add_argument(
        "--run_eval",
        dest="run_eval",
        default=False,
        action="store_true"
    )

    
    args = parser.parse_args()
    world_size = int(os.environ['WORLD_SIZE'])
    torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
    global_rank = torch.distributed.get_rank()
    rank = int(os.environ['LOCAL_RANK'])
    
    main(world_size, global_rank, rank, args)
    # mp.spawn(
    #     main,
    #     args=(world_size, args),
    #     nprocs=world_size,
    # ) 