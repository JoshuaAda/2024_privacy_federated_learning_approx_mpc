"""
Date: 2023-09-27
Author: Lukas Lüken

This module bundles the functionalities necessary to implement an approximate. MPC, i.e. the approximation of a model predictive control by neural networks in pytorch. This includes the architecture of the neural networks, the storage of model weights and scaling factors as well as model parameters and the loading of these models based on the stored data. Also functionalities for the use of these approx. MPC in a closed-loop like the "make_step" function are included.

"""

# %% Imports
import json
from dataclasses import dataclass, asdict
from typing import Tuple
import torch
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
# %%
def plot_history(history):
    history["total_epochs"] = [idx for idx, epoch in enumerate(history["epochs"])]

    fig, ax = plt.subplots(1,1)
    ax.plot(history["total_epochs"],history["train_loss"],label="train loss")
    ax.plot(history["total_epochs"],history["val_loss"],label="val loss")
    ax.set_xlabel("Total Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    # set ticks for total epochs as int
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # set x_lim to total epochs
    ax.set_xlim([0,history["total_epochs"][-1]])
    # grid
    ax.grid(True)

    return fig, ax

@dataclass
class ApproxMPCSettings:


    def __init__(self,lb_x=torch.tensor([0,0]),ub_x=torch.tensor([10,10]),lb_u=torch.tensor([-10,-10]),ub_u=torch.tensor([10,10]),
                 n_in=4,n_out=2,n_layers=2,n_neurons=200,act_fn='relu',
                 input_vars=("x1_0", "x2_0", "u_prev_0"),output_vars=("u0", "u1"),scaling_mode="bounds"):
        self.lb_x=lb_x
        self.ub_x=ub_x
        self.lb_u=lb_u
        self.ub_u=ub_u
        """Settings for the approximate MPC.

            """
        # dimensions
        self.n_in=n_in
        self.n_out=n_out

        # NN architecture
        self.n_layers=n_layers
        self.n_neurons=n_neurons
        self.act_fn=act_fn

        # Input variables
        self.input_vars=input_vars
        self.output_vars=output_vars

        # Scaling
        self.scaling_mode=scaling_mode

        # Bounds

        # lb_x1: float = 0.0
        # ub_x1: float = 10.0
        # lb_x2: float = 0.0
        # ub_x2: float = 10.0
        # lb_u1: float = -10.0
        # ub_u1: float = 10.0
        # lb_u2: float = -10.0
        # ub_u2: float = 10.0


    def to_dict(self):
        return asdict(self)
    
    # function to save settings after initialization as json
    def save_settings(self, folder_path=None, file_name="approx_MPC_settings"):
        settings_dict = self.to_dict()
        # make all tensors to np arrays to lists
        for k,v in settings_dict.items():
            if isinstance(v,torch.Tensor):
                settings_dict[k] = v.cpu().numpy().tolist()
        if folder_path is None:
            save_pth = Path(file_name+".json")
        else:
            save_pth = Path(folder_path,file_name+".json")        
        with open(save_pth,"w") as f:
            json.dump(settings_dict,f,indent=4)
        print("settings saved to: ", save_pth)

    @classmethod
    def from_dict(cls, settings_dict):
        return cls(**settings_dict)
    
    @classmethod
    def from_json(cls, folder_pth=None, file_name="approx_MPC_settings"):
        if folder_pth is None:
            load_pth = Path(file_name+".json")
        else:
            load_pth = Path(folder_pth,file_name+".json")
        with open(load_pth,"r") as f:
            settings_dict = json.load(f)
        return cls.from_dict(settings_dict)


# approx. mpc class
class ApproxMPC():
    """
    Class for the implementation of an approximate MPC. This includes the architecture of the neural networks, the storage of model weights and scaling factors as well as model parameters and the loading of these models based on the stored data. Also functionalities for the use of these approx. MPC in a closed-loop like the "make_step" function are included.

    """
    def __init__(self,settings):
        self.settings = settings

        self.torch_data_type = torch.float64

        # initialize neural network
        self.init_ann()

        # initialize scaling factors from bounds
        if self.settings.scaling_mode == "bounds":
            self.set_scaling_from_bounds()

        # initialize device
        self.set_device()

    def init_ann(self):
        """Initialize the neural network architecture.
        """
        n_layers = self.settings.n_layers
        n_neurons = self.settings.n_neurons
        n_in = self.settings.n_in
        n_out = self.settings.n_out
        act_fn = self.settings.act_fn
        self.ann = self.generate_ffNN(n_layers,n_neurons,n_in,n_out,act_fn)
        
        if hasattr(self,"torch_data_type"):
            self.ann.to(self.torch_data_type)

        print("Neural network initialized with architecture: ", self.ann)
        # print number of nn parameters
        n_params = sum(p.numel() for p in self.ann.parameters() if p.requires_grad)
        print("---------------------------------")
        print("Number of trainable parameters: ", n_params)
        print("---------------------------------")
        self.n_params = n_params
    
    def set_device(self,device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.ann.to(device)
        # all scaling params to device
        self.x_range = self.x_range.to(device)
        self.x_shift = self.x_shift.to(device)
        self.y_range = self.y_range.to(device)
        self.y_shift = self.y_shift.to(device)

    @staticmethod
    def generate_ffNN(n_layers,n_neurons,n_in,n_out,act_fn='relu'):
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(torch.nn.Linear(n_in,n_neurons))
            else:
                layers.append(torch.nn.Linear(n_neurons,n_neurons))
            if act_fn == 'relu':
                layers.append(torch.nn.ReLU())
            elif act_fn == 'tanh':
                layers.append(torch.nn.Tanh())
            else:
                raise ValueError('act_fn must be either relu or tanh')
        ann = torch.nn.Sequential(*layers,torch.nn.Linear(n_neurons,n_out))
        return ann

    def set_scaling_from_bounds(self):
        x_min = torch.hstack((self.settings.lb_x,self.settings.lb_u))
        x_max = torch.hstack((self.settings.ub_x,self.settings.ub_u))
        x_range = x_max-x_min
        x_shift = x_min

        y_min = self.settings.lb_u
        y_max = self.settings.ub_u
        y_range = y_max-y_min
        y_shift = y_min

        self.x_range = x_range.to(self.torch_data_type)
        self.x_shift = x_shift.to(self.torch_data_type)
        self.y_range = y_range.to(self.torch_data_type)
        self.y_shift = y_shift.to(self.torch_data_type)

        print("Scaling factors set from data with mode: ", "bounds")
        print("x_range: ", self.x_range)
        print("x_shift: ", self.x_shift)
        print("y_range: ", self.y_range)
        print("y_shift: ", self.y_shift)

        self.scaling_mode = "bounds"

    def scale_inputs(self,x_data):
        assert self.scaling_mode == "bounds"
        x_scaled = (x_data-self.x_shift)/self.x_range
        return x_scaled

    def scale_outputs(self,y_data):
        assert self.scaling_mode == "bounds"
        y_scaled = (y_data-self.y_shift)/self.y_range
        return y_scaled

    def rescale_inputs(self,x_scaled):
        assert self.scaling_mode == "bounds"
        x_data = x_scaled*self.x_range+self.x_shift
        return x_data

    def rescale_outputs(self,y_scaled):
        assert self.scaling_mode == "bounds"
        y_data = y_scaled*self.y_range+self.y_shift
        return y_data

    def scale_dataset(self,x_data,y_data):
        x_scaled = self.scale_inputs(x_data)
        y_scaled = self.scale_outputs(y_data)
        return x_scaled, y_scaled

    ### Loading and saving
    def save_model(self,folder_path=None,file_name="approx_MPC_state_dict"):
        if folder_path is None:
            save_pth = Path(file_name+".pt")
        else:
            save_pth = Path(folder_path,file_name+".pt")
        torch.save(self.ann.state_dict(),save_pth)
        print("model saved to: ", save_pth)
    
    def save_model_settings(self,folder_path=None,file_name="approx_MPC_settings"):
        self.settings.save_settings(folder_path,file_name)

    # Use this method to load model parameters
    def load_state_dict(self,folder_pth=None,file_name="approx_MPC_state_dict"):
        if folder_pth is None:
            load_pth = Path(file_name+".pt")
        else:
            load_pth = Path(folder_pth,file_name+".pt")
        self.ann.load_state_dict(torch.load(load_pth))
        print("model loaded from: ", load_pth)

    ### Application as approx. MPC
    def make_step(self,x,scale_inputs=True,rescale_outputs=True, clip_outputs=True):
        """Make one step with the approximate MPC.
        Args:
            x (torch.tensor): Input tensor of shape (n_in,).
            scale_inputs (bool, optional): Whether to scale the inputs. Defaults to True.
            rescale_outputs (bool, optional): Whether to rescale the outputs. Defaults to True.
        Returns:
            np.array: Array of shape (n_out,).
        """

        # Check if inputs are tensors
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x,dtype=self.torch_data_type)

        if scale_inputs:
            x_scaled = self.scale_inputs(x)
        else:
            x_scaled = x
        with torch.no_grad():
            y_scaled = self.ann(x_scaled)
        if rescale_outputs:
            y = self.rescale_outputs(y_scaled)
        else:
            y = y_scaled        
        # Clip outputs to satisfy input constraints of MPC
        if clip_outputs:
            y = torch.clamp(y,self.settings.lb_u,self.settings.ub_u)
        return y.cpu().numpy()

    ### From here on, code quite independent on approximate MPC: consider moving to separate module, e.g. "Trainer"
    ## Training
    def train_step(self,optim,x,y):
        optim.zero_grad()            
        y_pred = self.ann(x)            
        loss = torch.nn.functional.mse_loss(y_pred,y)
        loss.backward()            
        optim.step()            
        return loss.item()
    
    def train_epoch(self,optim,train_loader):
        train_loss = 0.0
        # Training Steps
        for idx_train_batch, batch in enumerate(train_loader):
            x, y = batch
            loss = self.train_step(optim,x,y)         
            train_loss += loss
        n_train_steps = idx_train_batch+1
        train_loss = train_loss/n_train_steps
        return train_loss

    def validation_step(self,x,y):
        with torch.no_grad():
            y_pred = self.ann(x)
            loss=torch.nn.functional.mse_loss(y_pred,y)
        return loss.item()
    
    def validation_epoch(self,val_loader):
        val_loss = 0.0
        for idx_val_batch, batch in enumerate(val_loader):
            x_val, y_val = batch
            loss = self.validation_step(x_val,y_val)
            val_loss += loss
        n_val_steps = idx_val_batch+1
        val_loss = val_loss/n_val_steps
        return val_loss

    def train(self,N_epochs,optim,train_loader,val_loader=None,history=None,verbose=True):
        if history is None:
            history = {"epochs": [], "train_loss": [], "val_loss": []}
        for epoch in range(N_epochs):
            
            # Training
            train_loss = self.train_epoch(optim,train_loader)
            history["epochs"].append(epoch)
            history["train_loss"].append(train_loss)
            if verbose:
                print("Epoch: ",history["epochs"][-1])
                print("Train loss: ",history["train_loss"][-1])
            
            # Validation
            if val_loader is not None:
                val_loss = self.validation_epoch(val_loader)
                history["val_loss"].append(val_loss)
                if verbose:
                    print("Val loss: ",history["val_loss"][-1])
        return history

    def compute_grad_update(self,old_model, new_model, device=None):
        # maybe later to implement on selected layers/parameters
        if device:
            old_model, new_model = old_model.to(device), new_model.to(device)
        return [(new_param.data - old_param.data) for old_param, new_param in
                zip(old_model.parameters(), new_model.parameters())]

    def add_update_to_model(self,model, update, weight=1.0, device=None):
        if not update: return model
        if device:
            model = model.to(device)
            update = [param.to(device) for param in update]

        for param_model, param_update in zip(model.parameters(), update):
            param_model.data += weight * param_update.data
        return model

    def add_gradient_updates(self,grad_update_1, grad_update_2, weight=1.0):
        assert len(grad_update_1) == len(
            grad_update_2), "Lengths of the two grad_updates not equal"

        for param_1, param_2 in zip(grad_update_1, grad_update_2):
            param_1.data += param_2.data * weight

    def add_weights(self,grad_update_1, grad_update_2, weight=1.0):
        assert len(grad_update_1) == len(
            grad_update_2), "Lengths of the two grad_updates not equal"

        for param_1, param_2 in zip(grad_update_1, grad_update_2):
            grad_update_1[param_1].data += grad_update_2[param_2].data * weight
    def flatten(self,grad_update):
        return torch.cat([update.data.view(-1) for update in grad_update])

    def unflatten(self,flattened, normal_shape):
        grad_update = []
        for param in normal_shape:
            n_params = len(param.view(-1))
            grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
            flattened = flattened[n_params:]

        return grad_update

    def compute_distance_percentage(self,model, ref_model):
        percents, dists = [], []
        for layer, ref_layer in zip(model.parameters(), ref_model.parameters()):
            dist = torch.norm(layer - ref_layer)
            dists.append(dist.item())
            percents.append((torch.div(dist, torch.norm(ref_layer))).item())

        return percents, dists

    def cosine_similarity(self,grad1, grad2, normalized=False):
        """
        Input: two sets of gradients of the same shape
        Output range: [-1, 1]
        """

        cos_sim = F.cosine_similarity(self.flatten(grad1), self.flatten(grad2), 0, 1e-10)
        if normalized:
            return (cos_sim + 1) / 2.0
        else:
            return cos_sim
