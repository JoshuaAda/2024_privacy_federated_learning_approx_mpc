"""
Date: 2023-09-27
Author: Lukas Lüken

Pytorch script to load some data and train a feedforward neural network to approximate the MPC controller.
"""
import copy

# %% Imports
import torch
from pathlib import Path
from do_mpc.approximateMPC.approx_MPC import ApproxMPC, Trainer, FeedforwardNN
from approx_MPC import plot_history
import json
import numpy as np
import matplotlib.pyplot as plt

# %%
# Setup
seed = 0
torch.manual_seed(seed)
# np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#device='cpu'
torch_data_type = torch.float32
torch.set_default_dtype(torch_data_type)
file_pth = Path(__file__).parent.resolve()
print("Filepath: ",file_pth)

# %%
# Config
#########################################################
# Data
dataset='CSTR'

if dataset=='DI':
    train_data_file_name = "DI_dataset_train_100.pt"
    train_data_folder = file_pth.joinpath('datasets')
    val_data_file_name = "DI_dataset_val_100.pt"
    val_data_folder = file_pth.joinpath('datasets')
    # NN
    n_layers = 3  # (L = n+1)
    n_neurons = 200
    n_in = 3
    n_out = 1
elif dataset=='NDI':
    train_data_file_name = "NDI_dataset_train_100.pt"
    train_data_folder = file_pth.joinpath('datasets')
    val_data_file_name = "NDI_dataset_val_100.pt"
    val_data_folder = file_pth.joinpath('datasets')
    # NN
    n_layers = 3  # (L = n+1)
    n_neurons = 200
    n_in = 4
    n_out = 2
elif dataset=='CSTR':
    num_samples=200
    num_par=6
    files_name=["./sampling_"+str(k)+"/data_n"+str(num_samples)+"_opt_scaled.pth" for k in range(num_par)]
    train_data_file_name = "CSTR_dataset_train_100.pt"
    train_data_folder = file_pth.joinpath('datasets')
    val_data_file_name = "CSTR_dataset_val_100.pt"
    val_data_folder = file_pth.joinpath('datasets')
    # NN
    n_layers = 3
    n_neurons = 500
    n_in = 7
    n_out = 2
else:
    train_data_file_name = "BR_dataset_train_100.pt"
    train_data_folder = file_pth.joinpath('datasets')
    val_data_file_name = "BR_dataset_val_100.pt"
    val_data_folder = file_pth.joinpath('datasets')
    # NN
    n_layers = 5  # (L = n+1)
    n_neurons = 200
    n_in = 8
    n_out = 1




# NN Training
N_epochs = 200
batch_size = 200
lrs = [1e-2,1e-3,1e-4,1e-5]
overfit_batch = False # Default: False; Used to determine wether NN size is large enough and code is working w.o. bugs
N_part=6#5
non_uniform=False
local=False
verbose=True
#########################################################

# %%
# Data loading

# Load data.
if dataset=='CSTR':
    print("loading data...")
    X_train_scaled = list()
    Y_train_scaled = list()
    X_val_scaled= torch.tensor([])
    Y_val_scaled= torch.tensor([])
    for k,file_name in enumerate(files_name):
        data=torch.load(file_name,map_location=device)
        num_values=len(data.tensors[0])
        train_idx=4000#int(0.8*num_values)
        X_train_scaled.append(data.tensors[0][:train_idx,:])
        if k==0:
            X_val_scaled=data.tensors[0][train_idx:,:]
            Y_val_scaled=data.tensors[1][train_idx:,:]
        else:
            X_val_scaled=torch.cat((X_val_scaled, data.tensors[0][train_idx:,:]), dim=0)
            Y_val_scaled=torch.cat((Y_val_scaled, data.tensors[1][train_idx:,:]), dim=0)
        Y_train_scaled.append(data.tensors[1][:train_idx,:])
    n_train = X_train_scaled[0].shape[0]
    N_steps = int(torch.ceil(torch.tensor(n_train / batch_size)))

else:
    print("Loading data...")
    data_train = torch.load(train_data_folder.joinpath(train_data_file_name),map_location=device)
    data_val   = torch.load(val_data_folder.joinpath(val_data_file_name),map_location=device)
    print("Data loaded.")
    part_size=int(np.floor(data_train[0].shape[0]/N_part))
    part_size=1000
    sizes=part_size*np.ones(N_part)
    if non_uniform:
        sizes[0]=10#sizes[0]/5
        sizes[1]=20#sizes[1]/2
        sizes[2]=50
        sizes[3]=130#sizes[3]*1.5
        sizes[4]=290#sizes[4]*1.8
    sep_idx=np.zeros(N_part+1)
    sep_idx[1:]=np.cumsum(sizes)
    # Train data
    #X_train,Y_train,_,_ = data_train.tensors
    X_train=list()
    Y_train=list()
    for k in range(N_part):
        X_train.append(data_train[0][int(sep_idx[k]):int(sep_idx[k+1]),:n_in])
        #X_train.append(data_train[0][210:500,:])
        if dataset in ['DI', 'NDI']:
         Y_train.append(data_train[1][int(sep_idx[k]):int(sep_idx[k+1]),:n_out])
        else:
          Y_train.append(data_train[1][int(sep_idx[k]):int(sep_idx[k + 1])])
        #Y_train.append(data_train[1][210:500,:])
        Y_train[k] = torch.squeeze(Y_train[k][:, None], dim=1)
    L=10
    color=['b','r','k','c','g']
    for k in range(N_part):
        for m in range(int(sizes[k])):
            plt.plot(X_train[k][m*L:(m+1)*L,0].detach().cpu().numpy(),X_train[k][m*L:(m+1)*L,1].detach().cpu().numpy(),color[k])
    plt.show(block=False)
    if overfit_batch:
        X_train = X_train[0][:batch_size,:]
        Y_train = Y_train[0][:batch_size,:]



    # Validation data
    #X_val,Y_val,_,_ = data_val.tensors
    X_val = data_val[0][:,:n_in]
    if dataset in ['DI','NDI']:
     Y_val = data_val[1][:,:n_out]
    else:
     Y_val = data_val[1]
    Y_val = torch.squeeze(Y_val[:,None],dim=1)
    plt.figure()
    for m in range(int(X_val.shape[0]/L)):
            plt.plot(X_val[m*L:(m+1)*L,0].detach().cpu().numpy(),X_val[m*L:(m+1)*L,1].detach().cpu().numpy(),color[0])
    plt.show(block=False)


    n_train = X_train[0].shape[0]
    N_steps = int(torch.ceil(torch.tensor(n_train/batch_size)))


# Print statistics: n_train, N_steps, batch_size
print("----------------------------------")
# %%
# # Setup approx. MPC
print("Setting up approx. MPC...")
settings=list()
approx_mpc=list()
for k in range(N_part):
    #settings.append( ApproxMPCSettings(n_in=n_in,n_out=n_out,n_layers=n_layers,n_neurons=n_neurons))
    if dataset=='DI':
        lb_x = torch.tensor([-25, -5])
        ub_x = torch.tensor([25, 5])
        lb_u = torch.tensor([-1])
        ub_u = torch.tensor([1])
    elif dataset=='NDI':
        lb_x = torch.tensor([0.0,0.0])
        ub_x = torch.tensor([10.0,10.0])
        lb_u = torch.tensor([-10.0,-5.0])
        ub_u = torch.tensor([10.0,5.0])
    elif dataset=="CSTR":
        lbx = np.array([[0.1], [0.1], [50], [50]])
        ubx = np.array([[2], [2], [140], [140]])
        lbu = np.array([[5], [-8500]])
        ubu = np.array([[100], [0]])
        lbp = np.array([[127]])
        ubp = np.array([[133]])
        lb = np.concatenate((lbx, lbu, lbp), axis=0)
        ub = np.concatenate((ubx, ubu, ubp), axis=0)
    else:
        max_s_val=torch.max(X_val[:,1])
        max_s_train=0
        for s in range(N_part):

          if torch.max(X_train[s][:,1])>max_s_train:
              max_s_train=torch.max(X_train[s][:,1])
        max_s=max(max_s_val,max_s_train)

        max_p_val = torch.max(X_val[:, 3])
        max_p_train = 0
        for s in range(N_part):

            if torch.max(X_train[s][:, 3]) > max_p_train:
                max_p_train = torch.max(X_train[s][:, 3])
        max_p = max(max_p_val, max_p_train)


        lb_x = torch.tensor([0.0,-0.01,0.0,0.0,1.5,100,2])
        ub_x = torch.tensor([7,max_s,6,max_p,6,300,7])
        lb_u = torch.tensor([0.0])
        ub_u = torch.tensor([0.2])
    if dataset=='CSTR':
        approx_mpc=list()
        net = FeedforwardNN(n_in=n_in, n_out=n_out, n_neurons=n_neurons)
        for k in range(num_par):
            approx_mpc_instance = ApproxMPC(copy.deepcopy(net))
            approx_mpc_instance.shift_from_box(lbu.T, ubu.T, lb.T, ub.T)
            approx_mpc.append(approx_mpc_instance)

    else:
        settings.append(ApproxMPCSettings(lb_x=lb_x, ub_x=ub_x, lb_u=lb_u, ub_u=ub_u,
                                     n_in=n_in, n_out=n_out, n_layers=n_layers, n_neurons=n_neurons))
        approx_mpc.append(ApproxMPC(settings=settings[k]))
        if k==0:
            state_dict=approx_mpc[0].ann.state_dict()
        else:
            approx_mpc[k].ann.load_state_dict(copy.deepcopy(state_dict))
        approx_mpc[k].set_device(device)
print("Approx. MPC setup complete.")

# %%
# approx_mpc=ApproxMPC(settings=ApproxMPCSettings(lb_x=lb_x, ub_x=ub_x, lb_u=lb_u, ub_u=ub_u,
#                                  n_in=n_in, n_out=n_out, n_layers=n_layers, n_neurons=n_neurons))

# Scale data and setup data loaders
if dataset=='CSTR':
    pass
else:
    print("Scaling data...")
    X_train_scaled=list()
    Y_train_scaled=list()
    for k in range(N_part):
        X_train_scaled1, Y_train_scaled1 = approx_mpc[0].scale_dataset(X_train[k],Y_train[k])
        X_train_scaled.append(X_train_scaled1)
        Y_train_scaled.append(Y_train_scaled1)
        Y_train_scaled[k] = Y_train_scaled[k].unsqueeze(dim=1)

    X_val_scaled, Y_val_scaled = approx_mpc[0].scale_dataset(X_val,Y_val)
    Y_val_scaled=Y_val_scaled.unsqueeze(dim=1)
    print("Data scaled.")


# torch data loader
train_dataset=list()
train_loader=list()
for k in range(N_part):
    train_dataset.append(torch.utils.data.TensorDataset(X_train_scaled[k], Y_train_scaled[k]))
    train_loader.append(torch.utils.data.DataLoader(train_dataset[k],batch_size=batch_size,shuffle=True))
val_dataset = torch.utils.data.TensorDataset(X_val_scaled, Y_val_scaled)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=X_val_scaled.shape[0],shuffle=False)


# %%
# Training
print("Training...")
histories=list()
figs=list()
params=list()

if local:

    for k in range(1):
       # seed = 0
       # torch.manual_seed(seed)
       # approx_mpc1 = ApproxMPC(settings=ApproxMPCSettings(lb_x=lb_x, ub_x=ub_x, lb_u=lb_u, ub_u=ub_u,
       #                                                     n_in=n_in, n_out=n_out, n_layers=n_layers,
       #                                                     n_neurons=n_neurons))
       #approx_mpc1.ann.load_state_dict(copy.deepcopy(approx_mpc.ann.state_dict()))
       if dataset=="CSTR":
           history_pt = {"epochs": [], "train_loss": [], "val_loss": []}
           optim_list=[torch.optim.AdamW(approx_mpc[k].net.parameters(), lr=1e-3) for k in range(len(approx_mpc))]
           train_list=[Trainer(approx_mpc=approx_mpc[k]) for k in range(len(approx_mpc))]
           lr_scheduler_patience = 100  # train_config["lr_scheduler_patience"]
           lr_scheduler_cooldown = 10  # train_config["lr_scheduler_cooldown"]
           lr_reduce_factor = 0.1  # train_config["lr_reduce_factor"]
           min_lr = 1e-8  # train_config["min_lr"]

           # Early Stopping
           early_stop = True  # train_config["early_stop"]

           lr_scheduler_list = [torch.optim.lr_scheduler.ReduceLROnPlateau(optim_list[k], mode='min', factor=lr_reduce_factor,
                                                                     patience=lr_scheduler_patience,
                                                                     threshold=1e-5, threshold_mode='rel',
                                                                     cooldown=lr_scheduler_cooldown, min_lr=min_lr,
                                                                     eps=0.0) for k in range(N_part)]

           for k in range(len(approx_mpc)):
            history_pt=train_list[k].train(N_epochs,optim_list[k],train_loader[k],lr_scheduler_list[k],val_loader)
            fig, ax = plot_history(history_pt)
            histories.append(history_pt)
            figs.append(fig)
            run_hparams = {"n_layers": n_layers, "n_neurons": n_neurons, "n_in": n_in,
                           "n_out": n_out, "N_epochs": N_epochs, "batch_size": batch_size,
                           "lrs": lrs, "overfit_batch": overfit_batch,
                           "train_data_file_name": train_data_file_name,
                           "val_data_file_name": val_data_file_name,
                           "n_train": n_train, "N_steps": N_steps,
                           "optimizer": str(optim_list[0].__class__.__name__),
                           "train_loss": history_pt["train_loss"][-1],
                           "val_loss": history_pt["val_loss"][-1]}
            params.append(run_hparams)
       else:
           history_pt = {"epochs": [], "train_loss": [], "val_loss": []}
           for idx_lr, lr in enumerate(lrs):
                # optim=torch.optim.SGD(approx_mpc[k].ann.parameters(),lr=lr)
                optim = torch.optim.AdamW(approx_mpc[k].ann.parameters(), lr=lr)
                # optim = torch.optim.Adam(approx_mpc.ann.parameters(),lr=lr)
                # train
                history_pt = approx_mpc[k].train(N_epochs, optim, train_loader[k], val_loader, history_pt, verbose=True)
                # plot history
           fig, ax = plot_history(history_pt)
           histories.append(history_pt)
           figs.append(fig)
           run_hparams = {"n_layers": n_layers, "n_neurons": n_neurons, "n_in": n_in,
                          "n_out": n_out, "N_epochs": N_epochs, "batch_size": batch_size,
                          "lrs": lrs, "overfit_batch": overfit_batch,
                          "train_data_file_name": train_data_file_name,
                          "val_data_file_name": val_data_file_name,
                          "n_train": n_train, "N_steps": N_steps,
                          "optimizer": str(optim.__class__.__name__),
                          "train_loss": history_pt["train_loss"][-1],
                          "val_loss": history_pt["val_loss"][-1]}
           params.append(run_hparams)


else:
    if dataset=="CSTR":
        optim_list = [torch.optim.AdamW(approx_mpc[k].net.parameters(), lr=1e-3) for k in range(len(approx_mpc))]
        train_list = [Trainer(approx_mpc=approx_mpc[k]) for k in range(len(approx_mpc))]
        lr_scheduler_patience = 40  # train_config["lr_scheduler_patience"]
        lr_scheduler_cooldown = 10  # train_config["lr_scheduler_cooldown"]
        lr_reduce_factor = 0.1  # train_config["lr_reduce_factor"]
        min_lr = 1e-8  # train_config["min_lr"]

        # Early Stopping
        early_stop = True  # train_config["early_stop"]

        lr_scheduler_list = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(optim_list[k], mode='min', factor=lr_reduce_factor,
                                                       patience=lr_scheduler_patience,
                                                       threshold=1e-5, threshold_mode='rel',
                                                       cooldown=lr_scheduler_cooldown, min_lr=min_lr,
                                                       eps=0.0) for k in range(N_part)]

        histories = list()
        for k in range(N_part):
            histories.append({"epochs": [], "train_loss": [], "val_loss": []})
        for epoch in range(N_epochs):
            gradients = list()
            for k in range(N_part):
                history = histories[k]
                # Training
                backup = copy.deepcopy(approx_mpc[k].net)
                train_loss = train_list[k].train_epoch(optim_list[k], train_loader[k])
                lr_scheduler_list[k].step(train_loss)
                gradient = train_list[k].compute_grad_update(old_model=backup, new_model=approx_mpc[k].net,
                                                             device=device)
                gradients.append(gradient)
                approx_mpc[k].net.load_state_dict(backup.state_dict())
                history["epochs"].append(epoch)
                history["train_loss"].append(train_loss)
                if verbose:
                    print("Epoch: ", history["epochs"][-1])
                    print("Train loss: ", history["train_loss"][-1])
                    print("Learning Rate: ", optim_list[-1].param_groups[0]["lr"])
            aggregated_gradient = [torch.zeros(param.shape).to(device) for param in approx_mpc[0].net.parameters()]

            # aggregate and update server model

            #weights = torch.div(torch.tensor(sizes), torch.sum(torch.tensor(sizes)))

            for gradient in gradients:
                train_list[0].add_gradient_updates(aggregated_gradient, gradient)
            for k in range(N_part):
                train_list[k].add_update_to_model(approx_mpc[k].net, aggregated_gradient)
                # Validation
                if val_loader is not None:
                    val_loss = train_list[k].validation_epoch(val_loader)
                    histories[k]["val_loss"].append(val_loss)
                    if verbose:
                        print("Val loss: ", histories[k]["val_loss"][-1])
    else:
        histories=list()
        for k in range(N_part):
          histories.append( {"epochs": [], "train_loss": [], "val_loss": []})
        for idx_lr, lr in enumerate(lrs):
                optims = list()
                for k in range(N_part):
                    #optims.append(torch.optim.SGD(approx_mpc[k].ann.parameters(), lr=lr))
                    optims.append(torch.optim.AdamW(approx_mpc[k].ann.parameters(), lr=lr))
                # optim = torch.optim.Adam(approx_mpc.ann.parameters(),lr=lr)
                # train
                for epoch in range(N_epochs):
                    gradients=list()
                    for k in range(N_part):
                        history=histories[k]
                        # Training
                        backup=copy.deepcopy(approx_mpc[k].ann)
                        train_loss = approx_mpc[k].train_epoch(optims[k], train_loader[k])
                        gradient=approx_mpc[k].compute_grad_update(old_model=backup,new_model=approx_mpc[k].ann,device=device)
                        gradients.append(gradient)
                        approx_mpc[k].ann.load_state_dict(backup.state_dict())
                        history["epochs"].append(epoch)
                        history["train_loss"].append(train_loss)
                        if verbose:
                            print("Epoch: ", history["epochs"][-1])
                            print("Train loss: ", history["train_loss"][-1])
                    aggregated_gradient = [torch.zeros(param.shape).to(device) for param in approx_mpc[0].ann.parameters()]

                    # aggregate and update server model

                    weights = torch.div(torch.tensor(sizes), torch.sum(torch.tensor(sizes)))

                    for gradient, weight in zip(gradients, weights):
                        approx_mpc[0].add_gradient_updates(aggregated_gradient, gradient, weight=weight)
                    for k in range(N_part):
                        approx_mpc[k].add_update_to_model(approx_mpc[k].ann,aggregated_gradient)
                        # Validation
                        if val_loader is not None:
                            val_loss = approx_mpc[k].validation_epoch(val_loader)
                            histories[k]["val_loss"].append(val_loss)
                            if verbose:
                                print("Val loss: ", histories[k]["val_loss"][-1])





    for k in range(N_part):
        fig, ax = plot_history(histories[k])
        figs.append(fig)
        run_hparams = {"n_layers": n_layers, "n_neurons": n_neurons, "n_in": n_in,
                       "n_out": n_out, "N_epochs": N_epochs, "batch_size": batch_size,
                       "lrs": lrs, "overfit_batch": overfit_batch,
                       "train_data_file_name": train_data_file_name,
                       "val_data_file_name": val_data_file_name,
                       "n_train": n_train, "N_steps": N_steps,
                       "optimizer": str(optim_list[k].__class__.__name__),
                       "train_loss": histories[k]["train_loss"][-1],
                       "val_loss": histories[k]["val_loss"][-1]}
        params.append(run_hparams)
    # plot history




# %%
# Save model
print("Saving run...")
# approx_mpc.save_model_settings(file_name="approx_MPC_settings")
# approx_mpc.save_model(file_name="approx_MPC_state_dict")



# check if folder "run_i" exists, count up if it does
for i in range(100):
    run_folder = file_pth.joinpath("approx_mpc_models_fedsgd",f"run_{i}")
    if run_folder.exists():
        continue
    else:
        run_folder.mkdir()
        for k in range(N_part):
            #train_list[k].save_model_settings(folder_path=run_folder,file_name=f"approx_MPC_settings_{k}")
            approx_mpc[k].save_to_state_dict(directory=run_folder.joinpath(f"approx_MPC_state_dict_{k}"))
            figs[k].savefig(run_folder.joinpath(f"history_{k}.png"))
            # save run_hparams as json
            with open(run_folder.joinpath(f"run_hparams_{k}.json"), 'w') as fp:
                json.dump(params[k], fp)
        break



# %%
# Load model
# print("Loading model...")
# approx_mpc_settings_loaded = ApproxMPCSettings.from_json("approx_MPC_settings")
# approx_mpc_loaded = ApproxMPC(approx_mpc_settings_loaded)
# approx_mpc_loaded.load_state_dict(file_name="approx_MPC_state_dict")