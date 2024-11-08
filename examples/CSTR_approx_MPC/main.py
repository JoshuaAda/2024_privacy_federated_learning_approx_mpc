#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import time
import matplotlib
import torch


from do_mpc.approximateMPC.sampling import Sampler
from do_mpc.approximateMPC.approx_MPC import ApproxMPC, Trainer,FeedforwardNN

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

""" User settings: """
show_animation = True
store_results = False
matplotlib.use('TkAgg')

model = template_model()
mpc = template_mpc(model,silence_solver=False)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)
u0=np.array([[0.05],[-8500]])
p0=np.array([[130]])
mpc.x0 = x0
simulator.x0 = x0

mpc.set_initial_guess()
lbx = np.array([[0.1], [0.1], [50], [50]])
ubx = np.array([[2], [2], [140], [140]])
lbu = np.array([[5], [-8500]])
ubu = np.array([[100], [0]])
lbp= np.array([[125]])
ubp= np.array([[135]])
lb=np.concatenate((lbx,lbu,lbp),axis=0)
ub=np.concatenate((ubx,ubu,ubp),axis=0)
n_samples=200



net=FeedforwardNN(n_in=mpc.model.n_x+mpc.model.n_u+model.n_tvp,n_out=mpc.model.n_u,n_neurons=500,n_hidden_layers=2)
approx_mpc = ApproxMPC(net)
sampler=Sampler()
approx_mpc.shift_from_box(lbu.T,ubu.T,lb.T,ub.T)
trainer=Trainer(approx_mpc)
for k in range(6):
    lbp = np.array([[125+2*(k)]])
    ubp = np.array([[125+2*(k)]])
    data_dir = './sampling_'+str(k)
    #sampler.default_sampling(mpc,simulator,estimator,n_samples,lbx,ubx,lbu,ubu,data_dir,parametric=True,lbp=lbp,ubp=ubp)
    #trainer.scale_data(data_dir,n_samples)
#n_opt=7180
n_epochs=1000
#trainer.default_training(data_dir,n_samples,n_epochs)
#approx_mpc.save_to_state_dict('approx_mpc.pth')
approx_mpc.load_from_state_dict('./approx_mpc_models_fedsgd/run_11/approx_MPC_state_dict_0')

# Initialize graphic:
graphics = do_mpc.graphics.Graphics(simulator.data)


fig, ax = plt.subplots(5, sharex=True)
# Configure plot:
graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[3])
graphics.add_line(var_type='_u', var_name='F', axis=ax[4])
ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('T [K]')
ax[2].set_ylabel('$\Delta$ T [K]')
ax[3].set_ylabel('Q [kW]')
ax[4].set_ylabel('Flow [l/h]')
ax[4].set_xlabel('time [h]')
# Update properties for all prediction lines:
for line_i in graphics.pred_lines.full:
    line_i.set_linewidth(1)

label_lines = graphics.result_lines['_x', 'C_a']+graphics.result_lines['_x', 'C_b']
ax[0].legend(label_lines, ['C_a', 'C_b'])
label_lines = graphics.result_lines['_x', 'T_R']+graphics.result_lines['_x', 'T_K']
ax[1].legend(label_lines, ['T_R', 'T_K'])

fig.align_ylabels()
fig.tight_layout()
plt.ion()

timer = Timer()
stage_cost=0
for k in range(100):
    timer.tic()
    #p0 = np.array([[np.random.uniform(125,135)]])
    x = np.concatenate((x0,u0,p0),axis=0).squeeze()
    u0_old=u0
    u0 = approx_mpc.make_step(x,clip_to_bounds=False)
    #u0 = mpc.make_step(x0)
    timer.toc()
    template = simulator.get_p_template()
    #template['m_k'] = np.random.uniform(4,6)
    template['alpha'] = np.random.uniform(0.95,1.05)
    template['beta'] = np.random.uniform(0.9,1.1)
    #template['C_A0'] = np.random.uniform(4.5,5.7)#(4.5 + 5.7) / 2


    def p_fun(t_curr):
        return template


    simulator.set_p_fun(p_fun)

    template_tvp = simulator.get_tvp_template()


    def tvp_fun(t_curr):
        for k in range(21):
            template_tvp['T_in'] = p0
        return template_tvp


    simulator.set_tvp_fun(tvp_fun)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    stage_cost+=(x0[0][0]-0.7)**2+(x0[1][0]-0.6)**2+0.1*(u0[0][0]/100-u0_old[0][0]/100)**2+1e-3*(u0[1][0]/2000-u0_old[1][0]/2000)**2
    if show_animation:
        graphics.plot_results(t_ind=k)
        #graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

timer.info()
timer.hist()
print(stage_cost)
input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'CSTR_robust_MPC')
