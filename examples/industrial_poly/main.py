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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import time


from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from do_mpc.approximateMPC.sampling import Sampler
from do_mpc.approximateMPC.approx_MPC import ApproxMPC, Trainer,FeedforwardNN


""" User settings: """
show_animation = True
store_results = False

"""
Get configured do-mpc modules:
"""

model = template_model()
mpc = template_mpc(model,silence_solver=True)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of the controller and simulator:
delH_R_real = 950.0
c_pR = 5.0

# x0 is a property of the simulator - we obtain it and set values.
x0 = simulator.x0

x0['m_W'] = 10000.0
x0['m_A'] = 853.0
x0['m_P'] = 26.5

x0['T_R'] = 92.0 + 273.15
x0['T_S'] = 90.0 + 273.15
x0['Tout_M'] = 90.0 + 273.15
x0['T_EK'] = 35.0 + 273.15
x0['Tout_AWT'] = 35.0 + 273.15
x0['accum_monom'] = 300.0

x0['T_adiab'] = x0['m_A']*delH_R_real/((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']

# Finally, the controller gets the same initial state.
mpc.x0 = x0
lbx = np.array([[0], [0], [26], [361.15],[298],[298],[288],[288],[270],[0],[0]])
ubx = np.array([[1000], [1000],[1000], [365.15], [400], [400], [400], [400], [30000], [382.15], [1000]])
lbp= np.array([[1]])
lbu = np.array([[0], [270],[270]])
ubu = np.array([[30000], [400],[300]])
ubp= np.array([[100]])
lb=np.concatenate((lbx,lbu,lbp),axis=0)
ub=np.concatenate((ubx,ubu,ubp),axis=0)

n_samples=1000

data_dir = './sampling'

net=FeedforwardNN(n_in=mpc.model.n_x+mpc.model.n_u+mpc.model.n_tvp,n_out=mpc.model.n_u)
approx_mpc = ApproxMPC(net)
sampler=Sampler()
approx_mpc.shift_from_box(lbu.T,ubu.T,lb.T,ub.T)
trainer=Trainer(approx_mpc)
sampler.default_sampling(mpc,n_samples,lbx,ubx,lbu,ubu,parametric=True,lbp=lbp,ubp=ubp)
#n_opt=7180
n_epochs=2000
trainer.default_training(data_dir,n_samples,n_epochs,)
approx_mpc.save_to_state_dict('approx_mpc.pth')
approx_mpc.load_from_state_dict('approx_mpc.pth')

# Which is used to set the initial guess:
mpc.set_initial_guess()

# Initialize graphic:
graphics = do_mpc.graphics.Graphics(mpc.data)


fig, ax = plt.subplots(7, sharex=True, figsize=(16,9))
plt.ion()
# Configure plot:
graphics.add_line(var_type='_x', var_name='T_R', axis=ax[0])
graphics.add_line(var_type='_x', var_name='accum_monom', axis=ax[1])
graphics.add_line(var_type='_u', var_name='m_dot_f', axis=ax[2])
graphics.add_line(var_type='_u', var_name='T_in_M', axis=ax[3])
graphics.add_line(var_type='_u', var_name='T_in_EK', axis=ax[4])
graphics.add_line(var_type='_x', var_name='Q', axis=ax[5])
graphics.add_line(var_type='_x', var_name='m_P', axis=ax[6])

ax[0].set_ylabel('T_R [K]')
ax[1].set_ylabel('acc. monom')
ax[2].set_ylabel('m_dot_f')
ax[3].set_ylabel('T_in_M [K]')
ax[4].set_ylabel('T_in_EK [K]')
ax[5].set_ylabel('Q')
ax[4].set_xlabel('time')

fig.align_ylabels()
plt.ion()

for k in range(100):

    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'industrial_poly')
