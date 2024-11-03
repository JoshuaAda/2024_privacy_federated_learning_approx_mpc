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
import casadi as cas
import pdb
import sys
import time
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

from template_model import template_model
from template_lqr import template_lqr
from template_simulator import template_simulator

""" User settings: """
store_results = True

"""
Get configured do-mpc modules:
"""
model = template_model()
lqr = template_lqr(model)
simulator = template_simulator(model)

"""
Set initial state
"""
x0 = np.array([[2],[1],[3],[1]])
simulator.x0 = x0

"""
Run MPC main loop:
"""

for k in range(50):
    u0 = lqr.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = y_next
    
"""
Setup graphic:
"""
fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data, figsize=(16,9))
graphics.plot_results()
graphics.reset_axes()
plt.show()

# Store results:
if store_results:
    do_mpc.data.save_results([simulator], 'results_oscillatingMasses_LQR')