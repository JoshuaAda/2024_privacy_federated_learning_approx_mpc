
# %% Imports
import numpy as np
import do_mpc
from pathlib import Path

import pandas as pd
# import time
from timeit import default_timer as timer
import pickle as pkl
# %% Config
#####################################################
class Sampler:
    def __init__(self):
        pass
    def default_sampling(self,mpc,simulator,estimator,n_samples,lbx,ubx,lbu,ubu,data_dir='./sampling',parametric=False,lbp=None,ubp=None):
        #lbx,ubx,lbu,ubu=self.boxes_from_mpc(mpc)
        self.simulator=simulator
        self.estimator=estimator
        self.approx_mpc_sampling_plan_box(n_samples,lbx,ubx,lbu,ubu,parametric,lbp,ubp,data_dir)
        self.approx_mpc_closed_loop_sampling(mpc,n_samples,data_dir=data_dir,parametric=parametric)

    def boxes_from_mpc(self,mpc):
        pass

    def approx_mpc_sampling_plan_box(self,n_samples,lbx,ubx,lbu,ubu,parametric=False,lbp=None,ubp=None,data_dir='./sampling',overwrite=True):
        # Samples
        data_dir=Path(data_dir)
        sampling_plan_name = 'sampling_plan' + '_n' + str(n_samples)
        #overwrite = True
        id_precision = np.ceil(np.log10(n_samples)).astype(int)


        #####################################################

        # %% Functions

        def gen_x0():
            x0 = np.random.uniform(lbx, ubx)
            return x0

        def gen_u_prev():
            u_prev = np.random.uniform(lbu, ubu)
            return u_prev
        if parametric:
            def gen_p():
                p = np.random.uniform(lbp, ubp)
                return p

        # %%
        # Sampling Plan

        assert n_samples <= 10 ** (id_precision + 1), "Not enough ID-digits to save samples"
        # Initialize sampling planner
        sp = do_mpc.sampling.SamplingPlanner()
        sp.set_param(overwrite=overwrite)
        sp.set_param(id_precision=id_precision)
        sp.data_dir = data_dir.__str__()+"/"

        # Set sampling vars
        sp.set_sampling_var('x0', gen_x0)
        sp.set_sampling_var('u_prev', gen_u_prev)
        if parametric:
            sp.set_sampling_var('p', gen_p)
        # Generate sampling plan
        plan = sp.gen_sampling_plan(n_samples=n_samples)

        # Export
        sp.export(sampling_plan_name)


    def approx_mpc_sampling_plan_func(self,n_samples,gen_x0,gen_u_prev,data_dir='./sampling',overwrite=True):
        # Samples
        data_dir=Path(data_dir)
        sampling_plan_name = 'sampling_plan' + '_n' + str(n_samples)
        #overwrite = True
        id_precision = np.ceil(np.log10(n_samples)).astype(int)


        #####################################################

        # %% Functions



        # %%
        # Sampling Plan

        assert n_samples <= 10 ** (id_precision + 1), "Not enough ID-digits to save samples"
        # Initialize sampling planner
        sp = do_mpc.sampling.SamplingPlanner()
        sp.set_param(overwrite=overwrite)
        sp.set_param(id_precision=id_precision)
        sp.data_dir = data_dir.__str__()+"/"

        # Set sampling vars
        sp.set_sampling_var('x0', gen_x0)
        sp.set_sampling_var('u_prev', gen_u_prev)

        # Generate sampling plan
        plan = sp.gen_sampling_plan(n_samples=n_samples)

        # Export
        sp.export(sampling_plan_name)

    # import pickle as pkl
    # with open('./sampling_test/test_sampling_plan.pkl','rb') as f:
    #     plan = pkl.load(f)


    # Control Problem
    #from nl_double_int_nmpc.template_model import template_model
    #from nl_double_int_nmpc.template_mpc import template_mpc
    #from nl_double_int_nmpc.template_simulator import template_simulator

    #from nlp_handler import NLPHandler





    #####################################################
    # %% MPC
    def approx_mpc_closed_loop_sampling(self,mpc,n_samples,data_dir='./sampling',overwrite_sampler=True,parametric=False):
        # %% Config
        #####################################################
        suffix='_n'+str(n_samples)
        sampling_plan_name = 'sampling_plan'
        sample_name = 'sample'
        data_dir=Path(data_dir)
        #return_full_mpc_data = True

        ## How are samples named? (DEFAULT)
        #sample_name = 'sample'
        #suffix = '_n4000'
        sampling_plan_name = sampling_plan_name + suffix  # 'sampling_plan'+suffix

        #overwrite_sampler = False
        samples_dir = data_dir.joinpath('samples' + suffix)
        #samples_dir = data_dir+'samples' + suffix

        # Data
        #test_run = False
        # filter_success_runs = False
        data_file_name = 'data'

        # Assertion for scaling
        #for val in [mpc._x_scaling.cat, mpc._p_scaling.cat, mpc._u_scaling.cat]:
        #    assert (np.array(val)==1).all(), "you have to consider scaling: change opt_x_num to consider scaled values"

        # %% NLP Handler
        # setup NLP Handler
        #nlp_handler = NLPHandler(mpc)

        # %% Functions
        if parametric:
            # Sampling functions
            def run_mpc_closed_loop(x0, u_prev,p):
                mpc.reset_history()
                mpc.x0 = x0
                mpc.u0 = u_prev
                p_total=np.repeat(p,40)
                u_prev_total=np.zeros((40,2))
                mpc.set_initial_guess()
                template = mpc.get_tvp_template()
                def tvp_fun(t_curr):
                    for k in range(mpc.settings.n_horizon + 1):
                        template['_tvp', k, 'T_in'] = p
                    return template

                mpc.set_tvp_fun(tvp_fun)
                start = timer()
                mpc.reset_history()
                self.simulator.reset_history()
                self.estimator.reset_history()

                # set initial values and guess

                mpc.x0 = x0
                self.simulator.x0 = x0
                self.estimator.x0 = x0

                mpc.set_initial_guess()
                u_prev_curr=u_prev
                # run the closed loop for 150 steps
                for k in range(40):
                    u_prev_total[k]=u_prev_curr.reshape((2,))
                    u0 = mpc.make_step(x0)
                    u_prev_curr=u0
                    if mpc.solver_stats["success"] ==False:
                        break
                    y_next = self.simulator.make_step(u0)
                    x0 = self.estimator.make_step(y_next)

                # we return the complete data structure that we have obtained during the closed-loop run


                end = timer()

                stats = {}
                stats["t_make_step"] = end - start
                stats["success"] = mpc.solver_stats["success"]
                stats["iter_count"] = mpc.solver_stats["iter_count"]

                if "t_wall_total" in mpc.solver_stats:
                    stats["t_wall_total"] = mpc.solver_stats["t_wall_total"]
                else:
                    stats["t_wall_total"] = np.nan

                    # if return_full:
                    #    ### get solution
                    #    nlp_sol, p_num = nlp_handler.get_mpc_sol(mpc)
                    #    z_num = nlp_handler.extract_numeric_primal_dual_sol(nlp_sol)
                    ### reduced solution
                    # z_num, p_num = nlp_handler.get_reduced_primal_dual_sol(nlp_sol,p_num)
                    #    return u0, stats, np.array(z_num), np.array(p_num), mpc.data
                    # else:
                return self.simulator.data, stats, u_prev_total, p_total
            def sample_function(x0, u_prev,p):
                return run_mpc_closed_loop(x0, u_prev,p)
        else:
            # Sampling functions
            def run_mpc_closed_loop(x0, u_prev):
                mpc.reset_history()
                mpc.x0 = x0
                mpc.u0 = u_prev
                mpc.set_initial_guess()



                start = timer()
                mpc.reset_history()
                self.simulator.reset_history()
                self.estimator.reset_history()

                # set initial values and guess

                mpc.x0 = x0
                self.simulator.x0 = x0
                self.estimator.x0 = x0

                mpc.set_initial_guess()

                # run the closed loop for 150 steps
                for k in range(2):
                    u0 = mpc.make_step(x0)
                    y_next = self.simulator.make_step(u0)
                    x0 = self.estimator.make_step(y_next)

                # we return the complete data structure that we have obtained during the closed-loop run

                end = timer()

                stats = {}
                stats["t_make_step"] = end - start
                stats["success"] = mpc.solver_stats["success"]
                stats["iter_count"] = mpc.solver_stats["iter_count"]

                if "t_wall_total" in mpc.solver_stats:
                    stats["t_wall_total"] = mpc.solver_stats["t_wall_total"]
                else:
                    stats["t_wall_total"] = np.nan

                    # if return_full:
                    #    ### get solution
                    #    nlp_sol, p_num = nlp_handler.get_mpc_sol(mpc)
                    #    z_num = nlp_handler.extract_numeric_primal_dual_sol(nlp_sol)
                    ### reduced solution
                    # z_num, p_num = nlp_handler.get_reduced_primal_dual_sol(nlp_sol,p_num)
                    #    return u0, stats, np.array(z_num), np.array(p_num), mpc.data
                    # else:
                return self.simulator.data, stats
            # Sampling function
            def sample_function(x0, u_prev):
                return run_mpc_closed_loop(x0, u_prev)

        # %% Sampling Plan
        # Import sampling plan
        # with open(data_dir+sampling_plan_name+'.pkl','rb') as f:
        with open(data_dir.joinpath(sampling_plan_name+'.pkl'),'rb') as f:
            plan = pkl.load(f)

        # %% Sampler
        sampler = do_mpc.sampling.Sampler(plan)
        sampler.data_dir = str(samples_dir)+'/'
        sampler.set_param(overwrite=overwrite_sampler)
        sampler.set_param(sample_name=sample_name)

        sampler.set_sample_function(sample_function)

        # %% Main - Sample Data
        #if test_run:
        #    sampler.sample_idx(0)
        #else:
        sampler.sample_data()

        # %% Data Handling
        dh = do_mpc.sampling.DataHandler(plan)

        dh.data_dir = str(samples_dir)+'/'
        dh.set_param(sample_name = sample_name)
        dh.set_post_processing('u0', lambda x: x[0]['_u'])
        dh.set_post_processing('x0', lambda x: x[0]['_x'])
        dh.set_post_processing('u_prev', lambda x: x[2])
        dh.set_post_processing('p', lambda x: x[3])
        dh.set_post_processing('status', lambda x: x[1]["success"])
        dh.set_post_processing('t_make_step', lambda x: x[1]["t_make_step"])
        dh.set_post_processing('t_wall', lambda x: x[1]["t_wall_total"])
        dh.set_post_processing('iter_count', lambda x: x[1]["iter_count"])
        df = pd.DataFrame(dh[:])
        n_data = df.shape[0]
        df.to_pickle(str(data_dir) + '/' + data_file_name + '_n{}'.format(n_data) + '_all' + '.pkl')
        # %% Save
        # Filter opt and Save
        df = pd.DataFrame(dh.filter(output_filter=lambda status: status == True))
        n_data_opt = df.shape[0]
        df.to_pickle(str(data_dir) + '/' + data_file_name + '_n{}'.format(n_data) + '_opt' + '.pkl')
    def approx_mpc_open_loop_sampling(self,mpc,n_samples,data_dir='./sampling',overwrite_sampler=True,parametric=False):
        # %% Config
        #####################################################
        suffix='_n'+str(n_samples)
        sampling_plan_name = 'sampling_plan'
        sample_name = 'sample'
        data_dir=Path(data_dir)
        #return_full_mpc_data = True

        ## How are samples named? (DEFAULT)
        #sample_name = 'sample'
        #suffix = '_n4000'
        sampling_plan_name = sampling_plan_name + suffix  # 'sampling_plan'+suffix

        #overwrite_sampler = False
        samples_dir = data_dir.joinpath('samples' + suffix)
        #samples_dir = data_dir+'samples' + suffix

        # Data
        #test_run = False
        # filter_success_runs = False
        data_file_name = 'data'

        # Assertion for scaling
        #for val in [mpc._x_scaling.cat, mpc._p_scaling.cat, mpc._u_scaling.cat]:
        #    assert (np.array(val)==1).all(), "you have to consider scaling: change opt_x_num to consider scaled values"

        # %% NLP Handler
        # setup NLP Handler
        #nlp_handler = NLPHandler(mpc)

        # %% Functions
        if parametric:
            # Sampling functions
            def run_mpc_one_step(x0, u_prev,p):
                mpc.reset_history()
                mpc.x0 = x0
                mpc.u0 = u_prev
                mpc.set_initial_guess()
                template = mpc.get_tvp_template()
                def tvp_fun(t_curr):
                    for k in range(mpc.settings.n_horizon + 1):
                        template['_tvp', k, 'T_in'] = p
                    return template

                mpc.set_tvp_fun(tvp_fun)
                start = timer()
                u0 = mpc.make_step(x0)
                end = timer()

                stats = {}
                stats["t_make_step"] = end - start
                stats["success"] = mpc.solver_stats["success"]
                stats["iter_count"] = mpc.solver_stats["iter_count"]

                if "t_wall_total" in mpc.solver_stats:
                    stats["t_wall_total"] = mpc.solver_stats["t_wall_total"]
                else:
                    stats["t_wall_total"] = np.nan

                    # if return_full:
                    #    ### get solution
                    #    nlp_sol, p_num = nlp_handler.get_mpc_sol(mpc)
                    #    z_num = nlp_handler.extract_numeric_primal_dual_sol(nlp_sol)
                    ### reduced solution
                    # z_num, p_num = nlp_handler.get_reduced_primal_dual_sol(nlp_sol,p_num)
                    #    return u0, stats, np.array(z_num), np.array(p_num), mpc.data
                    # else:
                return u0, stats
            def sample_function(x0, u_prev,p):
                return run_mpc_one_step(x0, u_prev,p)
        else:
            # Sampling functions
            def run_mpc_one_step(x0, u_prev):
                mpc.reset_history()
                mpc.x0 = x0
                mpc.u0 = u_prev
                mpc.set_initial_guess()

                start = timer()
                u0 = mpc.make_step(x0)
                end = timer()

                stats = {}
                stats["t_make_step"] = end-start
                stats["success"] = mpc.solver_stats["success"]
                stats["iter_count"] = mpc.solver_stats["iter_count"]

                if "t_wall_total" in mpc.solver_stats:
                    stats["t_wall_total"] = mpc.solver_stats["t_wall_total"]
                else:
                    stats["t_wall_total"] = np.nan

                #if return_full:
                #    ### get solution
                #    nlp_sol, p_num = nlp_handler.get_mpc_sol(mpc)
                #    z_num = nlp_handler.extract_numeric_primal_dual_sol(nlp_sol)
                    ### reduced solution
                    # z_num, p_num = nlp_handler.get_reduced_primal_dual_sol(nlp_sol,p_num)
                #    return u0, stats, np.array(z_num), np.array(p_num), mpc.data
                #else:
                    return u0, stats

            # Sampling function
            def sample_function(x0, u_prev):
                return run_mpc_one_step(x0, u_prev)

        # %% Sampling Plan
        # Import sampling plan
        # with open(data_dir+sampling_plan_name+'.pkl','rb') as f:
        with open(data_dir.joinpath(sampling_plan_name+'.pkl'),'rb') as f:
            plan = pkl.load(f)

        # %% Sampler
        sampler = do_mpc.sampling.Sampler(plan)
        sampler.data_dir = str(samples_dir)+'/'
        sampler.set_param(overwrite=overwrite_sampler)
        sampler.set_param(sample_name=sample_name)

        sampler.set_sample_function(sample_function)

        # %% Main - Sample Data
        #if test_run:
        #    sampler.sample_idx(0)
        #else:
        sampler.sample_data()

        # %% Data Handling
        dh = do_mpc.sampling.DataHandler(plan)

        dh.data_dir = str(samples_dir)+'/'
        dh.set_param(sample_name = sample_name)
        dh.set_post_processing('u0', lambda x: x[0])
        dh.set_post_processing('status', lambda x: x[1]["success"])
        dh.set_post_processing('t_make_step', lambda x: x[1]["t_make_step"])
        dh.set_post_processing('t_wall', lambda x: x[1]["t_wall_total"])
        dh.set_post_processing('iter_count', lambda x: x[1]["iter_count"])
        #if return_full_mpc_data == True:
        #    dh.set_post_processing('z_num', lambda x: x[2])
        #    dh.set_post_processing('p_num', lambda x: x[3])
        #    dh.set_post_processing('mpc_data', lambda x: x[4])

        # if filter_success_runs:
        #     df = pd.DataFrame(dh.filter(output_filter = lambda status: status==True))
        # else:
        #     df = pd.DataFrame(dh[:])

        # n_data = df.shape[0]

        # # %% Save
        # if filter_success_runs:
        #     df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_opt' + '.pkl')
        # else:
        #     df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_all' + '.pkl')

        df = pd.DataFrame(dh[:])
        n_data = df.shape[0]
        df.to_pickle(str(data_dir) + '/' + data_file_name + '_n{}'.format(n_data) + '_all' + '.pkl')
        # %% Save
        # Filter opt and Save
        df = pd.DataFrame(dh.filter(output_filter = lambda status: status==True))
        n_data_opt = df.shape[0]
        df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_opt' + '.pkl')

        # Save all

