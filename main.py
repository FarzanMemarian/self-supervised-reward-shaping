import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SumxmaryWriter


from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail, gail_lipschitz, ppo_lipschitz
import a2c_ppo_acktr.arguments as arguments #import get_args, get_init, get_init_Lipschitz
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from a2c_ppo_acktr.envs import VecPyTorch
from baselines.common.atari_wrappers import FrameStack, ClipRewardEnv, WarpFrame

from procgen import ProcgenEnv
import argparse

import sys
sys.path.append(os.getcwd())
import pickle
from a2c_ppo_acktr.utils import init
from utils import myutils
import mujoco_py

from os import listdir
from os.path import isfile, join
import math
import psutil
import tracemalloc
import linecache
# from memory_profiler import profile
# breakpoint()


# ------------ REWARD NETWORKS ------------
class net_MLP(nn.Module):

    def __init__(self, 
                input_size,
                rew_sign,
                rew_mag,
                FC1_dim = 256,
                FC2_dim = 256,
                FC3_dim = 256,  
                out_dim=1):
        super().__init__()
        # an affine operation: y = Wx + b
        
        self.n_dim = input_size
        self.fc1 = nn.Linear(self.n_dim, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, FC3_dim)
        self.fc4 = nn.Linear(FC3_dim, out_dim)

        self.rew_sign = rew_sign
        self.rew_mag = rew_mag

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.rew_sign == "free":
            x = self.fc4(x)
        elif self.rew_sign == "neg":
            x = - F.relu(self.fc4(x))
        elif self.rew_sign == "pos":
            x =  F.relu(self.fc4(x))
        elif self.rew_sign == "pos_sigmoid":
            x =  self.rew_mag*torch.sigmoid(self.fc4(x))
        elif self.rew_sign == "neg_sigmoid":
            x =  - self.rew_mag*torch.sigmoid(self.fc4(x))
        elif self.rew_sign == "tanh":
            x =  self.rew_mag*torch.tanh(self.fc4(x))
        return x

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 


class net_CNN(nn.Module):
    def __init__(self, 
            observation_space_shape, 
            rew_sign, 
            rew_mag,
            final_conv_channels=10):
        super().__init__()
        depth = observation_space_shape[0]
        n_dim = observation_space_shape[1]
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.rew_sign = rew_sign
        self.rew_mag = rew_mag

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        conv1_output_width = conv2d_size_out(n_dim, 8,4)
        conv2_output_width = conv2d_size_out(conv1_output_width, 4,2)
        conv3_output_width = conv2d_size_out(conv2_output_width, 3,1)
        conv4_output_width = conv2d_size_out(conv3_output_width, 7,1)
        FC_input_size = conv4_output_width * conv4_output_width * final_conv_channels
        
        self.conv1 = nn.Conv2d(depth, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv4 = nn.Conv2d(64, final_conv_channels, 7, stride=1)
        self.FC =    nn.Linear(FC_input_size, 1)
        # self.main = nn.Sequential(
        #             nn.Conv2d(n_dim, 32, 8, stride=4), nn.ReLU(),
        #             nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        #             nn.Conv2d(64, 66, 3, stride=1), nn.ReLU(), 
        #             nn.Conv2d(64, final_conv_channels, 7, stride=1), nn.ReLU(), 
        #             Flatten(),
        #             nn.Linear(FC_input_size, 1)
        #             )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) # flatten
        if self.rew_sign == "free":
            x = self.FC(x)
        elif self.rew_sign == "neg":
            x = - F.relu(self.FC(x))
        elif self.rew_sign == "pos":
            x =  F.relu(self.FC(x))
        elif self.rew_sign == "pos_sigmoid":
            x =  self.rew_mag*torch.sigmoid(self.FC(x))
        elif self.rew_sign == "neg_sigmoid":
            x =  - self.rew_mag*torch.sigmoid(self.FC(x))
        elif self.rew_sign == "tanh":
            x =  self.rew_mag*torch.tanh(self.FC(x))
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features    


# Auxiliary functions

class reward_cl():

    def __init__(self, device, observation_space_shape, lr, rew_sign, rew_mag, rew_kwargs):

        # new networks
        self.device = device

        if len(observation_space_shape) == 1:
            # num_stacked_obs = 4
            self.reward_net = net_MLP(observation_space_shape[0], rew_sign, rew_mag, FC1_dim=rew_kwargs['FC_dim'], FC2_dim=rew_kwargs['FC_dim'], out_dim=1).to(self.device)
        elif len(observation_space_shape) == 3:
            self.reward_net = net_CNN(observation_space_shape, rew_sign, rew_mag).to(self.device)
        
        # NOTE: by commenting the following lines, we rely on Pytorch's initialization. 
        # Pytorch uses Kaiming Initialization which is good for linear layers with ReLu activations
        # self.init_weights_var = 0.05
        # self._init_weights(self.reward_net) 

        self.lr = lr
        # create the optimizer
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.001, amsgrad=False)

        # theta based variables
        self.reward_input_batch = None
        self.theta_size = self._get_size_theta()
        # self.grad_R_theta = np.zeros((self.MDP.n_dim, self.theta_size))

    def _get_size_theta(self):
        size = 0
        for f in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            dims = f.size()
            layer_size = 1
            for dim in dims:
                layer_size *= dim
            size += layer_size
        return size
 
    def _init_weights(self, reward_net):
        with torch.no_grad():
            for layer_w in reward_net.parameters():
                torch.nn.init.normal_(layer_w, mean=0.0, std=self.init_weights_var)
                # torch.nn.init.xavier_normal_(layer_w, gain=1.0)

    def reward_net_input_method(self,obs):
        return obs

    def reward_net_input_batch_traj_method(self, traj):
        reward_input_batch = torch.cat([torch.unsqueeze(trans, dim=0) for trans in traj], dim=0)
        # reward_input_batch.requires_grad_(True)
        return reward_input_batch

    def reward_net_input_batch_traj_method_stacked(self, traj):
        stacked_obs_list = []
        for idx in range(len(traj)):
            if idx == 0:
                stacked_obs = torch.cat([traj[0][0], traj[0][0], traj[0][0], traj[0][0]], dim=1) 
            elif idx == 1:
                stacked_obs = torch.cat([traj[1][0], traj[0][0], traj[0][0], traj[0][0]], dim=1) 
            elif idx == 2:
                stacked_obs = torch.cat([traj[2][0], traj[1][0], traj[0][0], traj[0][0]], dim=1) 
            else:
                stacked_obs = torch.cat([traj[idx][0], traj[idx-1][0], traj[idx-2][0], traj[idx-3][0]], dim=1)
            stacked_obs_list.append(stacked_obs)
        return torch.cat(stacked_obs_list, dim=0)

    def _get_flat_grad(self):
        # this part is to get the thetas to be used for l2 regularization
        # grads_flat = torch.zeros(self.theta_size)
        grads_flat_list = []
        start_pos = 0
        for idx, weights in enumerate(self.reward_net.parameters()):
            # SHOULD BE DOUBLE CHECKED 1234567891011
            num_flat_features = self._num_flat_features(weights)

            try:
                grads = copy.deepcopy(weights.grad.view(-1, num_flat_features))      
            except Exception as e:
                print("No gradient error")

            # grads_flat[start_pos:start_pos+num_flat_features] = grads[:]
            # start_pos += num_flat_features
            grads_flat_list.append(grads)
        grads_flat = torch.unsqueeze(torch.cat(grads_flat_list, dim=1), dim=0)
        return grads_flat

    def get_flat_weights(self):
        # this part is to get the thetas to be used for l2 regularization
        weights_flat = torch.zeros(self.theta_size)
        start_pos = 0
        for weights in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            num_flat_features = self._num_flat_features(weights)
            weights = copy.deepcopy(weights.view(-1, num_flat_features).detach())
            weights_flat[start_pos:start_pos+num_flat_features] = weights[:]
            start_pos += num_flat_features
        return weights_flat

    def _num_flat_features(self, x):
        size = x.size()  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

    def _get_size_theta(self):
        size = 0
        for f in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            dims = f.size()
            layer_size = 1
            for dim in dims:
                layer_size *= dim
            size += layer_size
        return size


class train_sparse_rank():

    def __init__(self, kwargs, myargs, init_params):
        if not myargs.seed == -12345: 
            # seed is provided as command line argument and nothing needs to be done
            pass
        else:        
            if os.getenv('SEED'):
                myargs.seed = int(os.getenv('SEED'))
            else:
                raise ValueError('SEED not provided as command line argument or as an enviornment variable')

        if myargs.save_name:
            add_to_save_name = f"-{myargs.save_name}"
        else:
            add_to_save_name = ""

        if myargs.shaping:
            shaping_str = "-shaping"
        else:
            shaping_str = ""

        if myargs.sparse_cntr:
            sparse_cntr_str = f"-sparseCntr"
        else:
            sparse_cntr_str = ""

        if myargs.sparseness:
            sparseness_str = f"-sp{myargs.sparseness}"
        else:
            sparseness_str = ""


        myargs.save_name = f"RLfRD-{myargs.env_name}{shaping_str}-{myargs.sparse_rew_type}-{myargs.rew_sign}{sparseness_str}{sparse_cntr_str}" + add_to_save_name + f"-s{myargs.seed}"

        self.kwargs = kwargs
        self.myargs = myargs
        self.init_params = init_params

        self.device = myutils.assign_gpu_device(self.myargs)


        self.log_dir = myargs.log_dir + "/" + myargs.save_name 
        eval_log_dir = self.log_dir + "_eval"

        self.log_file_name = myargs.env_name
        # utils.cleanup_log_dir(log_dir)
        # utils.cleanup_log_dir(eval_log_dir)


        if not myargs.continue_:
            utils.create_dir(self.log_dir)
        # utils.create_dir(eval_log_dir)

        # self.save_path_trained_models is for storing the trained model
        self.save_path_trained_models = os.path.join(myargs.save_dir, myargs.algo, myargs.save_name)

        if self.myargs.train_from_given_ranked_demos:
            self.save_path_new_trajs = self.myargs.ranked_demos_address + "/train"
            self.save_path_new_trajs_val = self.myargs.ranked_demos_address + "/val"

        else:
            self.save_path_new_trajs = os.path.join(myargs.save_dir, myargs.algo, myargs.save_name, "new_trajs")

        if not myargs.continue_:
            utils.create_dir(self.save_path_trained_models)
            if not self.myargs.train_from_given_ranked_demos:
                utils.create_dir(self.save_path_new_trajs)

        # # Create forlder for tensorboard
        # self.writer = SummaryWriter(f'runs/visualization')

        torch.manual_seed(myargs.seed)
        torch.cuda.manual_seed_all(myargs.seed)
        np.random.seed(myargs.seed)

        if myargs.cuda and torch.cuda.is_available() and myargs.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)

        self.envs = make_vec_envs(myargs.env_name, myargs.seed, myargs.num_processes,
            myargs.gamma, self.log_dir, self.device, allow_early_resets=True, num_frame_stack=2, **kwargs)
        # envs = ProcgenEnv(num_envs=myargs.env_name, env_name="heistpp", **kwargs)
        # envs = gym.make(myargs.env_name, **kwargs)

        self.is_atari = myargs.is_atari

        if myargs.env_name in ["MountainCar-v0", "Reacher-v2", "Acrobot-v1", "Thrower-v2"]:
            hidden_size_policy = 10
        else:
            hidden_size_policy = 64


        if self.myargs.continue_:
            # Load the pretrained policy
            print(f'Loading policy for continuing run {self.myargs.save_name} .....')
            model_save_address_policy = os.path.join(self.save_path_trained_models, self.myargs.save_name + f"_policy.pt")
            self.actor_critic, ob_rms = torch.load(model_save_address_policy, map_location=self.device)

        else:
            self.actor_critic = Policy(
                self.envs.observation_space.shape,
                self.envs.action_space,
                self.device,
                base_kwargs={'recurrent': myargs.recurrent_policy, 'hidden_size': hidden_size_policy})


        self.actor_critic.to(self.device)

        if myargs.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                alpha=myargs.alpha,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'ppo':
            self.agent = algo.PPO(
                self.actor_critic,
                myargs.clip_param,
                myargs.ppo_epoch,
                myargs.num_mini_batch,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, myargs.value_loss_coef, myargs.entropy_coef, acktr=True)



        # Initialize the reward function
        # self.reward_obj = reward_cl(myargs.num_processes, self.device, self.envs.observation_space.shape, myargs.rew_lr, myargs.rew_sign, myargs.rew_mag)
        self.num_rew_nets = myargs.num_rew_nets
        if self.myargs.env_name in ["MountainCar-v0", "Reacher-v2", "Acrobot-v1", "Thrower-v2"]:
            FC_dim_rew = 10
        else:
            FC_dim_rew = 60

        self.reward_objs = [reward_cl(self.device, self.envs.observation_space.shape, myargs.rew_lr, myargs.rew_sign, myargs.rew_mag, rew_kwargs={'FC_dim':FC_dim_rew}) for i in range(self.num_rew_nets)]

        if self.myargs.continue_:

            # Load the pretrained reward function
            print(f'Loading the reward function for continuing run  {self.myargs.save_name} .....')
            for reward_idx, reward_obj in enumerate(self.reward_objs):   
                model_save_address = os.path.join(self.save_path_trained_models, self.myargs.save_name + f"_reward_{reward_idx}.pt")
                checkpoint = torch.load(model_save_address, map_location=self.device)
                self.reward_objs[reward_idx].reward_net.load_state_dict(checkpoint['model_state_dict'])
                self.reward_objs[reward_idx].optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.reward_objs[reward_idx].reward_net.train()



        self.rollouts = RolloutStorage(myargs.num_steps, myargs.num_processes,
                              self.envs.observation_space.shape, self.envs.action_space,
                              self.actor_critic.recurrent_hidden_state_size)

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        with torch.no_grad():
            rew_nets_step_list = [reward_obj.reward_net(obs) for reward_obj in self.reward_objs]
        self.old_rews_nets_step = torch.mean(torch.cat(rew_nets_step_list, dim=1), dim=1)

        if not myargs.continue_:
            with open(self.log_dir + f"/policy_stats.txt", "w") as file: 
                file.write("overal_tr_iter_idx, updates , num timesteps , FPS, number of Last training episodes, dist_entropy, value_loss, action_loss, mean reward, median reward, min reward, max reward \n")

            with open(self.log_dir + f"/rew_weights_stats.txt", "w") as file: 
                file.write("reward_mean, reward_std \n")

            with open(self.log_dir + f"/rew_losses.txt", "w") as file: 
                file.write("g_value \n")

            with open(self.log_dir + f"/rew_losses_val.txt", "w") as file: 
                file.write("g_value \n")

            with open(self.log_dir + "/buffer_stats.txt", "w") as file: 
                file.write("mean, range, std, mean_new, range_new, std_new \n")

        # if not myargs.run_type == "pretrain_only":
            # self.new_trajs_list = deque(maxlen= self.init_params['size_of_new_trajs_list']) # deque for storing new trajs produced by the policy
            # self.new_trajs_returns_list = deque(maxlen= self.init_params['size_of_new_trajs_list'])

        self.new_trajs_list = []
        self.new_trajs_list_val = []


        if self.myargs.train_from_given_ranked_demos:
            self.sample_trajs_from_memory = True

            if "devv" in self.myargs.save_name:
                max_train_load = 50
            else:
                max_train_load = -1

            # *************************************************
            # Load new_trajs_returns_list
            with open(f"{self.save_path_new_trajs}/trajs_returns_all.pkl", "rb") as file:
                new_trajs_returns_list_temp = pickle.load(file)
            self.new_trajs_returns_list = new_trajs_returns_list_temp[:max_train_load]

            print("loading training ranked trajs ....")
            start_loading_trajs = time.time()

            for idx in range(len(self.new_trajs_returns_list[0:max_train_load])):
                # print(f"loading traj {idx}th out of {len(self.new_trajs_returns_list)} trajs ....")
                traj = torch.load(self.save_path_new_trajs+f"/traj_{idx}.pt", map_location=self.device)
                self.new_trajs_list.append(traj)
            print(f"Total time loading training ranked trajs: {time.time()-start_loading_trajs}")


            # *************************************************
            # Load new_trajs_returns_list_val
            with open(f"{self.save_path_new_trajs_val}/trajs_returns_all.pkl", "rb") as file:
                new_trajs_returns_list_val_temp = pickle.load(file)
            self.new_trajs_returns_list_val = new_trajs_returns_list_val_temp[:max_train_load]

            print("loading val ranked trajs ....")
            start_loading_trajs = time.time()
            for idx in range(len(self.new_trajs_returns_list_val[0:max_train_load])):
                # print(f"loading traj {idx}th out of {len(self.new_trajs_returns_list)} trajs ....")
                traj = torch.load(self.save_path_new_trajs_val+f"/traj_{idx}.pt", map_location=self.device)
                self.new_trajs_list_val.append(traj)
            print(f"Total time loading val ranked trajs: {time.time()-start_loading_trajs}")


        else:
            self.new_trajs_returns_list = []
            self.sample_trajs_from_memory = False


        self.new_trajs_last_idx = 0
        self.size_of_new_trajs_list = init_params['size_of_new_trajs_list']


        if self.myargs.skip_rew_eval:
            print(f"Skip loading validation trajectories ..... {self.myargs.save_name}")

        elif self.myargs.run_type == "main_opt_demos":

            # print(f"Loading policies and producing validation trajectories ..... {self.myargs.save_name}")
            # save_path_policy = os.path.join("./trained_models", self.myargs.algo, f"{self.myargs.algo}_GT_{self.myargs.env_name}-stacked")
            # names = listdir(save_path_policy)
            # eval_trajs_list = []
            # eval_trajs_return_list = []

            # if "devv" in self.myargs.save_name:
            #     skip = 50
            # else:
            #     skip = 5

            # for name in names[0:-1:skip]:
            #     address = os.path.join(save_path_policy,name)
            #     actor_critic, ob_rms = torch.load(address, map_location=self.device)
            #     actor_critic.to(self.device)

            #     produced_traj = self.produce_trajs_from_policy_sparsified_reward(actor_critic, 1, self.myargs.sparseness, self.init_params['produced_traj_length'], self.myargs.env_name, is_random=False)
            #     produced_traj, produced_traj_return = myutils.trajs_calc_return_no_device(produced_traj, self.myargs.discounted_rew, self.myargs.gamma)
            #     eval_trajs_list.append(produced_traj[0])
            #     eval_trajs_return_list.append(produced_traj_return[0]) 
            #     # for i in range(68):
            #     #     print (f'np.mean: { torch.mean( torch.tensor([item[0][0,i] for item in produced_traj[0] ]))  }')
            self.val_pairs, self.val_pairs_returns =[], []
            if self.myargs.train_from_given_ranked_demos:
                try:
                    self.val_pairs, self.val_pairs_returns, _ = myutils.create_pairs_distance_apart_device_memory(self.new_trajs_list_val, self.new_trajs_returns_list_val, self.init_params['num_eval_pairs'], self.myargs.priority_sampling, self.device, self.init_params["difference_factor"])
            #         # pairs, returns = myutils.create_pairs_no_step_no_subsample(ranked_traj_list, traj_returns, batch_size, self.myargs.priority_sampling)
            #         # if any pair is returned, the returns should be different as this is guaranteed in myutils.create_pairs_no_step
            #         # pairs.extend(pairs_raw), returns.extend(returns_raw)
                except Exception as e:
                    print("********************************************** \n \
                        ********************************* \n \
                        problems with create_pairs for evaluation ......")

        #     del eval_trajs_list
        #     del eval_trajs_return_list

        #     if myargs.num_opt_demo > 0 and not myargs.continue_:
        #         print(f"Loading policies and producing demo trajectories ..... {self.myargs.save_name}")
        #         # names = [myargs.saved_models_name+"_"+str(item)+".pt" for item in suffix_list]
        #         self.demos_opt, self.demos_opt_returns = [], []
        #         produced_trajs = []
        #         for name in names[-myargs.num_opt_demo:]:
        #             address = os.path.join(save_path_policy,name)
        #             actor_critic, ob_rms = torch.load(address, map_location=self.device)
        #             actor_critic.to(self.device)
        #             produced_traj = self.produce_trajs_from_policy_sparsified_reward(actor_critic, 1, self.myargs.sparseness, self.init_params['produced_traj_length'], self.myargs.env_name, is_random=False)
        #             produced_trajs.append(produced_traj[0])
        #         self._add_trajs_to_new_trajs_list_hardDrive(produced_trajs, 0, self.init_params['limited_buffer_length'], self.log_dir, add_only_non_zero_trajs=False, address=self.save_path_new_trajs)

        # else:
        #     pass

        # if self.myargs.num_overal_updates > 0: # if a non-zero value is provided for myargs.num_overal_updates, 
        #                                        # the value of self.init_params["num_overal_updates"] will be overwritten 
        #     self.init_params["num_overal_updates"] = self.myargs.num_overal_updates 

        #     print(f"Loading training trajectories ..... {self.myargs.save_name}")
        #     with open(trajs_address + '/all_trajectories', 'rb') as f:
        #         trajs_init = torch.load(f, map_location=self.device)

        #     trajs_total_num = len(trajs_init)
        #     traj_idxs_tr = np.arange(0, trajs_total_num, init_params['training_trajectory_skip'])
        #     traj_idxs_val = traj_idxs_tr[:-3] + 1
        #     demos_train_init = [trajs_init[idx] for idx in traj_idxs_tr]
        #     demos_val_init = [trajs_init[idx] for idx in traj_idxs_val]
        #     self.demos_train, self.demos_train_returns = myutils.trajs_calc_return_no_device(demos_train_init)
        #     self.demos_val, self.demos_val_returns = myutils.trajs_calc_return_no_device(demos_val_init)


        # self.ranked_trajs_list, self.returns = myutils.cut_trajs_calc_return_no_device(trajs, self.init_params['demo_horizon'])

        #  FOLLOWING 4 LINES ARE USED IF WE USE OLD METHOD OF SAMPLLING SUM-TRAJECTORIES FROM INITIAL DEMONSTRATIONS
        # self.demos_subsampled_list, self.demos_returns_subsampled = myutils.subsample_demos(trajs, self.init_params['subsample_length'], self.init_params['subsample_increment'], self.init_params['demo_horizon'])
        # self.demos_subsampled_list_val, self.demos_returns_subsampled_val = myutils.subsample_demos(trajs_val, self.init_params['subsample_length'], self.init_params['subsample_increment'], self.init_params['demo_horizon'])

        # self.demos_subsampled_list, self.demos_returns_subsampled = myutils.subsample_demos_true_return(trajs, self.init_params['subsample_length'], self.init_params['subsample_increment'], self.init_params['demo_horizon'])
        # self.demos_subsampled_list_val, self.demos_returns_subsampled_val = myutils.subsample_demos_true_return(trajs_val, self.init_params['subsample_length'], self.init_params['subsample_increment'], self.init_params['demo_horizon'])
        
        if not myargs.continue_:
            self._save_attributes_and_hyperparameters()

    def train(self):  

        # if self.myargs.pretrain in ["yes"]:
        #     print(f"pretraining the reward function ....{self.myargs.save_name}")
        #     # UPDATE REWARD based on initial set of demonstrations *****************************************************

        #     # Pre-train the policy based on the pre-trained reward
        #     if self.myargs.run_type == "pretrain_only":
        #         print(f"***************** Stopped right after pretraining *****************")
        #         return


        # MAIN TRAINING LOOP
        start_time = time.time()
        # num_pol_updates_at_each_overal_update = int(self.myargs.num_env_steps_tr) // self.myargs.num_steps // self.myargs.num_processes // self.init_params["num_overal_updates"]
        self.init_params["num_overal_updates"] = int(self.myargs.num_env_steps_tr) // self.myargs.num_steps // self.myargs.num_processes // self.init_params["num_pol_updates_at_each_overal_update"]
        num_pol_updates_at_each_overal_update = self.init_params["num_pol_updates_at_each_overal_update"]

        if self.myargs.continue_:
            start_rew_updates = True
            start_shaping = True
            add_only_non_zero_trajs = False # This is true until set to False
            start_pol_updates = True

            # read the last overal_rew_tr_iter_idx
            rew_weights_stats_loaded = np.loadtxt(f"{self.log_dir}/rew_weights_stats.txt", skiprows=1)
            overal_rew_tr_iter_idx = int(rew_weights_stats_loaded[-1,-1])
            policy_stats_loaded = np.loadtxt(f"{self.log_dir}/policy_stats.txt", skiprows=1)
            overal_pol_tr_iter_idx = int(policy_stats_loaded[-1,0])
            del rew_weights_stats_loaded
            del policy_stats_loaded

            # Load new_trajs_returns_list
            trajs_return_save_address =  f"{self.save_path_new_trajs}/trajs_returns_all.pkl"
            with open(trajs_return_save_address, "rb") as file:
                self.new_trajs_returns_list = pickle.load(file)

        else:
            start_rew_updates = False
            start_shaping = False
            add_only_non_zero_trajs = True # This is true until set to False
            overal_rew_tr_iter_idx = 0
            overal_pol_tr_iter_idx = 0
            if self.myargs.shaping:
                start_pol_updates = True
            else:
                start_pol_updates = False
                
        previous_overal_rew_tr_iter_idx = overal_rew_tr_iter_idx
        overal_rew_pretrain_iter_idx = 0 # this includes pre-training iterations
        total_time_policy = 0
        total_time_reward = 0

        while overal_rew_tr_iter_idx < self.init_params["num_overal_updates"] + previous_overal_rew_tr_iter_idx:
            print(f"Training iter number: {overal_rew_tr_iter_idx}")

            # UPDATE POLICY *****************************************************
            if "devv" in self.myargs.save_name:
                num_pol_updates_at_each_overal_update = 1
                # pass

            # note, the policy will not actually be updated if start_rew_updates == False
            if start_pol_updates:
                overal_pol_tr_iter_idx += 1

            self.update_policy(overal_pol_tr_iter_idx, overal_rew_tr_iter_idx, num_pol_updates_at_each_overal_update, 
                start_pol_updates, start_shaping, add_only_non_zero_trajs, dont_collect_trajs=False)

            # print(f"Memory usage in MiB line 576:   {self.memory_usage_psutil()}")
            # print("memory usage in line 576")
            # print(torch.cuda.memory_summary(abbreviated=True))

            # UPDATE REWARD *****************************************************
            if not start_rew_updates and np.count_nonzero(self.new_trajs_returns_list)>=self.myargs.min_good_trajs:
                # This if statement should only be executed once

                # the following line is to add some bad trajectories from the untrained policy to increase the variance in the buffer
                self.update_policy(overal_pol_tr_iter_idx, overal_rew_tr_iter_idx, num_pol_updates_at_each_overal_update, 
                    start_pol_updates, start_shaping, add_only_non_zero_trajs=False, dont_collect_trajs=False)
                
                if np.std(self.new_trajs_returns_list) > 0.0:
                    # we can't start reward learning unless there is a positive variance in the buffer
                    start_rew_updates = True
                    start_shaping = True
                    start_pol_updates = True
                    add_only_non_zero_trajs = False


            if start_rew_updates:
                overal_rew_tr_iter_idx += 1
                _, _ = self.update_reward(overal_rew_tr_iter_idx)
                # print(f"Memory usage in MiB line 599:   {self.memory_usage_psutil()}")
            else:
                print(f"No reward update for this iteration")


            # save both the policy and the reward every self.init_params['save_reward_int'] overal iterations
            if overal_rew_tr_iter_idx % self.init_params['save_reward_int'] == 0 and overal_rew_tr_iter_idx != 0 and start_rew_updates:

                # SAVE REWARD NETWORKS
                for reward_idx, reward_obj in enumerate(self.reward_objs):
                    # model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}_reward_{reward_idx}_iter_{overal_rew_tr_iter_idx}.pt")
                    model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}_reward_{reward_idx}.pt")

                    torch.save({'stage': 'train',
                                'model_state_dict': reward_obj.reward_net.state_dict(),
                                'optimizer_state_dict': reward_obj.optimizer.state_dict()}, 
                                model_save_address)

                # SAVE POLICY
                # model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}_policy_iter_{overal_rew_tr_iter_idx}.pt")
                model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}_policy.pt")

                torch.save([
                    self.actor_critic,
                    getattr(utils.get_vec_normalize(self.envs), 'ob_rms', None)
                ], model_save_address) 

            if overal_rew_tr_iter_idx == 0:
                overal_rew_pretrain_iter_idx += 1
                print(f"overal_rew_pretrain_iter_idx: {overal_rew_pretrain_iter_idx}")
            if overal_rew_pretrain_iter_idx > self.init_params["num_overal_updates"]/2:
                print("Pretrain took too long and training did not start. Breaking the program now ...")
                break # this breaks the while loop

            # # sample trajectories from the new policy and add to the buffer
            # produced_trajs = self.produce_trajs_from_policy_sparsified_reward(self.actor_critic, self.init_params['num_trajs_produced_each_iter'], self.myargs.sparseness, self.init_params['produced_traj_length'], is_random=False)
            # rew_mean, rew_range, rew_std, rew_mean_new, rew_range_new, rew_std_new = self._add_trajs_to_new_trajs_list_hardDrive(produced_trajs)
            # with open(self.log_dir + '/buffer_stats.txt', 'a') as f:
            #     f.write(f"{rew_mean:.10f} {rew_range:.10f} {rew_std:.10f} {rew_mean_new:.10f} {rew_range_new:.10f} {rew_std_new:.10f} \n")          

        total_training_time = time.time() - start_time
        print(f"Total training time: {total_training_time}")

        if not self.myargs.dont_remove_buffer:
            # REMOVE THE REPLAY BUFFER FROM THE HARD-DRIVE TO AVOID TAKING TOO MUCH SPACE
            save_path_new_trajs_rel = os.path.relpath(self.save_path_new_trajs, start = os.curdir)
            print ("REMOVE THE REPLAY BUFFER FROM THE HARD DRIVE TO AVOID TAKING TOO MUCH SPACE ...")
            os.system(f'rm -rf {save_path_new_trajs_rel}')

    def train_from_given_ranked_demos(self):  

        # if self.myargs.pretrain in ["yes"]:
        #     print(f"pretraining the reward function ....{self.myargs.save_name}")
        #     # UPDATE REWARD based on initial set of demonstrations *****************************************************

        #     # Pre-train the policy based on the pre-trained reward
        #     if self.myargs.run_type == "pretrain_only":
        #         print(f"***************** Stopped right after pretraining *****************")
        #         return



        # MAIN TRAINING LOOP
        start_time = time.time()
        # num_pol_updates_at_each_overal_update = int(self.myargs.num_env_steps_tr) // self.myargs.num_steps // self.myargs.num_processes 
        num_policy_updates = int(self.myargs.num_env_steps_tr) // self.myargs.num_steps // self.myargs.num_processes // self.init_params["num_pol_updates_at_each_overal_update"]


        start_rew_updates = True
        start_shaping = True
        overal_rew_tr_iter_idx = 0
        overal_pol_tr_iter_idx = 0
        start_pol_updates = True
        add_only_non_zero_trajs = False
                
        total_time_policy = 0
        total_time_reward = 0

        while overal_rew_tr_iter_idx < self.init_params['num_rew_updates']:
            print(f"Training iter number for reward: {overal_rew_tr_iter_idx}")
            self.update_reward(overal_rew_tr_iter_idx)
            overal_rew_tr_iter_idx += 1


        # SAVE REWARD NETWORKS
        for reward_idx, reward_obj in enumerate(self.reward_objs):
            # model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}_reward_{reward_idx}_iter_{overal_rew_tr_iter_idx}.pt")
            model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}_reward_{reward_idx}.pt")
            torch.save({'stage': 'train',
                        'model_state_dict': reward_obj.reward_net.state_dict(),
                        'optimizer_state_dict': reward_obj.optimizer.state_dict()}, 
                        model_save_address)


        while overal_pol_tr_iter_idx < num_policy_updates:
            print(f"Training iter number for policy: {overal_pol_tr_iter_idx}")
            self.update_policy(overal_pol_tr_iter_idx, overal_rew_tr_iter_idx, self.init_params["num_pol_updates_at_each_overal_update"], 
                start_pol_updates, start_shaping, add_only_non_zero_trajs, dont_collect_trajs=True)
            overal_pol_tr_iter_idx += 1

            # SAVE POLICY
            # model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}_policy_iter_{overal_rew_tr_iter_idx}.pt")
            model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}_policy.pt")
            torch.save([
                self.actor_critic,
                getattr(utils.get_vec_normalize(self.envs), 'ob_rms', None)
            ], model_save_address) 

    def update_reward(self, overal_tr_iter_idx):

        start = time.time()

        if self.myargs.use_linear_lr_decay_rew:
            # decrease learning rate linearly
            for reward_obj in self.reward_objs:
                utils.update_linear_schedule(
                    reward_obj.optimizer, overal_tr_iter_idx, self.init_params["num_overal_updates"],
                    self.myargs.rew_lr)
        elif self.myargs.use_increase_decrease_lr_rew:
            for reward_obj in self.reward_objs:
                utils.update_linear_schedule_increase_decrease(
                    reward_obj.optimizer, overal_tr_iter_idx, self.init_params["num_overal_updates"],
                    self.myargs.rew_lr)

         # Update g according to produced trajectories
        pairs, returns, pair_select_time_total, rew_update_time_total, load_trajs_time_total = self.grad_g_theta_update(overal_tr_iter_idx, self.init_params['num_rew_training_batches'], self.num_rew_nets, 
                            self.init_params['batch_size'], demos_or_policy='demos', pretrain_or_train="pretrain", discounted_rew=self.myargs.discounted_rew)

        # # save reward after each self.init_params['save_reward_int'] overal iterations
        # if overal_tr_iter_idx % self.init_params['save_reward_int'] == 0 and overal_tr_iter_idx != 0:
        #     for reward_idx, reward_obj in enumerate(self.reward_objs):
        #         model_save_address = os.path.join(self.save_path_trained_models, self.myargs.save_name + f"_reward_{reward_idx}_"  + str(overal_tr_iter_idx)  + ".pt")
        #         torch.save(reward_obj.reward_net, model_save_address)

        rew_time_total = time.time()-start

        print(f"rew_time_total: {rew_time_total}, pair_select_time_total: {pair_select_time_total}, load_trajs_time_total: {load_trajs_time_total}, in one training_iter \n")

        return pairs, returns

    def update_policy(self, overal_tr_iter_idx, overal_rew_tr_iter_idx, num_updates, start_pol_updates, start_shaping, 
        add_only_non_zero_trajs, dont_collect_trajs):

        kwargs, myargs = self.kwargs, self.myargs
        episode_rews_from_info = deque(maxlen=myargs.num_processes)
        episode_rew_net_return = deque(maxlen=myargs.num_processes)
        start = time.time()
        total_update_time = 0
        total_step_time = 0
        total_dense_rew_calc_time = 0

        if myargs.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                self.agent.optimizer, overal_tr_iter_idx, self.init_params["num_overal_updates"],
                self.agent.optimizer.lr if myargs.algo == "acktr" else myargs.lr)
        elif myargs.use_increase_decrease_lr_pol:
            utils.update_linear_schedule_increase_decrease(
                self.agent.optimizer, overal_tr_iter_idx, self.init_params["num_overal_updates"],
                self.agent.optimizer.lr if myargs.algo == "acktr" else myargs.lr)            

        unrolled_trajs_all = deque(maxlen=self.init_params["size_of_new_trajs_list"])
        no_run, no_run_no_cntr = self._specify_env_rew_type(self.is_atari)


        for j in range(num_updates):
            # At each j, each of the policy will be unrolled on each of the myargs.num_processes environments 
            # for myargs.num_steps steps

            return_nets_episode = torch.zeros((myargs.num_rew_nets, myargs.num_processes), device=self.device)
            return_GT_episode_cntr = np.zeros(myargs.num_processes)
            return_GT_episode_run = np.zeros(myargs.num_processes) 
            num_succ_run_forward = np.zeros(myargs.num_processes) 
            displacement_forward_till_rew = np.zeros(myargs.num_processes) 
            steps_till_rew = np.zeros(myargs.num_processes) 
            displacement_forward_episode_total = np.zeros(myargs.num_processes) 
            num_succ_not_done = np.zeros(myargs.num_processes)
            return_sparse_episode = np.zeros(myargs.num_processes)
            return_dense_plus_cntr_episode = np.zeros(myargs.num_processes)
            num_succ_run_forward_avg_steps = np.zeros(myargs.num_processes)
            num_succ_not_done_avg_steps = np.zeros(myargs.num_processes)
            num_steps_taken_to_rew = np.zeros(myargs.num_processes)
            displacement_total_from_infos = np.zeros(myargs.num_processes)

            unrolled_trajs = [[] for _ in range(myargs.num_processes)]


            for step in range(myargs.num_steps):

                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])

                if myargs.env_name in ["CartPole-v0", "MountainCar-v0", "Acrobot-v1"]: # Added by Farzan Memarian
                    # action = action[0]
                    action_fed = torch.squeeze(action)
                else:
                    action_fed = action

                step_time_start = time.time()
                obs, reward_GT, done, infos = self.envs.step(action_fed)
                total_step_time += time.time() - step_time_start

                # reward for current state action pair and next obs
                rews_run_step, rews_cntr_step = self._calc_rews_run_cntr_step(no_run_no_cntr, no_run, infos)
                if self.is_atari or myargs.sparse_rew_type == "GT":
                    reward_GT = torch.squeeze(reward_GT)
                    reward_sparse = reward_GT
                else:
                    (reward_sparse, displacement_forward_till_rew, steps_till_rew, 
                     displacement_forward_episode_total, num_succ_run_forward, 
                     num_steps_taken_to_rew, num_succ_not_done) = self.calc_sparse_reward(done, infos, displacement_forward_till_rew, 
                                                steps_till_rew, displacement_forward_episode_total, num_succ_run_forward, 
                                                num_steps_taken_to_rew, num_succ_not_done, reward_GT, myargs.num_processes, myargs)

                num_succ_run_forward_avg_steps += num_succ_run_forward
                num_succ_not_done_avg_steps += num_succ_not_done

                rew_calc_start_time = time.time()
                with torch.no_grad():
                    rew_nets_step_list = [reward_obj.reward_net(obs) for reward_obj in self.reward_objs]
                total_dense_rew_calc_time += time.time() - rew_calc_start_time

                for rew_idx in range(len(self.reward_objs)):
                    return_nets_episode[rew_idx, :] += rew_nets_step_list[rew_idx].reshape(myargs.num_processes)

                # add rewards of the networks to the network calculated returns

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rews_from_info.append(info['episode']['r'])
                        # info stores the undiscounted return of each trajectory

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])

                rews_nets_step = torch.mean(torch.cat(rew_nets_step_list, dim=1), dim=1)

                if self.myargs.sparse_cntr:
                    cntr_coeff = torch.zeros(len(rews_cntr_step), device=self.device)
                    non_zero_idxs = torch.nonzero(reward_sparse)

                    if non_zero_idxs.size()[0] > 0:
                        for idx in non_zero_idxs:
                            cntr_coeff[idx] = self.myargs.cntr_coeff
                else:
                    cntr_coeff = torch.ones(len(rews_cntr_step), device=self.device) * self.myargs.cntr_coeff


                if self.myargs.shaping:
                    if start_shaping:
                        # Whether we use discounting in learning a reward or not, we use discounting when applying shaping
                        total_rews_step = reward_sparse.to(self.device) + cntr_coeff * torch.tensor(rews_cntr_step, device=self.device) + self.myargs.gamma * rews_nets_step - self.old_rews_nets_step 
                    else:
                        total_rews_step = reward_sparse.to(self.device) + cntr_coeff * torch.tensor(rews_cntr_step, device=self.device) 
                else:
                    total_rews_step = self.myargs.rew_coeff * rews_nets_step + cntr_coeff * torch.tensor(rews_cntr_step, device=self.device)

                total_rews_GT_step = reward_sparse.to(self.device) + cntr_coeff * torch.tensor(rews_cntr_step, device=self.device)

                total_rews_step_torch = torch.unsqueeze(total_rews_step, dim=1)

                for idx_proc, _done in enumerate(done):
                    return_GT_episode_cntr[idx_proc] +=  rews_cntr_step[idx_proc] # * myargs.gamma**step 
                    return_GT_episode_run[idx_proc]  +=  rews_run_step[idx_proc] # * myargs.gamma**step 
                    return_sparse_episode[idx_proc] += reward_sparse[idx_proc]
                    return_dense_plus_cntr_episode[idx_proc] += total_rews_step[idx_proc].item()

                if step < self.init_params["produced_traj_length"] and not dont_collect_trajs:
                    for idx, obs_item in enumerate(obs):
                        unrolled_trajs[idx].append([obs_item.clone().detach().cpu(), reward_sparse[idx] ])

                if start_pol_updates:
                    self.rollouts.insert(obs, recurrent_hidden_states, action,
                                    action_log_prob, value, total_rews_step_torch, masks, bad_masks)

                self.old_rews_nets_step = rews_nets_step.clone()

            if not dont_collect_trajs:
                self._add_trajs_to_new_trajs_list_hardDrive(unrolled_trajs, overal_rew_tr_iter_idx, 
                    self.init_params['limited_buffer_length'], self.log_dir, add_only_non_zero_trajs, 
                    address=self.save_path_new_trajs, is_val=False) 
            # for traj in unrolled_trajs:
            #     unrolled_trajs_all.append(traj)

            return_GT_episode_total_my_calc = return_GT_episode_cntr + return_GT_episode_run
            #  END OF EIPISODE OR MAXIMUM NUMBER OF STEPS
            for idx in range(myargs.num_processes):
                episode_rew_net_return.append(torch.mean(return_nets_episode[:, idx]).item())

            if start_pol_updates:
                start_update = time.time()
                with torch.no_grad():
                    next_value = self.actor_critic.get_value(
                        self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                        self.rollouts.masks[-1]).detach()

                self.rollouts.compute_returns(next_value, myargs.use_gae, myargs.gamma,
                                     myargs.gae_lambda, myargs.use_proper_time_limits)

                value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

                num_succ_run_forward_avg_steps /= myargs.num_steps
                num_succ_not_done_avg_steps /= myargs.num_steps
                self.rollouts.after_update()

                total_update_time += time.time() - start_update

                if len(episode_rews_from_info) > 1:
                    total_num_steps = (j + 1) * myargs.num_processes * myargs.num_steps
                    end = time.time()
                    with open(self.log_dir + f"/policy_stats.txt", "a") as file: 
                        file.write( 
                            f'{overal_tr_iter_idx:>5} {j:>5} \
                            {total_num_steps:>8} {int(total_num_steps / (end - start)):.10f} \
                            {len(episode_rews_from_info):>4} {dist_entropy:.10} \
                            {value_loss:.10} {action_loss:.10}\
                            {np.mean(episode_rews_from_info):.10} {np.median(episode_rews_from_info):.10} \
                            {np.min(episode_rews_from_info):.10} {np.max(episode_rews_from_info):.10} {np.std(episode_rews_from_info):.10}\
                            {np.mean(episode_rew_net_return):.10} {np.std(episode_rew_net_return):.10} \
                            {np.mean(return_GT_episode_cntr):.10} {np.std(return_GT_episode_cntr):.10} \
                            {np.mean(return_GT_episode_run):.10}  {np.std(return_GT_episode_run):.10} \
                            {np.mean(return_sparse_episode):.10} {np.std(return_sparse_episode):.10} \
                            {np.mean(return_dense_plus_cntr_episode):.10} {np.std(return_dense_plus_cntr_episode):.10} \
                            {np.mean(num_succ_run_forward_avg_steps):.10} {np.std(num_succ_run_forward_avg_steps):.10} \
                            {np.mean(num_succ_not_done_avg_steps):.10} {np.std(num_succ_not_done_avg_steps):.10} \
                            {np.mean(displacement_forward_episode_total):.10} {np.std(displacement_forward_episode_total):.10} \n' )

        total_time = time.time()-start
        print(f"total policy_time in one training_iter: {total_time}")
        print(f"total policy update time in one overal training iter: {total_update_time}")
        print(f"total step time: {total_step_time}")
        print(f"total_dense_rew_calc_time: {total_dense_rew_calc_time}")

    def _add_trajs_to_new_trajs_list_hardDrive(self, produced_trajs, overal_tr_iter_idx, limited_buffer_length, log_dir, add_only_non_zero_trajs, address, is_val=False):
        """ 
        Stores new trajs into the hard drive to save memory 
        In the pretraining phase, it only adds trajectories with non-zero return to the buffer
        """
        # print("Saving produced trajs on hard drive ....")
        produced_trajs_list_all, produced_trajs_returns_all = myutils.trajs_calc_return_no_device(produced_trajs, self.myargs.discounted_rew, self.myargs.gamma)
        if add_only_non_zero_trajs:
            produced_trajs_list, produced_trajs_returns = [], []
            for traj, ret in zip(produced_trajs_list_all, produced_trajs_returns_all):
                if ret > 0:
                    produced_trajs_list.append(traj)
                    produced_trajs_returns.append(ret)
        else: 
            produced_trajs_list, produced_trajs_returns = produced_trajs_list_all, produced_trajs_returns_all

        if limited_buffer_length:
            start_idx, end_idx = self._calc_start_stop_and_update_new_trajs_last_idx(len(produced_trajs_list))
            for idx, total_idx in enumerate(np.arange(start_idx, end_idx+1)):
                traj_save_address =  f"{address}/traj_{total_idx}.pt"
                torch.save(produced_trajs_list[idx], traj_save_address)                 
            
            if not is_val:
                self.new_trajs_returns_list[start_idx:end_idx+1] = produced_trajs_returns[:]
            elif is_val:
                self.new_trajs_returns_list_val[start_idx:end_idx+1] = produced_trajs_returns[:]
        else:
            if not is_val:
                last_idx = len(self.new_trajs_returns_list)
                for idx, traj in enumerate(produced_trajs_list):
                    total_idx = idx + last_idx
                    traj_save_address =  f"{address}/traj_{total_idx}.pt"
                    torch.save(produced_trajs_list[idx], traj_save_address) 
                self.new_trajs_returns_list.extend(produced_trajs_returns)

            elif is_val:
                last_idx = len(self.new_trajs_returns_list_val)
                for idx, traj in enumerate(produced_trajs_list):
                    total_idx = idx + last_idx
                    traj_save_address =  f"{address}/traj_{total_idx}.pt"
                    torch.save(produced_trajs_list[idx], traj_save_address) 
                self.new_trajs_returns_list_val.extend(produced_trajs_returns)

        trajs_return_save_address =  f"{address}/trajs_returns_all.pkl"
        with open(trajs_return_save_address, "wb") as file:
            if not is_val:
                pickle.dump(self.new_trajs_returns_list, file)
            else:
                pickle.dump(self.new_trajs_returns_list_val, file)




        # Get statistics of the buffer and the new trajs
        if len(produced_trajs_returns) > 0:
            rew_mean = np.mean(self.new_trajs_returns_list)
            rew_range = np.abs(np.max(self.new_trajs_returns_list) - np.min(self.new_trajs_returns_list))
            rew_std = np.std(self.new_trajs_returns_list)
            rew_mean_new = np.mean(produced_trajs_returns)
            rew_range_new = np.abs(np.max(produced_trajs_returns) - np.min(produced_trajs_returns))
            rew_std_new = np.std(produced_trajs_returns)
        else:
            rew_mean_new, rew_range_new, rew_std_new = 0, 0, 0
            if len(self.new_trajs_returns_list) > 0:
                rew_mean = np.mean(self.new_trajs_returns_list)
                rew_range = np.abs(np.max(self.new_trajs_returns_list) - np.min(self.new_trajs_returns_list))
                rew_std = np.std(self.new_trajs_returns_list)
            else: 
                rew_mean, rew_range, rew_std = 0, 0, 0

        if log_dir != "NONE":
            with open( f"{log_dir}/buffer_stats.txt", 'a') as file:
                file.write(f"{rew_mean:.10f} {rew_range:.10f} {rew_std:.10f} {rew_mean_new:.10f} {rew_range_new:.10f} {rew_std_new:.10f} {overal_tr_iter_idx:>5}\n")

    def _add_trajs_to_new_trajs_list_memory_FIFO(self, produced_trajs, start_rew_updates):
        """ 
        first in first out 
        """
        produced_trajs_list_all, produced_trajs_returns_all = myutils.trajs_calc_return_no_device(produced_trajs, self.myargs.discounted_rew, self.myargs.gamma)
        if not start_rew_updates:
            produced_trajs_list, produced_trajs_returns = [], []
            for traj, rew in zip(produced_trajs_list_all, produced_trajs_returns_all):
                if rew > 0:
                    produced_trajs_list.append(traj)
                    produced_trajs_returns.append(rew)
        else: 
            produced_trajs_list, produced_trajs_returns = produced_trajs_list_all, produced_trajs_returns_all

        start_idx, end_idx = self._calc_start_stop_and_update_new_trajs_last_idx(len(produced_trajs_list))
        self.new_trajs_list[start_idx:end_idx] = produced_trajs_list[:]
        self.new_trajs_returns_list[start_idx:end_idx] = produced_trajs_returns[:]
        if len(produced_trajs_returns) > 0:
            rew_mean = np.mean(self.new_trajs_returns_list)
            rew_range = np.abs(np.max(self.new_trajs_returns_list) - np.min(self.new_trajs_returns_list))
            rew_std = np.std(self.new_trajs_returns_list)
            rew_mean_new = np.mean(produced_trajs_returns)
            rew_range_new = np.abs(np.max(produced_trajs_returns) - np.min(produced_trajs_returns))
            rew_std_new = np.std(produced_trajs_returns)
        else:
            rew_mean_new, rew_range_new, rew_std_new = 0, 0, 0
            if len(self.new_trajs_returns_list) > 0:
                rew_mean = np.mean(self.new_trajs_returns_list)
                rew_range = np.abs(np.max(self.new_trajs_returns_list) - np.min(self.new_trajs_returns_list))
                rew_std = np.std(self.new_trajs_returns_list)
            else: 
                rew_mean, rew_range, rew_std = 0, 0, 0
        return rew_mean, rew_range, rew_std, rew_mean_new, rew_range_new, rew_std_new

    def _add_trajs_to_new_trajs_list_memory_RW(self, produced_trajs):
        """
        only replaces the worst trajectories
        trajectories stored on ram
        RW --> stands for Replce worst
        """
        # poduced_trajs = myutils.produce_trajs_from_policy(self.actor_critic, self.init_params['num_trajs_produced_each_iter'], self.init_params['produced_traj_length'], self.kwargs, self.myargs)
        produced_trajs_list_all, produced_trajs_returns_all = myutils.trajs_calc_return_no_device(produced_trajs, self.myargs.discounted_rew, self.myargs.gamma)
        if not start_rew_updates:
            produced_trajs_list, produced_trajs_returns = [], []
            for traj, rew in zip(produced_trajs_list_all, produced_trajs_returns_all):
                if rew > 0:
                    produced_trajs_list.append(traj)
                    produced_trajs_returns.append(rew)
        else: 
            produced_trajs_list, produced_trajs_returns = produced_trajs_list_all, produced_trajs_returns_all
        
        start_idx, end_idx = self._calc_start_stop_and_update_new_trajs_last_idx_v1(len(produced_trajs_list))
        if len(self.new_trajs_returns_list) > 1:
            self.new_trajs_returns_list, self.new_trajs_list = (list(t) for t in zip(*sorted(zip(self.new_trajs_returns_list, self.new_trajs_list))))
        self.new_trajs_list[start_idx:end_idx] = produced_trajs_list[:]
        self.new_trajs_returns_list[start_idx:end_idx] = produced_trajs_returns[:]

        if len(produced_trajs_returns) > 0:
            rew_mean = np.mean(self.new_trajs_returns_list)
            rew_range = np.abs(np.max(self.new_trajs_returns_list) - np.min(self.new_trajs_returns_list))
            rew_std = np.std(self.new_trajs_returns_list)
            rew_mean_new = np.mean(produced_trajs_returns)
            rew_range_new = np.abs(np.max(produced_trajs_returns) - np.min(produced_trajs_returns))
            rew_std_new = np.std(produced_trajs_returns)
        else:
            rew_mean_new, rew_range_new, rew_std_new = 0, 0, 0
            if len(self.new_trajs_returns_list) > 0:
                rew_mean = np.mean(self.new_trajs_returns_list)
                rew_range = np.abs(np.max(self.new_trajs_returns_list) - np.min(self.new_trajs_returns_list))
                rew_std = np.std(self.new_trajs_returns_list)
            else: 
                rew_mean, rew_range, rew_std = 0, 0, 0
        return rew_mean, rew_range, rew_std, rew_mean_new, rew_range_new, rew_std_new

    def _calc_start_stop_and_update_new_trajs_last_idx(self, num_trajs):
        # first in first out
        if self.new_trajs_last_idx+num_trajs >= self.init_params["size_of_new_trajs_list"]:
            self.new_trajs_last_idx = self.myargs.num_opt_demo
        start_idx, end_idx = self.new_trajs_last_idx, self.new_trajs_last_idx+num_trajs-1
        self.new_trajs_last_idx += num_trajs
        return start_idx, end_idx

    def _calc_start_stop_and_update_new_trajs_last_idx_v1(self, num_trajs):
        # only replaces the last elements (of the sorted buffer)
        if self.new_trajs_last_idx+num_trajs <= self.init_params["size_of_new_trajs_list"]:
            start_idx, end_idx = self.new_trajs_last_idx, self.new_trajs_last_idx+num_trajs
            self.new_trajs_last_idx += num_trajs
        else:   
            start_idx, end_idx = self.init_params["size_of_new_trajs_list"]-1-num_trajs, self.init_params["size_of_new_trajs_list"]-1
        return start_idx, end_idx

    def grad_g_theta_update(self, overal_tr_iter_idx, num_batches, num_rew_nets, batch_size, demos_or_policy, pretrain_or_train, discounted_rew):
        """
        this function should only be called when it's possible to produce trajectory pairs from the buffer
        """
        # zero the gradient buffer for all the reward networks
            
        criterion = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')

        losses_all_nets = []
        accuracies_all_nets = []

        # print(f"***************** Updating reward_obj: {rew_obj_idx} \n")

        loss_per_rew_net = []
        accuracy_per_rew_net = []
        pairs_all = []
        returns_all = []
        pair_select_time_total = 0 
        load_trajs_time_total = 0
        rew_update_time_total = 0

        for batch_counter in range(num_batches):
            # Iterate over all reward networks for training
            for rew_obj_idx, reward_obj in enumerate(self.reward_objs):
                loss_item, accuracy, pairs, returns, pair_select_time, rew_update_time, time_loading_trajs = self._grad_individual_rew_obj(batch_size, reward_obj, criterion, discounted_rew)
                pair_select_time_total += pair_select_time
                rew_update_time_total += rew_update_time
                load_trajs_time_total += time_loading_trajs
                pairs_all.extend(pairs)
                returns_all.extend(returns)
                if loss_item != "no pair":
                    loss_per_rew_net.append(loss_item)
                    accuracy_per_rew_net.append(accuracy)

        # Here, after all updates, we write onto the rew_losses
        assert len(loss_per_rew_net) > 0
        mean_g = np.mean(loss_per_rew_net)
        std_g = np.std(loss_per_rew_net)
        mean_accuracy = np.mean(accuracy_per_rew_net)
        std_accuracy = np.std(accuracy_per_rew_net)
        with open(self.log_dir + f"/rew_losses.txt", "a") as file: 
            file.write(f" {mean_g:.10f} {std_g:.10f} {mean_accuracy:.10f} {std_accuracy:.10f} {pair_select_time_total:>5} {rew_update_time_total:>5} {overal_tr_iter_idx:>5} \n")


        # log magnitute of reward weights
        reward_weights_list = [torch.norm(reward_obj.get_flat_weights()) for reward_obj in self.reward_objs]
        reward_weights_mean = np.mean([item.item() for item in reward_weights_list])
        reward_weights_std = np.std([item.item() for item in reward_weights_list])
        reward_weights_min = np.min([item.item() for item in reward_weights_list])
        reward_weights_max = np.max([item.item() for item in reward_weights_list])
        with open(self.log_dir + f"/rew_weights_stats.txt", "a") as file: 
            file.write(f" {reward_weights_mean:.10f} {reward_weights_std:.10f} {reward_weights_min:.10f} {reward_weights_max:.10f} {overal_tr_iter_idx:>5} \n")


        # validation is performed after all batches are used for training
        if not self.myargs.skip_rew_eval:
            # Iterate over all reward networks for validation
            start = time.time()
            loss_per_rew_net = []
            accuracy_per_rew_net = []
            for rew_obj_idx, reward_obj in enumerate(self.reward_objs):
                if self.val_pairs:
                    loss_item, accuracy = self._individual_rew_obj_validation(self.val_pairs, self.val_pairs_returns, reward_obj, criterion, self.myargs.discounted_rew) 
                    loss_per_rew_net.append(loss_item)
                    accuracy_per_rew_net.append(accuracy)

            mean_g = np.mean(loss_per_rew_net)
            std_g = np.std(loss_per_rew_net)
            mean_accuracy = np.mean(accuracy_per_rew_net)
            std_accuracy = np.std(accuracy_per_rew_net)
            end = time.time()
            total_time_rew_eval = end - start
            with open(self.log_dir + f"/rew_losses_val.txt", "a") as file: 
                file.write(f" {mean_g:.10f} {std_g:.10f} {mean_accuracy:.10f} {std_accuracy:.10f} {total_time_rew_eval:>5} {overal_tr_iter_idx:>5} \n")
        return pairs_all, returns_all, pair_select_time_total, rew_update_time_total, load_trajs_time_total

    def _individual_rew_obj_validation(self, pairs, returns, reward_obj, criterion, discounted_rew):
        """
        uses the validation pairs and returns to compute the validaion accuracy of the reward networks
        """
        # ***********************************

        with torch.no_grad():
            return_traj_preds_list = []
            for (traj_i, traj_j), (rank_i, rank_j) in zip(pairs, returns):
                # return_traj = torch.zeros((num_pairs,1), requires_grad=True, device=self.device)
                # grad_theta = torch.zeros(agent.theta_size)

                # return_theta_traj_j = self.return_theta_traj_calc(traj_j return_traj_j, idx)
                # return_theta_traj_i = self.return_theta_traj_calc(traj_j return_traj_i, idx)
                
                assert rank_i != rank_j
                reward_input_batch_j = reward_obj.reward_net_input_batch_traj_method(traj_j)
                reward_input_batch_i = reward_obj.reward_net_input_batch_traj_method(traj_i)

                reward_output_batch_j = reward_obj.reward_net(reward_input_batch_j)
                reward_output_batch_i = reward_obj.reward_net(reward_input_batch_i)

                if discounted_rew:
                    num_rows = reward_output_batch_j.size()[0]
                    weights = torch.tensor([self.myargs.gamma**idx for idx in range(num_rows)], device=self.device)
                    weights = torch.unsqueeze(weights, dim=1)
                    reward_sum_j = torch.unsqueeze(torch.sum(weights * reward_output_batch_j, dim=0), dim=0) # element-wise multiplication
                    reward_sum_i = torch.unsqueeze(torch.sum(weights * reward_output_batch_i, dim=0), dim=0)
                else:
                    reward_sum_j = torch.unsqueeze(torch.sum(reward_output_batch_j, dim=0), dim=0)
                    reward_sum_i = torch.unsqueeze(torch.sum(reward_output_batch_i, dim=0), dim=0)

                if rank_j > rank_i:
                    return_sum_pair = torch.cat([reward_sum_j, reward_sum_i], dim=1) 
                    return_traj_preds_list.append(return_sum_pair)
                elif rank_j < rank_i:
                    return_sum_pair = torch.cat([reward_sum_i, reward_sum_j], dim=1)
                    return_traj_preds_list.append(return_sum_pair)


            if len(return_traj_preds_list) > 0:
                # update the reward function after every batch_size number of pairs
                return_traj_preds = torch.cat(return_traj_preds_list, dim=0)   
                high_return_idx = torch.zeros((len(return_traj_preds_list)), dtype=torch.long, requires_grad=False, device=self.device)
                accuracy = self.calc_accuracy(return_traj_preds)
                loss = criterion(return_traj_preds, high_return_idx)

                return loss.item(), accuracy
            else:
                return "no pair", "no pair"

    def _grad_individual_rew_obj(self, batch_size, reward_obj, criterion, discounted_rew):
        """
        reads collected returns from self.new_trajs_returns_list and uses self.save_path_new_trajs 
        """
        start = time.time()
        if self.sample_trajs_from_memory:                                                          
            pairs, returns, time_loading_trajs = myutils.create_pairs_distance_apart_device_memory(self.new_trajs_list, self.new_trajs_returns_list, batch_size, self.myargs.priority_sampling, self.device, self.init_params["difference_factor"])
        else:
            pairs, returns, time_loading_trajs = myutils.create_pairs_distance_apart_device_hardDrive(self.save_path_new_trajs, self.new_trajs_returns_list, batch_size, self.myargs.priority_sampling, self.device)
        pair_select_time = time.time() - start
        # pairs, returns = myutils.create_pairs_no_step_no_subsample(ranked_traj_list, traj_returns, batch_size, self.myargs.priority_sampling)
        # if any pair is returned, the returns should be different as this is guaranteed in myutils.create_pairs_no_step
        # pairs.extend(pairs_raw), returns.extend(returns_raw)

        # ***********************************
        start = time.time()
        reward_obj.reward_net.zero_grad()
        return_traj_preds_list = []
        if pairs:
            for (traj_i, traj_j), (rank_i, rank_j) in zip(pairs, returns):
                # return_traj = torch.zeros((num_pairs,1), requires_grad=True, device=self.device)
                # grad_theta = torch.zeros(agent.theta_size)

                # return_theta_traj_j = self.return_theta_traj_calc(traj_j return_traj_j, idx)
                # return_theta_traj_i = self.return_theta_traj_calc(traj_j return_traj_i, idx)
                
                assert rank_i != rank_j
                reward_input_batch_j = reward_obj.reward_net_input_batch_traj_method(traj_j)
                reward_input_batch_i = reward_obj.reward_net_input_batch_traj_method(traj_i)

                reward_output_batch_j = reward_obj.reward_net(reward_input_batch_j)
                reward_output_batch_i = reward_obj.reward_net(reward_input_batch_i)

                if discounted_rew:
                    num_rows = reward_output_batch_j.size()[0]
                    weights = torch.tensor([self.myargs.gamma**idx for idx in range(num_rows)], device=self.device)
                    weights = torch.unsqueeze(weights, dim=1)
                    reward_sum_j = torch.unsqueeze(torch.sum(weights * reward_output_batch_j, dim=0), dim=0) # element-wise multiplication
                    reward_sum_i = torch.unsqueeze(torch.sum(weights * reward_output_batch_i, dim=0), dim=0)
                else:
                    reward_sum_j = torch.unsqueeze(torch.sum(reward_output_batch_j, dim=0), dim=0)
                    reward_sum_i = torch.unsqueeze(torch.sum(reward_output_batch_i, dim=0), dim=0)

                if rank_j > rank_i:
                    return_sum_pair = torch.cat([reward_sum_j, reward_sum_i], dim=1) 
                    return_traj_preds_list.append(return_sum_pair)
                elif rank_j < rank_i:
                    return_sum_pair = torch.cat([reward_sum_i, reward_sum_j], dim=1)
                    return_traj_preds_list.append(return_sum_pair)

            # update the reward function after every batch_size number of pairs
            return_traj_preds = torch.cat(return_traj_preds_list, dim=0)   
            high_return_idx = torch.zeros((len(return_traj_preds_list)), dtype=torch.long, requires_grad=False, device=self.device)
            accuracy = self.calc_accuracy(return_traj_preds)
            loss = criterion(return_traj_preds, high_return_idx)
            loss.backward()
            reward_obj.optimizer.step()
            rew_update_time = time.time() - start
            return loss.item(), accuracy, pairs, returns, pair_select_time, rew_update_time, time_loading_trajs
        else:
            return "no pair", "no pair", "no pair", "no pair", pair_select_time, 0, 0

    def calc_accuracy(self, return_traj_preds):
        num_total = return_traj_preds.size()[0]
        num_correct = 0
        for i in range(num_total):
            if return_traj_preds[i,0] > return_traj_preds[i,1]:
                num_correct += 1
        return num_correct / num_total

    def calc_sparse_reward(self, done, infos, displacement_forward_till_rew, steps_till_rew, displacement_forward_episode_total, 
        num_succ_run_forward, num_steps_taken_to_rew, num_succ_not_done, reward_GT, num_envs, myargs):

        """
        unitsV2 uses myargs.num_steps as the maximum possible number of steps

        """
        sparseness = myargs.sparseness
        vel_thresh = 0
        reward_sparse = torch.zeros(num_envs)
        displacement_forward_step = np.zeros(num_envs)


        if myargs.sparse_rew_type == "steps":
            if myargs.env_name in ["InvertedPendulum-v2", "CartPole-v0"]:
                for idx, done_proc in enumerate(done):
                    if not done_proc: 
                        num_succ_not_done[idx] += 1 
                    else: 
                        num_succ_not_done[idx] = 0
                
                    if num_succ_not_done[idx] >= sparseness: 
                        reward_sparse[idx] = 1
                    else:  
                        reward_sparse[idx] = 0
            elif myargs.env_name == "InvertedDoublePendulum-v2":
                for idx, info in enumerate(infos):
                    angle_range = sparseness
                    angles = info["angles"]
                    if abs(angles[1]) < angle_range*math.pi/180:
                        reward_sparse[idx] = 1
                    else:
                        reward_sparse[idx] = 0
            else:
                for idx, info in enumerate(infos):
                    if info['reward_run'] > vel_thresh: 
                        num_succ_run_forward[idx] += 1 
                    else: 
                        num_succ_run_forward[idx] = 0
                    
                    if num_succ_run_forward[idx] >= sparseness: 
                        reward_sparse[idx] = 1
                    else:  
                        reward_sparse[idx] = 0

        elif myargs.sparse_rew_type in ["unitsV2", "units"]:
            for idx, info in enumerate(infos):
                displacement_forward_step[idx] = info["x_position"] - info["x_position_before"]
            displacement_forward_till_rew += displacement_forward_step
            displacement_forward_episode_total += displacement_forward_step
            steps_till_rew += np.ones(num_envs)

            for idx in range(np.shape(displacement_forward_till_rew)[0]):
                # reward_sparse[idx] = myargs.num_steps * (displacement_forward_till_rew[idx] // sparseness) / steps_till_rew[idx]
                if displacement_forward_till_rew[idx] > sparseness:
                    if myargs.sparse_rew_type == "unitsV2":
                        reward_sparse[idx] = (2 - (steps_till_rew[idx] / myargs.num_steps)) * (displacement_forward_till_rew[idx] // sparseness)  
                    elif myargs.sparse_rew_type == "units":
                        reward_sparse[idx] = displacement_forward_till_rew[idx] // sparseness
                    displacement_forward_till_rew[idx] = 0
                    steps_till_rew[idx] = 0
                else:
                    reward_sparse[idx] = 0

        # elif myargs.sparse_rew_type in ["episodic"]:
        #     steps_till_rew += np.ones(num_envs)
        #     if myargs.env_name in ["Reacher-v2", "MountainCar-v0"]:
        #         done = [item['done_dist'] for item in infos]
        #         for idx, done_proc in enumerate(done):
        #             if done_proc: 
        #                 reward_sparse[idx] = 1000/steps_till_rew[idx]
        #                 steps_till_rew[idx] = 0
        #             else:  
        #                 reward_sparse[idx] = 0 # This is redundant, reward_sparse is zero by default

        elif myargs.sparse_rew_type in ["episodic"]:
            if myargs.env_name in ["MountainCar-v0", "Reacher-v2", "Acrobot-v1", "Thrower-v2"] or myargs.env_name in ["HalfCheetah-v3", "Hopper-v3", "Walker2d-v3", "Swimmer-v3"]:
                done = [item['done_dist'] for item in infos]
                for idx, done_proc in enumerate(done):
                    if done_proc: 
                        reward_sparse[idx] = 1000



        else:
            raise Exception("Issues with myargs.sparse_rew_type")
            
        # IF done is reached, several variables need to be set to zero
        # Whether does is reached because of end of episode, or because of failure
        # or reaching a goal state
        for idx, done_ in enumerate(done):
            if done_:
                displacement_forward_till_rew[idx] = 0
                steps_till_rew[idx] = 0
                num_succ_run_forward[idx] = 0
                num_steps_taken_to_rew[idx] = 0
                # reward_sparse[idx] = 0    # Farzan: We don't need to manually set the sparse reward to zero, the masks will take care of this

        return reward_sparse, displacement_forward_till_rew, steps_till_rew, displacement_forward_episode_total, num_succ_run_forward, num_steps_taken_to_rew, num_succ_not_done

    def produce_trajs_from_policy_sparsified_reward(self, actor_critic, number_trajs, sparseness, traj_length, env_name, is_random):
        '''
        This function attaches the sparsified reward to the produced states. 
        '''
        
        kwargs, myargs = self.kwargs, self.myargs

        torch.set_num_threads(1)

        # CREATE AN ENVIRONMENT
        num_envs = 1
        log_dir = None
        seed = np.random.randint(0, 10000)
        env = make_vec_envs(myargs.env_name, seed, num_envs,
            myargs.gamma, log_dir, self.device, allow_early_resets=False, num_frame_stack=2,  **kwargs)
        # We need to use the same statistics for normalization as used in training
        # vec_norm = utils.get_vec_normalize(env)

        vel_thresh = 0

        all_trajs = []
        for traj_idx in range(number_trajs):
            traj = []
            recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)
            obs = env.reset()

            # LOOP FOR ONE TRAJECTORY
            transition_counter = 0
            # num_succ_run_forward = 0
            # num_succ_not_done = 0
            num_succ_run_forward = np.zeros(num_envs)
            num_succ_not_done = np.zeros(num_envs)
            displacement_forward_till_rew = np.zeros(num_envs)
            steps_till_rew = np.zeros(num_envs)
            displacement_forward_episode_total = np.zeros(num_envs)
            num_steps_taken_to_rew = np.zeros(num_envs)

            while transition_counter <= traj_length:
                transition_counter += 1
                if not is_random:
                    with torch.no_grad():
                        value, action, _, recurrent_hidden_states = actor_critic.act(
                            obs, recurrent_hidden_states, masks, deterministic=False)
                else:
                    action = env.action_space.sample()
                    action = torch.tensor(action, device=self.device)

                if env_name in ["CartPole-v0", "MountainCar-v0", "Acrobot-v1"]: # Added by Farzan Memarian
                    # action = action[0]
                    action_fed = torch.squeeze(action)
                else:
                    action_fed = action

                obs, reward_dense, done, infos = env.step(action_fed)

                (reward_sparse, displacement_forward_till_rew, steps_till_rew, displacement_forward_episode_total, 
                    num_succ_run_forward, num_steps_taken_to_rew, num_succ_not_done) = self.calc_sparse_reward(done, infos, displacement_forward_till_rew, 
                    steps_till_rew, displacement_forward_episode_total, num_succ_run_forward, num_steps_taken_to_rew, num_succ_not_done, reward_GT, num_envs, myargs)

                # if myargs.rew_cntr == "True":
                #     reward_cntr = torch.tensor(infos[0]['reward_cntr'], device=self.device)
                #     total_rews_step_sparse = reward_sparse + self.myargs.cntr_coeff * reward_cntr
                # elif myargs.rew_cntr == "False":
                traj.append([obs[0].clone().detach().cpu(), reward_sparse])
            # ALL TRAJS ARE PRODUCED AT THIS POINT
            all_trajs.append(traj)
        env.close()

        return all_trajs

    def produce_trajs_from_policy(self, actor_critic, number_trajs, traj_length, kwargs, myargs):

        # torch.set_num_threads(1)

        # print (f"******************** myargs.cuda_num: {myargs.cuda_num}")
        # print (f"******************** myargs.cuda: {myargs.cuda}")
        # CUDA_VISIBLE_DEVICES_raw_str = os.environ["CUDA_VISIBLE_DEVICES"]
        # print(f"******************** CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES_raw_str} ********************")
        # if not myargs.cuda_num:
        #     CUDA_VISIBLE_DEVICES_raw_str_splitted = CUDA_VISIBLE_DEVICES_raw_str.split(",")
        #     CUDA_VISIBLE_DEVICES_raw_str_splitted_list_int = [int(item) for item in CUDA_VISIBLE_DEVICES_raw_str_splitted]
        #     cuda_num = f"cuda" # :{CUDA_VISIBLE_DEVICES_raw_str_splitted_list_int[0]}
        #     print(f"******************** No cuda_num provided, cuda_num set to what slurm sets NVIDIA_VISIBLE_DEVICES")
        #     print(f"******************** cuda_num: {cuda_num}  ********************")
        #     self.device = torch.device(cuda_num if myargs.cuda else "cpu")
        # else:
        #     print(f"******************** cuda-num provided")
        #     self.device = torch.device(myargs.cuda_num if myargs.cuda else "cpu")
        # print(f"******************** self.device: {self.device} \n********************  type(self.device): {type(self.device)}   ")
        # print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

        # CREATE AN ENVIRONMENT
        log_dir = None
        seed = np.random.randint(0, 10000)
        num_envs = 1
        env = make_vec_envs(myargs.env_name, seed, num_envs,
            myargs.gamma, log_dir, self.device, allow_early_resets=True, num_frame_stack=2,  **kwargs)
        # We need to use the same statistics for normalization as used in training
        vec_norm = utils.get_vec_normalize(env)

        all_trajs = []
        for traj_idx in range(number_trajs):
            traj = []

            recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)
            obs = env.reset()
            # traj.append([obs,0])

            # LOOP FOR ONE TRAJECTORY
            transition_counter = 0
            while transition_counter <= traj_length:
                transition_counter += 1
                with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                        obs, recurrent_hidden_states, masks, deterministic=False)

                # Obser reward and next obs
                obs, reward, done, _ = env.step(action)
                # traj.append([copy.deepcopy(obs),copy.deepcopy(reward)])
                traj.append([obs[0].clone().detach().cpu(), copy.deepcopy(reward)])

            # ALL TRAJS ARE PRODUCED AT THIS POINT
            all_trajs.append(traj)
        env.close()

        return all_trajs

    def _save_attributes_and_hyperparameters(self):
        print ("saving attributes_and_hyperparameters .....")



        with open(self.log_dir + "/init_params.txt", "w") as file:
            for key in self.init_params:
                file.write(f"{key} : {self.init_params[key]} \n" )

        args_dict = vars(self.myargs)
        with open(self.log_dir +"/args_dict.pkl", "wb") as f:
            pickle.dump(args_dict, f)
        with open(self.log_dir + "/myargs.txt", "w") as file:
            for key in args_dict:
                file.write(f"{key} : {args_dict[key]} \n" )
    
        with open(self.log_dir +"/kwargs.pkl", "wb") as f:
            pickle.dump(self.kwargs, f)    
        with open(self.log_dir + "/kwargs.txt", "w") as file:
            for key in self.kwargs:
                file.write(f"{key} : {self.kwargs[key]} \n" )

    def _specify_env_rew_type(self, is_atari):
        no_run_no_cntr = False
        no_run = False
        if is_atari:
            no_run_no_cntr, no_run = True, True
        elif self.myargs.env_name in ["MountainCar-v0", "Acrobot-v1", "CartPole-v0",  "InvertedPendulum-v2", "InvertedDoublePendulum-v2"]:
            no_run_no_cntr = True
        elif self.myargs.env_name in ["Reacher-v2", "Thrower-v2",  "MountainCarContinuous-v0"]:
            no_run = True
        else:
            pass
        return no_run, no_run_no_cntr

    def _calc_rews_run_cntr_step(self, no_run_no_cntr, no_run, infos):
        if no_run_no_cntr:
            rews_cntr_step = [0 for item in infos]
            rews_run_step =  [0 for item in infos] 
        elif no_run:
            rews_cntr_step = [item['reward_cntr'] for item in infos]
            rews_run_step =  [0 for item in infos] 
        else:
            rews_cntr_step = [item['reward_cntr'] for item in infos]
            rews_run_step =  [item['reward_run'] for item in infos] 
        return rews_run_step, rews_cntr_step

    def memory_usage_psutil(self):
        # return the memory usage in MB
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / float(2 ** 20)
        return mem


class produce_ranked_trajs(train_sparse_rank):

    # ADDED BY FARZAN MEMARIAN 
    # ***************************** 
    """
    Purpose: Reads checkpointed policies and produces a given number of rollouts for each policy
    """

    def __init__(self, kwargs, myargs):
        if not myargs.seed == -12345: 
            # seed is provided as command line argument and nothing needs to be done
            pass
        else:        
            if os.getenv('SEED'):
                myargs.seed = int(os.getenv('SEED'))
            else:
                raise ValueError('SEED not provided as command line argument or as an enviornment variable')



        myargs.save_name = myargs.save_name + f"-s{myargs.seed}"
        myargs.saved_models_name = myargs.saved_models_name + f"-s{myargs.seed}"


        self.kwargs, self.myargs = kwargs, myargs
        # creating the folders
        self.saved_models_name = myargs.saved_models_name
        trajs_address = os.path.join("./ranked_trajs", myargs.save_name)
        self.save_path_new_trajs = os.path.join(trajs_address, "train")  # overwrites the self.save_path_new_trajs in the parent class
        self.save_path_new_trajs_val = os.path.join(trajs_address, "val")
        self.new_trajs_list = []
        self.new_trajs_returns_list = []
        self.new_trajs_returns_list_val = []
        self.new_trajs_last_idx = 0
        utils.create_dir(trajs_address)
        utils.create_dir(self.save_path_new_trajs)
        utils.create_dir(self.save_path_new_trajs_val)
        # num_updates = int(myargs.num_env_steps_tr) // myargs.num_steps // myargs.num_processes
        # suffix_list = np.arange(0, num_updates, myargs.save_interval)
        # names = [myargs.saved_models_name+"_"+str(item)+".pt" for item in suffix_list]
        from os import listdir
        from os.path import isfile, join
        # names.append(myargs.saved_models_name+"_"+str(num_updates-1)+".pt")
        policies = []

        log_dir = "NONE"
        torch.set_num_threads(1)

        self.device = myutils.assign_gpu_device(myargs)
        np.random.seed(myargs.seed)


        saved_models_address = os.path.join("./trained_models", myargs.algo, self.saved_models_name)
        names = listdir(saved_models_address)

        for name in names:
            address = os.path.join(saved_models_address,name)
            actor_critic, _ = torch.load(address, map_location=self.device)
            policies.append(actor_critic)
        # ***************************** END

        # if "devv" in myargs.save_name:
        #     num_skip_policy = 20 # for dev put this 10
        # else:
        #     num_skip_policy = 2

        validation_ratio = 10

        produced_trajs = []
        num_pol_samples = 1
        print(f"Loading policies and producing demo trajectories ..... {myargs.save_name}")
        for pol_counter, actor_critic in enumerate(policies[0:-1]):
            print(f"policy number: {pol_counter}")
            actor_critic.to(self.device)
            produced_traj = self.produce_trajs_from_policy_sparsified_reward(actor_critic, 1, myargs.sparseness, myargs.traj_length, myargs.env_name, is_random=False)
            if pol_counter % validation_ratio == 0 and pol_counter > 0:
                self._add_trajs_to_new_trajs_list_hardDrive([produced_traj[0]], 0, False, log_dir, add_only_non_zero_trajs=False, address=self.save_path_new_trajs_val, is_val=True)
            else:
                self._add_trajs_to_new_trajs_list_hardDrive([produced_traj[0]], 0, False, log_dir, add_only_non_zero_trajs=False, address=self.save_path_new_trajs, is_val=False)

        # save the returns
        with open(self.save_path_new_trajs  + "/new_trajs_returns_list.pkl", "wb") as f:
            pickle.dump(self.new_trajs_returns_list, f) 

        # produced_trajs = []
        # print(f"Loading policies and producing validation trajectories ..... {myargs.save_name}")
        # for counter, actor_critic in enumerate(policies[1:-1:num_skip_policy]):
        #     print(f"policy number: {counter}")
        #     actor_critic.to(self.device)
        #     produced_traj = self.produce_trajs_from_policy(actor_critic, 1, myargs.traj_length, kwargs, myargs)
        #     produced_trajs.append(produced_traj[0])
        # rew_mean, rew_range, rew_std, rew_mean_new, rew_range_new, rew_std_new = self._add_trajs_to_new_trajs_list_hardDrive(produced_trajs, add_only_non_zero_trajs=False, address=self.save_path_new_trajs_val, is_val=True)
        # # save the returns
        # with open(self.save_path_new_trajs_val + "/new_trajs_returns_list.pkl", "wb") as f:
        #     pickle.dump(self.new_trajs_returns_list_val, f) 



        # for policy_counter, policy in enumerate(policies):

        #     if policy_counter % num_skip_policy == 0:
        #         print(f"policy number: {policy_counter}")
        #         for _ in range(num_pol_samples):
        #             actor_critic = policy
        #             actor_critic = actor_critic.to(device)
        #             traj = []
        #             # CREATE AN ENVIRONMENT
        #             seed = np.random.randint(1,100000)
        #             env = make_vec_envs(myargs.env_name, seed, 1,
        #                 myargs.gamma, log_dir, device, False, num_frame_stack=2, **kwargs)

        #             # We need to use the same statistics for normalization as used in training
        #             vec_norm = utils.get_vec_normalize(env)
        #             if vec_norm is not None:
        #                 vec_norm.eval()
        #                 vec_norm.ob_rms = ob_rms
        #             recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
        #             masks = torch.zeros(1, 1)

        #             obs = env.reset()
        #             # LOOP FOR ONE TRAJECTORY
        #             done = False
        #             done_penalty = torch.tensor(-1000)  # TODO: this should be investigated
        #             episode_reward = 0
        #             counter = 0
        #             while counter <= myargs.traj_length:
        #                 counter += 1
        #                 with torch.no_grad():
        #                     value, action, _, recurrent_hidden_states = actor_critic.act(
        #                         obs, recurrent_hidden_states, masks, deterministic=False)
        #                 # env.render()

        #                 # Obser reward and next obs
        #                 obs, reward, done, info = env.step(action)
        #                 # if done:
        #                 #     reward = done_penalty
        #                 traj.append([copy.deepcopy(obs),reward])
        #                 # if 'episode' in info[0].keys():
        #                 #     episode_reward = info['episode']['r']

        #             # ALL TRAJS ARE PRODUCED AT THIS POINT
        #             all_trajs.append(traj)

        # with open(trajs_address + '/all_trajectories', 'wb') as f:
        #     torch.save(all_trajs, f)


class train_GT_policy(train_sparse_rank):

    def __init__(self, kwargs, myargs):
        # ADDED BY FARZAN MEMARIAN 

        # ***************************** START
        # log_dir = os.path.expanduser(myargs.log_dir) + "/" + myargs.save_name 
        self.kwargs, self.myargs = kwargs, myargs
        self.init_params = []
        self.log_dir = myargs.log_dir + "/" + myargs.save_name 
        self.eval_log_dir = self.log_dir + "_eval"
        log_file_name = myargs.env_name
        # utils.cleanup_log_dir(log_dir)
        # utils.cleanup_log_dir(eval_log_dir)
        utils.create_dir(self.log_dir)
        utils.create_dir(self.eval_log_dir)

        # save_path_policy is for storing the trained model
        save_path_policy = os.path.join(myargs.save_dir, myargs.algo, myargs.save_name)
        utils.create_dir(save_path_policy)

        # ***************************** END

        self._save_attributes_and_hyperparameters()


        torch.manual_seed(myargs.seed)
        torch.cuda.manual_seed_all(myargs.seed)
        np.random.seed(myargs.seed)

        if myargs.cuda and torch.cuda.is_available() and myargs.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)

        device = myutils.assign_gpu_device(myargs)

        envs = make_vec_envs(myargs.env_name, myargs.seed, myargs.num_processes,
            myargs.gamma, self.log_dir, device, allow_early_resets=False,  num_frame_stack=2,  **kwargs)

        # envs = gym.make(myargs.env_name, **kwargs)
        # envs = VecPyTorch(envs, device)
        # The followint block added by Farzan Memarian
        # envs = ProcgenEnv(num_envs=myargs.num_processes, env_name="heistpp", **kwargs)
        # if len(envs.observation_space.shape) == 3:
        #     envs = WarpFrame(envs, dict_space_key="rgb")
        # if len(envs.observation_space.shape) == 1 and myargs.do_normalize == "True":
        #     if gamma is None:
        #         envs = VecNormalize(envs, ret=False)
        #     else:
        #         envs = VecNormalize(envs, gamma=gamma)
        # envs = VecPyTorch(envs, device, myargs)

        if self.myargs.env_name in ["MountainCar-v0", "Reacher-v2", "Acrobot-v1", "Thrower-v2"]:
            hidden_size_policy = 10
        else:
            hidden_size_policy = 64

        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            device,
            base_kwargs={'recurrent': myargs.recurrent_policy, 'hidden_size': hidden_size_policy})
        actor_critic.to(device)

        if myargs.algo == 'a2c':
            agent = algo.A2C_ACKTR(
                actor_critic,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                alpha=myargs.alpha,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'ppo':
            agent = algo.PPO(
                actor_critic,
                myargs.clip_param,
                myargs.ppo_epoch,
                myargs.num_mini_batch,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'acktr':
            agent = algo.A2C_ACKTR(
                actor_critic, myargs.value_loss_coef, myargs.entropy_coef, acktr=True)

        if myargs.gail:
            assert len(envs.observation_space.shape) == 1
            discr = gail.Discriminator(
                envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
                device)
            file_name = os.path.join(
                myargs.gail_experts_dir, "trajs_{}.pt".format(
                    myargs.env_name.split('-')[0].lower()))
            
            expert_dataset = gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20)
            drop_last = len(expert_dataset) > myargs.gail_batch_size
            gail_train_loader = torch.utils.data.DataLoader(
                dataset=expert_dataset,
                batch_size=myargs.gail_batch_size,
                shuffle=True,
                drop_last=drop_last)

        rollouts = RolloutStorage(myargs.num_steps, myargs.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rews_from_info = deque(maxlen=10)

        start = time.time()
        num_updates = int(myargs.num_env_steps_tr) // myargs.num_steps // myargs.num_processes

        with open(self.log_dir + "/" + log_file_name  + ".txt", "w") as file: 
            file.write("Updates , num timesteps , FPS, number of Last training episodes, dist_entropy, value_loss, action_loss, mean reward, median reward, min reward, max reward \n")

        with open(self.eval_log_dir  + "/" + log_file_name  + "_eval.txt", "w") as file: 
            file.write("num_episodes, median_reward, max_reward \n")


        # UPDATE POLICY *****************************************************
        for j in range(num_updates):
            if j % 5 == 0 and j != 0:
                print (f'update number {j}, ...... {myargs.env_name}, total_time: {time.time()-start}')
            if myargs.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    agent.optimizer.lr if myargs.algo == "acktr" else myargs.lr)


            for step in range(myargs.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                # Obser reward and next obs

                if myargs.env_name in ["CartPole-v0", "MountainCar-v0", "Acrobot-v1"]: # Added by Farzan Memarian
                    # action = action[0]
                    action_fed = torch.squeeze(action)
                else:
                    action_fed = action

                obs, reward, done, infos = envs.step(action_fed)

                # reward = torch.zeros((8,1))

                # envs.render()
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rews_from_info.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(copy.deepcopy(obs), recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            # Update value function at the end of episode
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            if myargs.gail:
                if j >= 10:
                    envs.venv.eval()

                gail_epoch = myargs.gail_epoch
                if j < 10:
                    gail_epoch = 100  # Warm up
                for _ in range(gail_epoch):
                    discr.update(gail_train_loader, rollouts,
                                 utils.get_vec_normalize(envs)._obfilt)

                for step in range(myargs.num_steps):
                    rollouts.rewards[step] = discr.predict_reward(
                        rollouts.obs[step], rollouts.actions[step], myargs.gamma,
                        rollouts.masks[step])

            rollouts.compute_returns(next_value, myargs.use_gae, myargs.gamma,
                                     myargs.gae_lambda, myargs.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            # save policy 
            if (j % myargs.save_interval == 0 or j == num_updates - 1) and myargs.save_dir != "":

                model_save_address = os.path.join(save_path_policy, myargs.save_name + "_" + str(j) + ".pt")
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], model_save_address)

            if j % myargs.log_interval == 0 and len(episode_rews_from_info) > 1:
                total_num_steps = (j + 1) * myargs.num_processes * myargs.num_steps
                end = time.time()
                # print(
                #     "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                #     .format(j, total_num_steps,
                #             int(total_num_steps / (end - start)),
                #             len(episode_rews_from_info), np.mean(episode_rews_from_info),
                #             np.median(episode_rews_from_info), np.min(episode_rews_from_info),
                #             np.max(episode_rews_from_info), dist_entropy, value_loss,
                #             action_loss))
                # with open(log_dir + "/" + log_file_name  + ".txt", "a") as file2: 
                #     file2.write("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                #     .format(j, total_num_steps,
                #             int(total_num_steps / (end - start)),
                #             len(episode_rews_from_info), np.mean(episode_rews_from_info),
                #             np.median(episode_rews_from_info), np.min(episode_rews_from_info),
                #             np.max(episode_rews_from_info), dist_entropy, value_loss,
                #             action_loss))

                with open(self.log_dir + "/" + log_file_name  + ".txt", "a") as file: 
                    file.write(f'{j:>5} {total_num_steps:>8} {int(total_num_steps / (end - start)):>7} {len(episode_rews_from_info):>4}\
                     {dist_entropy:.10} {value_loss:.10} {action_loss:.10} {np.mean(episode_rews_from_info):.10} \
                     {np.median(episode_rews_from_info):.10} {np.min(episode_rews_from_info):.10}\
                      {np.max(episode_rews_from_info):.10}  \n')

            # if (myargs.eval_interval is not None and len(episode_rewards) > 1
            #         and j % myargs.eval_interval == 0):
            #     ob_rms = utils.get_vec_normalize(envs).ob_rms
            #     evaluate(actor_critic, ob_rms, myargs.env_name, myargs.seed,
            #              myargs.num_processes, self.eval_log_dir, device)


class train_baseline_sparse_rew(train_sparse_rank):

    def __init__(self, kwargs, myargs, init_params):
        if myargs.seed == -12345: # this means the seed is not provided by the user in command line arguments, it must be read from enviornment variables
            if os.getenv('SEED'):
                myargs.seed = int(os.getenv('SEED'))
            else:
                raise ValueError('SEED not provided as command line argument or as an enviornment variable')
        else:
            # seed is provided as command line argument and nothing needs to be done
            pass

        myargs.save_name = myargs.save_name + f"-s{myargs.seed}"


        self.kwargs = kwargs
        self.myargs = myargs
        self.init_params = init_params

        self.device = myutils.assign_gpu_device(self.myargs)

        # Read ranked trajs
        self.log_dir = myargs.log_dir + "/" + myargs.save_name 
        eval_log_dir = self.log_dir + "_eval"
        self.log_file_name = myargs.env_name
        # utils.cleanup_log_dir(log_dir)
        # utils.cleanup_log_dir(eval_log_dir)
        if not myargs.continue_:
            utils.create_dir(self.log_dir)
        # utils.create_dir(eval_log_dir)

        # self.save_path_trained_models is for storing the trained model
        self.save_path_trained_models = os.path.join(myargs.save_dir, myargs.algo, myargs.save_name)
        if not myargs.continue_:
            utils.create_dir(self.save_path_trained_models)

        # # Create forlder for tensorboard
        # self.writer = SummaryWriter(f'runs/visualization')

        torch.manual_seed(myargs.seed)
        torch.cuda.manual_seed_all(myargs.seed)
        np.random.seed(myargs.seed)

        if myargs.cuda and torch.cuda.is_available() and myargs.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)

        self.envs = make_vec_envs(myargs.env_name, myargs.seed, myargs.num_processes,
            myargs.gamma, self.log_dir, self.device, allow_early_resets=False, num_frame_stack=2, **kwargs)
        # envs = ProcgenEnv(num_envs=myargs.env_name, env_name="heistpp", **kwargs)

        self.is_atari = self.myargs.is_atari

        # envs = gym.make(myargs.env_name, **kwargs)

        if self.myargs.env_name in ["MountainCar-v0", "Reacher-v2", "Acrobot-v1", "Thrower-v2"]:
            hidden_size_policy = 10
        else:
            hidden_size_policy = 64

        if self.myargs.continue_:
            # Load the pretrained policy
            print(f'Loading policy for continuing run {self.myargs.save_name} .....')
            model_save_address_policy = os.path.join(self.save_path_trained_models, self.myargs.save_name + ".pt")
            self.actor_critic, ob_rms = torch.load(model_save_address_policy, map_location=self.device)

        else:
            self.actor_critic = Policy(
                self.envs.observation_space.shape,
                self.envs.action_space,
                self.device,
                base_kwargs={'recurrent': myargs.recurrent_policy, 'hidden_size': hidden_size_policy})
            
        self.actor_critic.to(self.device)


        if myargs.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                alpha=myargs.alpha,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'ppo':
            self.agent = algo.PPO(
                self.actor_critic,
                myargs.clip_param,
                myargs.ppo_epoch,
                myargs.num_mini_batch,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, myargs.value_loss_coef, myargs.entropy_coef, acktr=True)


        self.rollouts = RolloutStorage(myargs.num_steps, myargs.num_processes,
                              self.envs.observation_space.shape, self.envs.action_space,
                              self.actor_critic.recurrent_hidden_state_size)

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        if not myargs.continue_:
            with open(self.log_dir + f"/policy.txt", "w") as file: 
                file.write("overal_tr_iter_idx, updates , num timesteps , FPS, number of Last training episodes, dist_entropy, value_loss, action_loss, mean reward, median reward, min reward, max reward \n")

            self._save_attributes_and_hyperparameters()

    def train(self):  
        # UPDATE POLICY *****************************************************
        self.update_policy(pretrain_or_train="train")

    def update_policy(self, pretrain_or_train):

        if self.myargs.continue_:

            # read the last 
            policy_stats_loaded = np.loadtxt(f"{self.log_dir}/policy.txt", skiprows=1)
            last_j = int(policy_stats_loaded[-1,1])
            del policy_stats_loaded

            first_time_min_good_trajs = True
        else:
            last_j = 0  
            first_time_min_good_trajs = False

            

        kwargs, myargs = self.kwargs, self.myargs
        episode_rews_from_info = deque(maxlen=myargs.num_processes)

        self.unrolled_trajs_all = []

        start = time.time()

        sparseness = myargs.sparseness
        num_updates = int(myargs.num_env_steps_tr) // myargs.num_steps // myargs.num_processes
        vel_thresh = 0

        no_run, no_run_no_cntr = self._specify_env_rew_type(self.is_atari)

        for j in range(last_j+1, last_j+1+num_updates):
            # at each j, each of the policies will be unrolled for up to myargs.num_steps steps, or until they get to an
            # absorbing state (like failure)
            if j % 5 == 0:
                total_time = time.time() - start
                print(f"Policy update {myargs.save_name}:    {j},    total_time: {total_time}")


            if myargs.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    self.agent.optimizer, j, num_updates,
                    self.agent.optimizer.lr if myargs.algo == "acktr" else myargs.lr)

            num_succ_run_forward = np.zeros(myargs.num_processes)
            num_succ_not_done = np.zeros(myargs.num_processes)
            return_GT_episode_cntr = np.zeros(myargs.num_processes)
            return_GT_episode_run = np.zeros(myargs.num_processes)
            return_sparse_episode = np.zeros(myargs.num_processes)
            return_sparse_plus_cntr_episode = np.zeros(myargs.num_processes)
            return_GT_episode_total = np.zeros(myargs.num_processes)
            displacement_forward_till_rew = np.zeros(myargs.num_processes)
            steps_till_rew = np.zeros(myargs.num_processes)
            displacement_forward_episode_total = np.zeros(myargs.num_processes)
            num_steps_taken_to_rew = np.zeros(myargs.num_processes)

            unrolled_trajs = [[] for _ in range(myargs.num_processes)]

            for step in range(myargs.num_steps):

                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])

                if myargs.env_name in ["CartPole-v0", "MountainCar-v0", "Acrobot-v1"]: # Added by Farzan Memarian
                    # action = action[0]
                    action_fed = torch.squeeze(action)
                else:
                    action_fed = action

                # reward for current state action pair and next state
                obs, reward_GT, done, infos = self.envs.step(action_fed)

                rews_run_step, rews_cntr_step = self._calc_rews_run_cntr_step(no_run_no_cntr, no_run, infos)

                if self.is_atari or myargs.sparse_rew_type == "GT":
                    reward_GT = torch.squeeze(reward_GT)
                    reward_sparse = reward_GT
                else:
                    (reward_sparse, displacement_forward_till_rew, steps_till_rew, 
                     displacement_forward_episode_total, num_succ_run_forward, 
                     num_steps_taken_to_rew, num_succ_not_done) = self.calc_sparse_reward(done, infos, displacement_forward_till_rew, 
                                                steps_till_rew, displacement_forward_episode_total, num_succ_run_forward, 
                                                num_steps_taken_to_rew, num_succ_not_done, reward_GT, myargs.num_processes, myargs)

                if self.myargs.sparse_cntr:
                    cntr_coeff = torch.zeros(len(rews_cntr_step))
                    non_zero_idxs = torch.nonzero(reward_sparse)

                    if non_zero_idxs.size()[0] > 0:
                        for idx in non_zero_idxs:
                            cntr_coeff[idx] = self.myargs.cntr_coeff
                else:
                    cntr_coeff = torch.ones(len(rews_cntr_step)) * self.myargs.cntr_coeff

                total_rews_step_sparse = reward_sparse + cntr_coeff * torch.tensor(rews_cntr_step)
                total_rews_step_sparse_tensor = torch.unsqueeze(total_rews_step_sparse, dim=1)

                # add rewards of the networks to the network calculated returns
                for idx_proc, _done in enumerate(done):
                    # if not _done:
                    return_GT_episode_cntr[idx_proc] +=  rews_cntr_step[idx_proc] # * myargs.gamma**step 
                    return_GT_episode_run[idx_proc]  +=  rews_run_step[idx_proc] # * myargs.gamma**step 
                    return_GT_episode_total[idx_proc]  +=  reward_GT[idx_proc] # * myargs.gamma**step 
                    return_sparse_episode[idx_proc] += reward_sparse[idx_proc].item()
                    return_sparse_plus_cntr_episode[idx_proc] += reward_sparse[idx_proc] + cntr_coeff[idx_proc] * rews_cntr_step[idx_proc]

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rews_from_info.append(info['episode']['r'])
                        # info stores the undiscounted return of each trajectory

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])

                if first_time_min_good_trajs:
                    # total_rews_step_sparse_torch_unsqueeze = torch.unsqueeze(total_rews_step_sparse_torch, dim=1)
                    self.rollouts.insert(obs.clone(), recurrent_hidden_states, action,
                                    action_log_prob, value, total_rews_step_sparse_tensor, masks, bad_masks)
                # self.rollouts.insert(obs, recurrent_hidden_states, action,
                #                 action_log_prob, value, reward_GT, masks, bad_masks)

                if not first_time_min_good_trajs:
                    for idx, obs_item in enumerate(obs):
                        unrolled_trajs[idx].append([obs_item.clone(), recurrent_hidden_states[idx], action[idx],
                                    action_log_prob[idx], value[idx], total_rews_step_sparse_tensor[idx], masks[idx], bad_masks[idx], reward_sparse[idx] ])


            if len(self.unrolled_trajs_all) < myargs.min_good_trajs and not first_time_min_good_trajs:
                self._add_trajs_with_positive_sparse_return(unrolled_trajs)

            if len(self.unrolled_trajs_all) >= myargs.min_good_trajs and not first_time_min_good_trajs:
                first_time_min_good_trajs = True

                # make sure there are exactly myargs.num_processes trajectories to be inserted into self.rollouts
                if len(self.unrolled_trajs_all) < myargs.num_processes:
                    num_normal_trajs_needed = myargs.num_processes - len(self.unrolled_trajs_all)
                    normal_trajs = self._produce_normal_trajs()
                    for idx in range(num_normal_trajs_needed):
                        self.unrolled_trajs_all.append(normal_trajs[idx])
                else:
                    self.unrolled_trajs_all = self.unrolled_trajs_all[:myargs.num_processes]


                # add pre-training trajs into the rollouts
                for idx in range(myargs.num_steps):
                    obs = torch.cat([torch.unsqueeze(traj[idx][0].clone().detach(), dim=0) for traj in self.unrolled_trajs_all], dim=0) 
                    recurrent_hidden_states = torch.cat([torch.unsqueeze(traj[idx][1].clone().detach(), dim=0) for traj in self.unrolled_trajs_all], dim=0) 
                    action = torch.cat([torch.unsqueeze(traj[idx][2].clone().detach(), dim=0) for traj in self.unrolled_trajs_all], dim=0) 
                    action_log_prob  = torch.cat([torch.unsqueeze(traj[idx][3].clone().detach(), dim=0) for traj in self.unrolled_trajs_all], dim=0) 
                    value = torch.cat([torch.unsqueeze(traj[idx][4].clone().detach(), dim=0) for traj in self.unrolled_trajs_all], dim=0) 
                    total_rews_step_sparse_torch = torch.cat([torch.unsqueeze(traj[idx][5].clone().detach(), dim=0) for traj in self.unrolled_trajs_all], dim=0) 
                    masks  = torch.cat([torch.unsqueeze(traj[idx][6].clone().detach(), dim=0) for traj in self.unrolled_trajs_all], dim=0) 
                    bad_masks = torch.cat([torch.unsqueeze(traj[idx][7].clone().detach(), dim=0) for traj in self.unrolled_trajs_all], dim=0) 
                    self.rollouts.insert(obs, recurrent_hidden_states, action,
                                                        action_log_prob, value, total_rews_step_sparse_torch, masks, bad_masks)



            # return_GT_episode_total_my_calc = return_GT_episode_cntr + return_GT_episode_run
            #  END OF EIPISODE OR MAXIMUM NUMBER OF STEPS
            if first_time_min_good_trajs:
                with torch.no_grad():
                    next_value = self.actor_critic.get_value(
                        self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                        self.rollouts.masks[-1]).detach()

                self.rollouts.compute_returns(next_value, myargs.use_gae, myargs.gamma,
                                         myargs.gae_lambda, myargs.use_proper_time_limits)

                value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

                self.rollouts.after_update()
                if len(episode_rews_from_info) > 1:
                    total_num_steps = (j + 1) * myargs.num_processes * myargs.num_steps
                    end = time.time()
                    with open(self.log_dir + f"/policy.txt", "a") as file: 
                        file.write( 
                            f'{0.00:>5} {j:>5} {total_num_steps:>8} \
                            {int(total_num_steps / (end - start)):.10f} {len(episode_rews_from_info):>4} \
                            {dist_entropy:.10} {value_loss:.10} {action_loss:.10} \
                            {np.mean(episode_rews_from_info):.10} {np.median(episode_rews_from_info):.10} \
                            {np.min(episode_rews_from_info):.10} {np.max(episode_rews_from_info):.10} \
                            {np.std(episode_rews_from_info):.10}\
                            {np.mean(return_sparse_episode):.10} {np.std(return_sparse_episode):.10}\
                            {np.mean(return_GT_episode_cntr):.10} {np.std(return_GT_episode_cntr):.10} \
                            {np.mean(return_GT_episode_run):.10}  {np.std(return_GT_episode_run):.10} \
                            {np.mean(return_GT_episode_total):.10}  {np.std(return_GT_episode_total):.10} \
                            {np.mean(return_sparse_plus_cntr_episode):.10}  {np.std(return_sparse_plus_cntr_episode):.10} \
                            {np.mean(displacement_forward_episode_total):.10}  {np.std(displacement_forward_episode_total):.10} \n' )


                if (j+1) % self.myargs.save_interval == 0:
                    if self.myargs.save_every_policy:
                        model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}_{j}.pt")
                    else:
                        model_save_address = os.path.join(self.save_path_trained_models, f"{self.myargs.save_name}.pt")
                    torch.save([
                        self.actor_critic,
                        getattr(utils.get_vec_normalize(self.envs), 'ob_rms', None)
                    ], model_save_address) 

    def _add_trajs_with_positive_sparse_return(self, unrolled_trajs):
        for traj in unrolled_trajs:
            traj_return = np.sum([item[-1] for item in traj])
            if traj_return > 0:
                self.unrolled_trajs_all.append(traj)

    def _produce_normal_trajs(self):
        myargs, kwargs = self.myargs, self.kwargs

        episode_rews_from_info = deque(maxlen=myargs.num_processes)
        episode_rew_net_return = deque(maxlen=myargs.num_processes)

        sparseness = myargs.sparseness
        vel_thresh = 0

        no_run, no_run_no_cntr = self._specify_env_rew_type(self.is_atari)

        num_succ_run_forward = np.zeros(myargs.num_processes)
        num_succ_not_done = np.zeros(myargs.num_processes)
        return_GT_episode_cntr = np.zeros(myargs.num_processes)
        return_GT_episode_run = np.zeros(myargs.num_processes)
        return_sparse_episode = np.zeros(myargs.num_processes)
        return_sparse_plus_cntr_episode = np.zeros(myargs.num_processes)
        return_GT_episode_total = np.zeros(myargs.num_processes)
        displacement_forward_till_rew = np.zeros(myargs.num_processes)
        steps_till_rew = np.zeros(myargs.num_processes)
        displacement_forward_episode_total = np.zeros(myargs.num_processes)
        num_steps_taken_to_rew = np.zeros(myargs.num_processes)

        unrolled_trajs = [[] for _ in range(myargs.num_processes)]

        for step in range(myargs.num_steps):

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                    self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                    self.rollouts.masks[step])

            if myargs.env_name in ["CartPole-v0", "MountainCar-v0", "Acrobot-v1"]: # Added by Farzan Memarian
                # action = action[0]
                action_fed = torch.squeeze(action)
            else:
                action_fed = action

            # reward for current state action pair and next state
            obs, reward_GT, done, infos = self.envs.step(action_fed)

            if self.is_atari or myargs.sparse_rew_type == "GT":
                reward_GT = torch.squeeze(reward_GT)
                reward_sparse = reward_GT
            else:
                (reward_sparse, displacement_forward_till_rew, steps_till_rew, 
                 displacement_forward_episode_total, num_succ_run_forward, 
                 num_steps_taken_to_rew, num_succ_not_done) = self.calc_sparse_reward(done, infos, displacement_forward_till_rew, 
                                            steps_till_rew, displacement_forward_episode_total, num_succ_run_forward, 
                                            num_steps_taken_to_rew, num_succ_not_done, reward_GT, myargs.num_processes, myargs)
           
            rews_run_step, rews_cntr_step = self._calc_rews_run_cntr_step(no_run_no_cntr, no_run, infos)


            if self.myargs.sparse_cntr:
                cntr_coeff = torch.zeros(len(rews_cntr_step))
                non_zero_idxs = torch.nonzero(reward_sparse)

                if non_zero_idxs.size()[0] > 0:
                    for idx in non_zero_idxs:
                        cntr_coeff[idx] = self.myargs.cntr_coeff
            else:
                cntr_coeff = torch.ones(len(rews_cntr_step)) * self.myargs.cntr_coeff

            total_rews_step_sparse = reward_sparse + cntr_coeff * torch.tensor(rews_cntr_step)
            total_rews_step_sparse_tensor = torch.unsqueeze(total_rews_step_sparse, dim=1)


            # add rewards of the networks to the network calculated returns
            for idx_proc, _done in enumerate(done):
                if not _done:
                    return_GT_episode_cntr[idx_proc] +=  rews_cntr_step[idx_proc] # * myargs.gamma**step 
                    return_GT_episode_run[idx_proc]  +=  rews_run_step[idx_proc] # * myargs.gamma**step 
                    return_GT_episode_total[idx_proc]  +=  reward_GT[idx_proc] # * myargs.gamma**step 
                    return_sparse_episode[idx_proc] += reward_sparse[idx_proc].item()
                    return_sparse_plus_cntr_episode[idx_proc] += reward_sparse[idx_proc] + cntr_coeff[idx_proc] * rews_cntr_step[idx_proc]


            for info in infos:
                if 'episode' in info.keys():
                    episode_rews_from_info.append(info['episode']['r'])
                    # info stores the undiscounted return of each trajectory

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            for idx, obs_item in enumerate(obs):
                unrolled_trajs[idx].append([obs_item.clone(), recurrent_hidden_states[idx], action[idx],
                            action_log_prob[idx], value[idx], total_rews_step_sparse_tensor[idx], masks[idx], bad_masks[idx], reward_sparse[idx] ])

        return unrolled_trajs


def main():
    myargs, kwargs = arguments.get_args()
    # print("After get_args")

    if myargs.main_function == "train_GT_policy":
        train_GT_policy(kwargs, myargs)

    if myargs.main_function == "train_GAIL":
        train_GAIL(kwargs, myargs)

    elif myargs.main_function == "produce_ranked_trajs":
        produce_ranked_trajs(kwargs, myargs)

    elif myargs.main_function == "produce_ranked_trajs_sparse":
        init_params = arguments.get_init(myargs)
        produce_ranked_trajs_sparse(kwargs, myargs, init_params)

    elif myargs.main_function == "train_sparse_rank":
        init_params = arguments.get_init(myargs)
        train_obj = train_sparse_rank(kwargs, myargs, init_params)
        if myargs.train_from_given_ranked_demos:
            train_obj.train_from_given_ranked_demos()
        else:
            train_obj.train()

    elif myargs.main_function == "train_baseline_sparse_rew":
        init_params = arguments.get_init_baseline(myargs)
        train_obj = train_baseline_sparse_rew(kwargs, myargs, init_params)
        train_obj.train()       

    elif myargs.main_function == "evaluate_policy_noisy_env":
        evaluate_policy_noisy_env_obj = evaluate_policy_noisy_env()
        evaluate_policy_noisy_env_obj.eval_func(myargs.eval_name)


    elif myargs.main_function == "visualize_policy":
        train_obj = visualize_policy(kwargs, myargs)


if __name__ == "__main__":
    main()






