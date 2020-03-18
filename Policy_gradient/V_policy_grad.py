import torch
import gym
import torch.optim as opt
import numpy as np




#from Feed_forward.Inv_Pendulum_Wrapper import ModifiedInvPendulum
from Feed_forward.Parall_Inv_Pend_Wrapper import Parallel_Mod_Pendulum
from Policy_gradient.Baseline_NN import Baseline_NN
from Policy_gradient.Policy_NN import Agent_nn
from torch.distributions import Normal
import torch.optim as opt



n_episodes= 30000
discount = 0.99
max_t_step = 200
learning_rate = 1e-3
sampl_std = 0.3
n_envs = 1


#env = Parallel_Mod_Pendulum(lambda: gym.make("Pendulum-v0"), num_envs = n_envs)

env = gym.make("Pendulum-v0")


pol_nn = Agent_nn().double()
base_nn = Baseline_NN().double()

params = list(pol_nn.parameters()) + list(base_nn.parameters())

optimiser = opt.Adam(params,learning_rate)




episode_overall_return = []

best_rwd = []


for i in range(n_episodes):



    current_st = env.reset()#.reshape(n_envs,-1)
    episode_rwd =  np.zeros(max_t_step) #np.empty((n_envs,1))
    episode_v_value = []
    episode_lp_acts = []
    undiscount_rwd = []


    t = 0

    for t in range(max_t_step): #max_t_step

        mean_action = pol_nn(torch.tensor(current_st))

        d = Normal(mean_action, sampl_std)

        action = d.sample()

        episode_lp_acts.append(d.log_prob(action))

        next_st,rwd, _, _ = env.step(action.numpy())

        predicted_value = base_nn(torch.tensor(current_st))

        episode_rwd = episode_rwd * discount

        episode_rwd = np.concatenate((episode_rwd,np.array([rwd])))

        episode_v_value.append(predicted_value)

        undiscount_rwd.append(rwd)

        current_st = next_st


    best_rwd.append(np.max(undiscount_rwd))

    policy_c = 0

    baseline_c = 0

    episode_overall_return.append(sum(undiscount_rwd))

    episode_rwd = np.flip(np.cumsum(np.flip(episode_rwd)))

    #episode_rwd = torch.flip(torch.cumsum(torch.flip(episode_rwd, (0,)), 0), (0,))  # works correctly


    # perform update for each time step
    for e in range(t+1):


        advantage = episode_rwd[e] - episode_v_value[e] # v_value

        policy_c += pol_nn.REINFORCE(episode_lp_acts[e],advantage)

        baseline_c += torch.pow(episode_rwd[e] - episode_v_value[e], 2)


    loss = policy_c + baseline_c

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()



    if i % 100 == 0:

        print("{}, Baseline loss {}, Policy cost {}, Return {}, best score {}".format(i, baseline_c[0],policy_c[0],sum(episode_overall_return)/100, sum(best_rwd) /100))

        episode_overall_return = []
        best_rwd = []

