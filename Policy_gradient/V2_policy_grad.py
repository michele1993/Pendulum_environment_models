import torch
import gym
import torch.optim as opt
import numpy as np


from torch.distributions import Normal


from Policy_gradient.Policy_NN import Agent_nn
from Policy_gradient.Baseline_NN import Baseline_NN

n_episodes= 10000
discount = 0.99
max_t_step = 200
learning_rate = 1e-3
batch_size = 1
std = 0.3

env = gym.make("Pendulum-v0")
pol_nn = Agent_nn().double()
base_nn = Baseline_NN().double()


params = list(pol_nn.parameters()) + list(base_nn.parameters())

optimiser = opt.Adam(params,learning_rate)

episode_overall_return = []


best_rwd = []


for i in range(n_episodes):

    current_st = env.reset()
    episode_rwd = torch.empty(0)
    episode_lp_action = torch.empty(0).float() #[]
    episode_states = np.empty(0)
    undiscount_rwd = []




    for t in range(max_t_step): #max_t_step

        episode_states= np.concatenate((episode_states,current_st),axis=0)

        mean_action = pol_nn(torch.tensor(current_st))

        d = Normal(mean_action,std) # try to replace with bernulli and single output

        action = d.sample()

        episode_lp_action = torch.cat([episode_lp_action,torch.unsqueeze(d.log_prob(action).float(),dim=-1)])

        next_st, rwd, done, _ = env.step(action.numpy())

        episode_rwd = episode_rwd * discount

        episode_rwd = torch.cat((episode_rwd,torch.tensor([rwd])),dim=-1)

        undiscount_rwd.append(rwd)


        if done:
            break

        current_st = next_st


    best_rwd.append(np.max(undiscount_rwd))

    predicted_value = base_nn(torch.tensor(episode_states.reshape(-1,3)))


    #episode_rwd = np.flip(np.cumsum(np.flip(episode_rwd)))

    episode_rwd = torch.flip(torch.cumsum(torch.flip(episode_rwd, (0,)), 0), (0,))


    advantage = episode_rwd.view(-1) - predicted_value.view(-1) # v_value


    # Update policy net

    policy_c = pol_nn.REINFORCE(episode_lp_action,advantage)

    baseline_c = sum(torch.pow(advantage, 2))

    loss =  policy_c + baseline_c  #pol_nn.REINFORCE(episode_lp_action[e],advantage) + torch.pow(episode_rwd[e] - episode_v_value[e], 2)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    episode_overall_return.append(sum(undiscount_rwd))


    if i % 100 == 0:

        print("{}, Baseline loss {}, Policy cost {}, Return {}, best score {}".format(i, baseline_c.data,policy_c.data,sum(episode_overall_return)/100, sum(best_rwd) /100))

        episode_overall_return = []
        best_rwd = []
