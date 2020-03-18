import gym
import torch
import torch.optim as op
import numpy as np

from Feed_forward.Inv_Pendulum_Wrapper import ModifiedInvPendulum
from Feed_forward.Parall_Inv_Pend_Wrapper import Parallel_Mod_Pendulum

from torch.distributions import Normal



#b_env = gym.make("Pendulum-v0")
#env = ModifiedInvPendulum(b_env)


t_steps = 200
discount = 0.95
n_episodes = 10000
ln_rate = 0.0001
ln_rate2 = 0.1
N_var = 0.1
n_envs = 5


env = Parallel_Mod_Pendulum(lambda: gym.make("Pendulum-v0"), num_envs = n_envs)

mean_vector = torch.tensor(np.random.randn(t_steps), requires_grad=True)





# print(env.unwrapped)
# print(env.env.env)
# print(env.env.env.__dict__)



sum_rwd=[]

for ep in range(1,n_episodes):


    env.reset()

    log_p_actions = []

    #av_collected_rwd = np.zeros((t_steps, n_envs))

    av_collected_rwd = torch.zeros((t_steps, n_envs))




    for i in range(t_steps):

        #env.render()

        d = Normal(mean_vector[i],N_var)

        #sampled_a = d.sample()

        sampled_as = d.sample((n_envs,))



        log_p_actions.append(d.log_prob(sampled_as))

        _, rwd ,_, _ = env.step(sampled_as.numpy())


        av_collected_rwd *= discount



        av_collected_rwd[i,np.arange(n_envs)] = torch.from_numpy(rwd).float()     #-= (av_collected_rwd[i] - rwd) * ln_rate2



    graph = True

    updated_means = []



    cum_av_collected_rwd = torch.flip(torch.cumsum(torch.flip(av_collected_rwd,(0,)),0),(0,)) # works correctly




    #print(mean_vector[0].grad_fn.next_functions[0][0])

    for e in range(t_steps):

        loss = sum(log_p_actions[e] * cum_av_collected_rwd[e])


        if e == t_steps -1:
            graph= False


        loss.backward(retain_graph= graph)

        #print(ep,"  ",mean_vector.grad.data[e])


        updated_means.append((mean_vector[e] + ln_rate * mean_vector.grad.data[e])) # rwd is negative so, wanna maximise it?

        mean_vector.grad.data[e].zero_() # do I actually need it? because updated is done every time step, rather than computing sum of loss, so no risk of backtracking to previous t steps ?


    sum_rwd.append(np.sum(av_collected_rwd.numpy())/n_envs)

    if ep % 100 == 0:

        print(ep," ",sum(sum_rwd) / 100)
        sum_rwd = []




    mean_vector = torch.tensor(updated_means, requires_grad=True)

env.close()

# test run: NEED TO REWRITE THIS SO THAT IT STARTS FROM SAME POSITION
test_t_steps = 200

env_2 = gym.make("Pendulum-v0")
env_2.render()

for t in range(test_t_steps):

    env_2.reset()


    _,rwd,_,_ = env_2.step([mean_vector[t].detach().numpy()])
    print(rwd)

env_2.close()
















#env.close()