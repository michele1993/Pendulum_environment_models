import gym
import torch
import torch.optim as op
import numpy as np

from Feed_forward.Inv_Pendulum_Wrapper import ModifiedInvPendulum

from torch.distributions import Normal


b_env = gym.make("Pendulum-v0")
env = ModifiedInvPendulum(b_env)

#b_env = gym.make("Pendulum-v0")
#env = ModifiedInvPendulum(b_env)


t_steps = 200
discount = 0.9
n_episodes = 10000
ln_rate = 0.0001
ln_rate2 = 0.1
N_var = 0.1

mean_vector = torch.randn(t_steps, requires_grad=True)


# print(env.unwrapped)
# print(env.env.env)
# print(env.env.env.__dict__)




sum_rwd=[]
best_rwd = []

cum_av_collected_rwd = np.zeros(t_steps)


for ep in range(1,n_episodes):


    env.reset()

    log_p_actions = []

    #av_collected_rwd = np.zeros((t_steps, n_envs))

    av_collected_rwd = np.zeros(t_steps)




    for i in range(t_steps):

        #env.render()

        d = Normal(mean_vector[i],N_var)

        #sampled_a = d.sample()

        sampled_as = d.sample()



        log_p_actions.append(d.log_prob(sampled_as))

        _, rwd ,_, _ = env.step([sampled_as.numpy()])


        av_collected_rwd *= discount



        # shall I use moving average but problem of varing time-steps?
        av_collected_rwd[i] = rwd

        sum_rwd.append(rwd)


    best_rwd.append(max(sum_rwd))




    graph = True

    updated_means = []


    av_collected_rwd = np.flip(np.cumsum(np.flip(av_collected_rwd))) # works correctly


    cum_av_collected_rwd = av_collected_rwd     #-= (cum_av_collected_rwd - av_collected_rwd) /ep #



    #print(mean_vector[0].grad_fn.next_functions[0][0])




    for e in range(t_steps):

        loss = log_p_actions[e] * cum_av_collected_rwd[e]


        if e == t_steps -1:
            graph= False


        loss.backward(retain_graph= graph)




        mean_vector[e].data.add_(ln_rate * mean_vector.grad[e].data)



    mean_vector.grad.zero_() # do I actually need it? because updated is done every time step, rather than computing sum of loss, so no risk of backtracking to previous t steps ?




    #print(mean_vector)





    if ep % 100 == 0:

        print(ep," ",np.sum(sum_rwd) / (100 * t_steps))
        print("best rwd", sum(best_rwd) / 100)
        sum_rwd = []
        best_rwd = []


    #print(updated_means)
    #mean_vector = torch.tensor(updated_means, requires_grad=True)

    #print(mean_vector)

    #print(mean_vector[0].grad_fn.next_functions)



# test run: NEED TO REWRITE THIS SO THAT IT STARTS FROM SAME POSITION

env.render()
test_t_steps= t_steps

for t in range(test_t_steps):

    env.reset()


    _,rwd,_,_ = env.step([mean_vector[t].detach().numpy()])
    print(rwd)

env.close()



# updated_means.append((mean_vector[e] + ln_rate * mean_vector.grad[e])) # rwd is negative so, wanna maximise it?