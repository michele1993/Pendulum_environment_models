import gym
import torch

from Feed_forward.Inv_Pendulum_Wrapper import ModifiedInvPendulum

from Feed_forward.Basic_f_forward.REINFORCE_basic import Agent


b_env = gym.make("Pendulum-v0")
env = ModifiedInvPendulum(b_env)





t_steps = 200
discount = 0.99
n_episodes = 30000
ln_rate2 = 0.1

t_print = 200

loss = 0



reinforce = Agent()


sum_rwd=[]

best_rwd = []

#idx_best_rwd = []

cum_av_collected_rwd = torch.zeros(t_steps) # DON'T NEED THIS


for ep in range(1,n_episodes):


    env.reset()

    av_collected_rwd = torch.zeros(t_steps)

    sampled_as = reinforce.sample_a()

    undiscounted_rwd = torch.zeros(t_steps)



    for i in range(t_steps):

        _, rwd ,_, _ = env.step([sampled_as[i].numpy()])

        av_collected_rwd *= discount

        # shall I use moving average but problem of varing time-steps?
        av_collected_rwd[i] = rwd

        undiscounted_rwd[i] = rwd



    sum_rwd.append(sum(undiscounted_rwd))

    best_rwd.append(torch.max(undiscounted_rwd)) # fix this because can't have sum of tuples
    #idx_best_rwd.append(torch.argmax(undiscounted_rwd))


    cum_av_collected_rwd = torch.flip(torch.cumsum(torch.flip(av_collected_rwd, (0,)), 0), (0,)) # works correctly

    loss += reinforce.update(cum_av_collected_rwd).detach()# Don't need the .detach()



    if ep % t_print == 0:


        print("loss: ", loss / t_print)
        print(ep," ", sum(sum_rwd) /t_print)
        print("best rwd ", sum(best_rwd) / t_print)
        #print("av_best_indx ", sum(idx_best_rwd)/200, "\n")
        sum_rwd = []
        best_rwd = []
        loss = 0
        #idx_best_rwd = []


# test
env = gym.wrappers.Monitor(env,"record")
#env.render()
test_t_steps= t_steps

env.reset()
for t in range(test_t_steps):

    _,rwd,_,_ = env.step([reinforce.params[t].detach().numpy()])
    print(rwd)

