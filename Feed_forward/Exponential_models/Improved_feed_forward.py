import gym
from Feed_forward.Inv_Pendulum_Wrapper import ModifiedInvPendulum
from Feed_forward.Exponential_models.Exponential_function import ExpF_agent
import torch

n_exp_fs = 6
n_t_steps = 100
n_eps = 30000
discount = 0.99
t_print = 100


b_env = gym.make("Pendulum-v0")
env = ModifiedInvPendulum(b_env)

exp_f = ExpF_agent(n_exps= n_exp_fs,t_steps = n_t_steps)

sum_rwd=[]
best_rwd = []
cum_av_collected_rwd = torch.zeros(n_t_steps)
loss = 0

for ep in range(n_eps):

    env.reset()
    sampled_as = exp_f.sample_actions()
    rwds = torch.zeros(n_t_steps)
    undis_rwds = torch.empty(n_t_steps)


    for t in range(n_t_steps):

        _, rwd, done, _ = env.step([sampled_as[t].numpy()])

        rwds *= discount

        rwds[t] = torch.from_numpy(rwd)
        undis_rwds[t] = torch.from_numpy(rwd)




    sum_rwd.append(sum(undis_rwds))
    best_rwd.append(torch.max(undis_rwds))
    cum_av_collected_rwd = torch.flip(torch.cumsum(torch.flip(rwds, (0,)), 0), (0,))

    loss += exp_f.update(cum_av_collected_rwd).detach() #


    if ep % t_print == 0:


        print("loss: ", loss / t_print)
        print(ep,"av rwd", sum(sum_rwd) /t_print)
        print("best rwd ", sum(best_rwd) / t_print)
        #print("av_best_indx ", sum(idx_best_rwd)/200, "\n")
        #print("miu: ",exp_f.miu)
        #print("decay: ", exp_f.decay_rate)
        sum_rwd = []
        best_rwd = []
        loss = 0



    # if ep % 500 ==0:
    #     print("actions", sampled_as)











