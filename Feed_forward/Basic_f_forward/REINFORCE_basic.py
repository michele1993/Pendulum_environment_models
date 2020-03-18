import torch

from torch.distributions import Normal

import torch.optim as opt

class Agent():


    def __init__(self, ln_rate =0.01, std = 0.3, t_steps = 200):

        self.std = std
        self.ln_rate = ln_rate

        self.params = torch.randn(t_steps, requires_grad=True)

        self.optimiser = opt.Adam([self.params],ln_rate)



    def sample_a(self):


        d = Normal(self.params, self.std)

        sampled_acts = d.sample()

        self.log_p = d.log_prob(sampled_acts)

        return sampled_acts


    def update(self, cum_rwd):

        loss = sum(torch.mul(cum_rwd,- self.log_p))

        self.optimiser.zero_grad()

        loss.backward()

        self.optimiser.step()

        return loss

        #self.mean_vec.data.add_(self.ln_rate * self.mean_vec.grad.data)
        #self.mean_vec.grad.zero_()

