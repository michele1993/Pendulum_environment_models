import gym
import numpy as np

class Parallel_Mod_Pendulum(gym.Wrapper):

    def __init__(self,make_env,num_envs=1):

        super(Parallel_Mod_Pendulum,self).__init__(make_env())

        self.num_envs = num_envs
        self.envs = [make_env() for env_idx in range(num_envs)]



    def reset(self):


        states = []
        for env in self.envs:

            state = env.reset()
            states.append((np.arccos(state[0]), state[2]))


        return np.asarray(states)




        #Reset always
        # for env in self.envs:
        #
        #     env.reset()
        #     env.unwrapped.state = [np.pi,0]
        #
        # return np.asanyarray(np.tile([np.pi,0],self.num_envs))




    def step(self,actions):

        next_states, rewards, dones, infos = [],[],[],[]

        actions = actions.reshape(self.num_envs)


        for env, action in zip(self.envs, actions):

            next_state,reward,done,info = env.step([action])
            next_states.append([next_state[0], next_state[1]])
            rewards.append(reward)
            dones.append(done)
            infos.append(info)


        return np.asarray(next_states), np.asarray(rewards), np.asarray(dones), np.asarray(infos)


    def reset_at(self,indx):

        self.envs[indx].reset()

        self.envs[indx].unwrapped.state = [np.pi,0]

        return self.envs[indx].unwrapped # easier to change to np.asanyarray(np.tile([np.pi,0],self.num_envs))

