#!/usr/bin/env python3


# ___________________________________________ Libraries and Definitions ___________________________________________ #


# Learning libraries
import gym

# ___________________________________________________ Parameters ___________________________________________________ #


# environment parameters
ENV_NAME = "FetchPush-v1"

# learning parameters
RANDOM_BATCH = 1000

# ______________________________________________ Classes and Functions ______________________________________________ #


class SAC_Agent:
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def random_play(self, batch_size):
        for episode in range(batch_size):
            self.env.reset()
            print("Episode ", episode)
            step = 0
            while True:
                step += 1
                self.env.render()
                action = self.env.action_space.sample()
                new_state, reward, done, _ = self.env.step(action)
                print("\tStep: ", step, "Reward = ", reward)
                if reward > -1:
                    return True
                if done:
                    if episode == batch_size-1:
                        return False
                    break

# _____________________________________________________ Main _____________________________________________________ #


if __name__ == '__main__':

    # Environment initialization
    env = gym.make(ENV_NAME)
    obs_space = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    print(obs_space, n_actions)

    # Agent initialization
    agent = SAC_Agent(env)
    success = agent.random_play(RANDOM_BATCH)
    if success:
        print("Goal achieved! Good job!")
    else:
        print("Bad play...")
