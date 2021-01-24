from HER import HER_Buffer, Experience
import tensorflow as tf
import numpy as np

RANDOM_BATCH = 1000
HER_CAPACITY = 1000

class HER_SAC_Agent:

    def __init__(self, env, her_buffer):
        self.env = env
        self.her_buffer = her_buffer
        self.env.reset()

    def getBuffer(self):
        return self.her_buffer

    def store_experience(self, experience, goal):
        """
        Store an experience in the HER buffer
        Parameters
        ----------
        experience: experience to store
        goal: the goal to concatenate to the states
        """
        state = experience.state_goal
        action = experience.action
        reward = self.env.compute_reward(state['achieved_goal'], goal, None)
        new_state = experience.new_state_goal
        done = experience.done
        state_goal = np.concatenate([state['observation'], goal])
        new_state_goal = np.concatenate([new_state['observation'], goal])
        self.her_buffer.append(
            Experience(state_goal, action, reward, new_state_goal, done))

    def play_episode(self, criterion="random"):
        """
        Play an episode choosing actions according to the selected criterion
        
        Parameters
        ----------
        criterion: strategy to choose actions ('random' or 'SAC')
        Returns
        -------
        experiences: all experiences taken by the agent in the episode 
        """
        state = self.env.reset()
        experiences = []
        done = False
        step = 0
        while not done:
            step += 1
            self.env.render()
            if criterion == "random":
                action = self.env.action_space.sample()
            elif criterion == "SAC":
                # TO DO: action = ACTION SELECTED WITH SAC ...
                action = 0
            else:
                print("ERROR: Wrong criterion for choosing the action")
            new_state, reward, done, _ = self.env.step(action)
            experiences.append(Experience(state, action, reward, new_state, done))
            state = new_state
            print("\tStep: ", step, "Reward = ", reward)
        return experiences

    def train(self, batch_size):
        """
        Train the agent with a batch_size number of episodes
        """
        for episode in range(batch_size):
            experiences = \
                self.play_episode(criterion="SAC")
            self.store_experience(experiences[-1], experiences[0].state['desired_goal'])
            for exp in experiences:
                self.store_experience(exp, experience[-1].state_goal['achieved_goal'])
        
        # TO DO: Minibatch and optimization [...]

    def random_play(self, batch_size):
        """
        Play a batch_size number of episode with random policy
        Returns True if the agent reach the goal
        """
        for episode in range(batch_size):
            state = self.env.reset()
            print("Random episode ", episode)
            experiences = self.play_episode()
            if experiences[-1].reward > -1:
                return True
        return False
