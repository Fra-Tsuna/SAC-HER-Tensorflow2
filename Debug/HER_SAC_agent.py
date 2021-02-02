#!/usr/bin/env python3


from HER import HER_Buffer, Experience
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
import random
from models import ActorNetwork, CriticNetwork, ValueNetwork
from tensorflow_addons.optimizers import RectifiedAdam
from normalizer import Normalizer


# Learning parameters
REWARD_SCALE = 20
LEARNING_RATE = 0.001
GAMMA = 0.98
TAU = 0.005
NORM_CLIP_RANGE = 5

# debug parameters
DEBUG_STATE = False
DEBUG_ACTION = False
DEBUG_LAST_EXP = False
DEBUG_FIRST_EXP = False
DEBUG_NORM_SAMPLE = False


class HER_SAC_Agent:

    def __init__(self, env, her_buffer, optimizer='Adam'):

        # env
        self.env = env
        self.her_buffer = her_buffer
        self.starting_state = self.env.reset()
        self.max_action = self.env.action_space.high[0]
        self.obs_size = self.env.observation_space['observation'].shape[0]
        self.goal_size = self.env.observation_space['desired_goal'].shape[0]
        self.state_size = self.obs_size + self.goal_size
        self.action_size = self.env.action_space.shape[0]

        # input shapes
        self.normal_state_shape = (self.state_size,)
        self.critic_state_shape = ((self.state_size + self.action_size),)

        # networks
        self.actor = ActorNetwork(self.normal_state_shape, self.action_size)
        self.critic_1 = CriticNetwork(self.critic_state_shape)
        self.critic_2 = CriticNetwork(self.critic_state_shape)
        self.value = ValueNetwork(self.normal_state_shape)
        self.target_value = ValueNetwork(self.normal_state_shape)

        # normalizers
        self.state_norm = Normalizer(size=self.obs_size, clip_range=NORM_CLIP_RANGE)
        self.goal_norm = Normalizer(size=self.goal_size, clip_range=NORM_CLIP_RANGE)

        # building value and target value
        input = tf.keras.Input(shape=(self.normal_state_shape), dtype=tf.float32)
        self.value(input)
        self.target_value(input)
        self.soft_update(tau = 1.0)

        # optimizers
        if optimizer == 'Adam':
            self.actor_optimizer = Adam(LEARNING_RATE)
            self.critic1_optimizer = Adam(LEARNING_RATE)
            self.critic2_optimizer = Adam(LEARNING_RATE)
            self.value_optimizer = Adam(LEARNING_RATE)
        elif optimizer == 'Rectified_Adam':
            self.actor_optimizer = RectifiedAdam(LEARNING_RATE)
            self.critic1_optimizer = RectifiedAdam(LEARNING_RATE)
            self.critic2_optimizer = RectifiedAdam(LEARNING_RATE)
            self.value_optimizer = RectifiedAdam(LEARNING_RATE)
        else:
            self.actor_optimizer = None
            self.critic1_optimizer = None
            self.critic2_optimizer = None
            print("Error: wrong or not supported optimizer")

    def getBuffer(self):
        return self.her_buffer

    def play_episode(self, criterion="random", epsilon=0):                          
        """
        Play an episode choosing actions according to the selected criterion
        
        Parameters
        ----------
        criterion: strategy to choose actions ('random' or 'SAC')
        epsilon: random factor for epsilon-greedy strategy

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
            if DEBUG_STATE:
                print("++++++++++++++++ DEBUG - STATE [AGENT.PLAY_EPISODE] ++++++++++++++++\n")
                time.sleep(1)
                print("----------------------------state----------------------------")
                print(state)
            if criterion == "random":
                action = self.env.action_space.sample()
            elif criterion == "SAC":
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    obs_norm = self.state_norm.normalize(state['observation'])
                    goal_norm = self.goal_norm.normalize(state['desired_goal'])
                    obs_goal = \
                        np.concatenate([obs_norm, goal_norm])
                    obs_goal = np.array(obs_goal, ndmin=2)
                    if DEBUG_STATE:
                        print("----------------------------obs_norm||goal----------------------------")
                        print(obs_goal)
                    action, _ = self.actor(obs_goal, noisy=False)
                    action = action.numpy()[0]
            else:
                print("ERROR: Wrong criterion for choosing the action")
            if DEBUG_STATE:
                a = input("\n\nPress Enter to continue...")
            new_state, reward, done, _ = self.env.step(action)
            experiences.append(Experience(state, action, reward, new_state, done))
            if DEBUG_ACTION:
                print("\n\n++++++++++++++++ DEBUG - TAKE ACTION [AGENT.PLAY_EPISODE] +++++++++++++++++\n")
                time.sleep(1)
                print("----------------------------action to take----------------------------")
                print(action)
                print("----------------------------new state----------------------------")
                print(new_state)
                print("----------------------------reward----------------------------")
                print(reward)
                print("----------------------------done----------------------------")
                print("----------------------------experience appended----------------------------")
                print(experiences[0])
                a = input("\n\nPress Enter to continue...")
            state = new_state
            #print("\tStep: ", step, "Reward = ", reward)        
        if DEBUG_LAST_EXP:
            print("\n\n++++++++++++++++ DEBUG - LAST EXPERIENCE [AGENT.PLAY_EPISODE] +++++++++++++++++\n")
            print(experiences[-1])
        if DEBUG_FIRST_EXP:
            print("\n\n++++++++++++++++ DEBUG - FIRST EXPERIENCE [AGENT.PLAY_EPISODE] +++++++++++++++++\n")
            print(experiences[0])
        return experiences

    def optimization(self, minibatch):
        """
        Update networks in order to learn the correct policy
        Parameters
        ----------
        minibatch: sample from the her buffer for the optimization
        Returns
        -------
        *_loss: loss of the correspondent network
        """
        # 1° step: unzip minibatch sampled from HER
        states, exp_actions, new_states, rewards, dones = [], [], [], [], []
        for exp in minibatch:
            states.append(exp.state)
            exp_actions.append(exp.action)
            new_states.append(exp.new_state)
            rewards.append(exp.reward)
            dones.append(exp.done)
        states = np.array(states, ndmin=2)
        new_states = np.array(new_states, ndmin=2)

        if DEBUG_NORM_SAMPLE:
            print("\n\n++++++++++++++++ DEBUG - HER NORM SAMPLE [AGENT.OPTIMIZATION] +++++++++++++++++\n")
            time.sleep(1)
            print("----------------------------element 0----------------------------")
            print(minibatch[0])
            print("----------------------------element -1----------------------------")
            print(minibatch[-1])
            print("----------------------------random element----------------------------")
            print(minibatch[random.randint(0, len(minibatch)-1)])
            a = input("\n\nPress Enter to continue...") 

        # 2° step: optimize value network
        with tf.GradientTape() as value_tape:
            actions, log_probs = self.actor(states, noisy=False)
            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            q = tf.minimum(q1, q2)
            v = self.value(states)
            value_loss = 0.5 * tf.reduce_mean((v - (q-log_probs))**2)
        variables = self.value.trainable_variables
        value_grads = value_tape.gradient(value_loss, variables)
        self.value_optimizer.apply_gradients(zip(value_grads, variables))

        # 3° step: optimize critic networks
        with tf.GradientTape() as critic1_tape:
            v_tgt = self.target_value(new_states)
            q_tgt = tf.stop_gradient([REWARD_SCALE*r for r in rewards] + 
                                     GAMMA*([not d for d in dones]*v_tgt))
            q1 = self.critic_1(states, exp_actions)
            critic1_loss = 0.5 * tf.reduce_mean((q1 - q_tgt)**2)
        with tf.GradientTape() as critic2_tape:
            v_tgt = self.target_value(new_states)
            q_tgt = tf.stop_gradient([REWARD_SCALE*r for r in rewards] + 
                                     GAMMA*([not d for d in dones]*v_tgt))
            q2 = self.critic_2(states, exp_actions)
            critic2_loss = 0.5 * tf.reduce_mean((q2 - q_tgt)**2)
        variables_c1 = self.critic_1.trainable_variables
        variables_c2 = self.critic_2.trainable_variables
        critic1_grads = critic1_tape.gradient(critic1_loss, variables_c1)
        critic2_grads = critic2_tape.gradient(critic2_loss, variables_c2)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, variables_c1))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, variables_c2))

        # 4° step: optimize actor network
        with tf.GradientTape() as actor_tape:
            actions, log_probs = self.actor(states, noisy=True)
            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            q = tf.minimum(q1, q2)
            actor_loss = tf.reduce_mean(log_probs - q)
        variables = self.actor.trainable_variables
        actor_grads = actor_tape.gradient(actor_loss, variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, variables))
        
        return value_loss, critic1_loss, critic2_loss, actor_loss

    def soft_update(self, tau=TAU):
        for source, target in zip(self.value.variables, self.target_value.variables):
            target.assign((1.0 - tau) * target + tau * source)

    def update_normalizer(self, batch, hindsight=False):
        if not hindsight:
            obs = [exp.state['observation'] for exp in batch]
            g = [exp.state['desired_goal'] for exp in batch]
        else:
            obs = [exp.state[0:-self.goal_size] for exp in batch]
            g = [exp.state[-self.goal_size:] for exp in batch]
        self.state_norm.update(np.array(obs))
        self.goal_norm.update(np.array(g))
        self.state_norm.recompute_stats()
        self.goal_norm.recompute_stats()

    def normalize_her_batch(self, her_batch):
        for i in range(len(her_batch)):
            her_batch[i].state[0:-self.goal_size] = \
                self.state_norm.normalize(her_batch[i].state[0:-self.goal_size])
            her_batch[i].state[-self.goal_size:] = \
                self.goal_norm.normalize(her_batch[i].state[-self.goal_size:])
            her_batch[i].new_state[0:-self.goal_size] = \
                self.state_norm.normalize(her_batch[i].new_state[0:-self.goal_size])
            her_batch[i].new_state[-self.goal_size:] = \
                self.goal_norm.normalize(her_batch[i].new_state[-self.goal_size:])
        return her_batch                                                                              # it is necessary??

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
