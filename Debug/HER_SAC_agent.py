#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import RectifiedAdam

from normalizer import Normalizer
from HER import HER_Buffer, Experience
from models import ActorNetwork, CriticNetwork, ValueNetwork


# Learning parameters
REWARD_SCALE = 20                            
LEARNING_RATE = 0.001
GAMMA = 0.98
TAU = 0.005
NORM_CLIP_RANGE = 5
CLIP_MAX = 200

# Debug parameters
DEBUG_STATE = False
DEBUG_ACTION = False
DEBUG_LAST_EXP = False
DEBUG_FIRST_EXP = False
DEBUG_NORM_SAMPLE = False
DEBUG_REW_ACT_DONE = False
DEBUG_VALUE_OPTIM = False
DEBUG_CRITIC_OPTIM = False
DEBUG_ACTOR_OPTIM = False


class HER_SAC_Agent:

    def __init__(self, env, her_buffer, optimizer='Adam'):

        # env
        self.env = env
        self.her_buffer = her_buffer
        self.starting_state = self.env.reset()
        self.max_timesteps = self.env.spec.max_episode_steps
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
        input_tensor = tf.keras.Input(shape=(self.normal_state_shape), dtype=tf.float32)
        self.value(input_tensor)
        self.target_value(input_tensor)
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
            raise TypeError("Wrong or not supported optimizer. \
                            [availiable 'Adam' or 'Rectified_Adam']")

    def getBuffer(self):
        """
        return the replay buffer of the agent
        """
        return self.her_buffer

    def play_episode(self, criterion="random", epsilon=0):                          
        """
        Play an episode choosing actions according to the selected criterion
        
        Parameters
        ----------
        criterion: strategy to choose actions ('random' or 'SAC')
        epsilon: random factor for epsilon-greedy exploration strategy

        Returns
        -------
        experiences: all experiences taken by the agent in the episode
        """
        state = self.env.reset()
        experiences = []
        done = False
        t = 0
        while t < self.max_timesteps and not done:
            t += 1
            #self.env.render()
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
                        print("++++++++++++++++ DEBUG - STATE [AGENT.PLAY_EPISODE] ++++++++++++++++\n")
                        print("----------------------------state----------------------------")
                        print(state)
                        print("----------------------------obs_norm||goal----------------------------")
                        print(obs_goal)
                        a = input("\n\nPress Enter to continue...")
                    action, _ = self.actor(obs_goal, noisy=False)
                    action = action.numpy()[0]
            else:
                raise TypeError("Wrong criterion for choosing the action. \
                                [available 'random' or 'SAC']")                
            new_state, reward, done, info = self.env.step(action)
            done = done or bool(info['is_success'])
            experiences.append(Experience(state, action, reward, new_state, done))
            if DEBUG_ACTION:
                print("\n\n++++++++++++++++ DEBUG - TAKE ACTION [AGENT.PLAY_EPISODE] +++++++++++++++++\n")
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
        losses of all optimization processes
        """
        # 1° step: unzip minibatch sampled from HER
        exp_actions, rewards, dones = [], [], []
        for exp in minibatch:
            exp_actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.done)
        states, new_states = self.preprocess_inputs(minibatch)
        if DEBUG_NORM_SAMPLE:
            print("\n\n++++++++++++++++ DEBUG - HER NORM STATES/GOAL [AGENT.OPTIMIZATION] +++++++++++++++++\n")
            print("----------------------------element 0----------------------------")
            print(states[0][0:-self.goal_size])
            print(states[0][-self.goal_size:])
            print("----------------------------element -1----------------------------")
            print(states[-1][0:-self.goal_size])
            print(states[-1][-self.goal_size:])
            print("----------------------------random element----------------------------")
            elem = random.randint(0, len(states)-1)
            print(elem)
            print(states[elem][0:-self.goal_size])
            print(states[elem][-self.goal_size:])
            a = input("\n\nPress Enter to continue...") 

        if DEBUG_REW_ACT_DONE:
            print("\n\n++++++++++++++++ DEBUG - HER SAMPLE REWARDS ACTIONS DONES [AGENT.OPTIMIZATION] +++++++++++++++++\n")
            print("----------------------------rewards----------------------------")
            print(rewards)
            print("----------------------------actions----------------------------")
            print(exp_actions)
            print("----------------------------dones----------------------------")
            print(dones)
            a = input("\n\nPress Enter to continue...")

        # 2° step: optimize value network
        actions, log_probs = self.actor(states, noisy=False)
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        q = tf.minimum(q1, q2)
        if DEBUG_VALUE_OPTIM:
            print("\n\n++++++++++++++++ DEBUG - VALUE OPTIMIZATION [AGENT.OPTIMIZATION] +++++++++++++++++\n")
            print("----------------------------shapes----------------------------")
            print("actions = ", actions.shape)
            print("states = ", states.shape)
            print("q = ", q.shape)
            print("log_probs = ", log_probs.shape)
            a = input("\n\nPress Enter to continue...")
        with tf.GradientTape() as value_tape:
            v = self.value(states)
            value_loss = 0.5 * tf.reduce_mean(tf.square(v - (q-log_probs)))
        value_grads = value_tape.gradient(value_loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.value.trainable_variables))

        # 3° step: optimize critic networks
        v_tgt = tf.reshape(self.target_value(new_states), -1)
        q_tgt = [REWARD_SCALE*r for r in rewards] + GAMMA*([not d for d in dones]*v_tgt)
        if DEBUG_CRITIC_OPTIM:
            print("\n\n++++++++++++++++ DEBUG - CRITIC OPTIMIZATION [AGENT.OPTIMIZATION] +++++++++++++++++\n")
            print("----------------------------values----------------------------")
            print("v target = ", v_tgt)
            print("q target = ", q_tgt)
            print("----------------------------operations----------------------------")
            print("[REWARD_SCALE*r for r in rewards] = ", [REWARD_SCALE*r for r in rewards])
            print("GAMMA*([not d for d in dones]*v_tgt) = ", GAMMA*([not d for d in dones]*v_tgt))
            a = input("\n\nPress Enter to continue...")
        with tf.GradientTape() as critic1_tape:
            q1 = tf.reshape(self.critic_1(states, exp_actions), -1)
            if DEBUG_CRITIC_OPTIM:
                print("----------------------------values in tape----------------------------")
                print("q1 = ", q1)
                print("q_tgt = ", q_tgt)
                print("q1 - qtgt = ", (q1-q_tgt))
            critic1_loss = 0.5 * tf.reduce_mean(tf.square(q1 - q_tgt))
        with tf.GradientTape() as critic2_tape:
            q2 = tf.reshape(self.critic_2(states, exp_actions), -1)
            critic2_loss = 0.5 * tf.reduce_mean(tf.square(q2 - q_tgt))
        critic1_grads = critic1_tape.gradient(critic1_loss, self.critic_1.trainable_variables)
        critic2_grads = critic2_tape.gradient(critic2_loss, self.critic_2.trainable_variables)
        self.critic1_optimizer.apply_gradients(
            zip(critic1_grads, self.critic_1.trainable_variables))
        self.critic2_optimizer.apply_gradients(
            zip(critic2_grads, self.critic_2.trainable_variables))

        # 4° step: optimize actor network
        with tf.GradientTape() as actor_tape:
            actions, log_probs = self.actor(states, noisy=True)
            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            q = tf.minimum(q1, q2)
            actor_loss = tf.reduce_mean(log_probs - q)
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))
        
        return value_loss, critic1_loss, critic2_loss, actor_loss

    def soft_update(self, tau=TAU):
        """
        Target value soft update

        Parameters
        ----------
        tau: weight for the value parameters to do the soft update
        """
        for source, target in zip(self.value.variables, self.target_value.variables):
            target.assign((1.0 - tau) * target + tau * source)

    def update_normalizer(self, batch, hindsight=False):
        """
        Update normalizer parameters

        Parameters
        ----------
        batch: batch of experiences for the updating
        hindsight: True if the experiences are in the HER representation
        """
        if not hindsight:
            obs = [exp.state['observation'] for exp in batch]
            g = [exp.state['desired_goal'] for exp in batch]
        else:
            obs = [exp.state[0:-self.goal_size] for exp in batch]
            g = [exp.state[-self.goal_size:] for exp in batch] 
        self.state_norm.update(np.clip(obs, -CLIP_MAX, CLIP_MAX))
        self.goal_norm.update(np.clip(g, -CLIP_MAX, CLIP_MAX))

    def preprocess_inputs(self, her_batch):
        """
        Normalize states, goals and re-convert into HER representation
        
        Parameters
        ----------
        her_batch: batch of experiences expressed in the HER representation

        Returns
        -------
        input tensor for the networks
        """
        states, new_states, goals, new_goals = [], [], [], []
        for i in range(len(her_batch)):
            states.append(her_batch[i].state[0:-self.goal_size])
            new_states.append(her_batch[i].new_state[0:-self.goal_size])
            goals.append(her_batch[i].state[-self.goal_size:])
            new_goals.append(her_batch[i].new_state[-self.goal_size:])
        states = self.state_norm.normalize(np.clip(states, -CLIP_MAX, CLIP_MAX))
        new_states = self.state_norm.normalize(np.clip(new_states, -CLIP_MAX, CLIP_MAX))
        goals = self.goal_norm.normalize(np.clip(goals, -CLIP_MAX, CLIP_MAX))
        new_goals = self.goal_norm.normalize(np.clip(new_goals, -CLIP_MAX, CLIP_MAX))
        inputs = np.array(np.concatenate([states, goals], axis=1), ndmin=2)
        new_inputs = np.array(np.concatenate([new_states, new_goals], axis=1), ndmin=2)
        return inputs, new_inputs

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
