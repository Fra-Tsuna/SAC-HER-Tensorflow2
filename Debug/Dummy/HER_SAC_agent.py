#!/usr/bin/env python3

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import RectifiedAdam

from normalizer import Normalizer
from HER import HER_Buffer, Experience
from models import ActorNetwork, CriticNetwork


# Learning parameters
MINIBATCH_SAMPLE_SIZE = 256
LEARNING_RATE = 3e-4
LR_TEMPERATURE = 3e-4
GAMMA = 0.99
TAU = 0.005
NORM_CLIP_RANGE = 5
CLIP_MAX = 200

# Debug parameters
DEBUG_STATE = False
DEBUG_ACTION = False
DEBUG_POST_ACT = False
DEBUG_LAST_EXP = False
DEBUG_FIRST_EXP = False
DEBUG_NORM_SAMPLE = False
DEBUG_REW_ACT_DONE = False
DEBUG_CRITIC_OPTIM = False
DEBUG_ACTOR_OPTIM = False


class HER_SAC_Agent:

    def __init__(self, env, her_buffer, temperature="auto", optimizer='Adam', render=True):

        # env
        self.env = env
        self.render = render
        self.her_buffer = her_buffer
        self.starting_state = self.env.reset()
        self.max_timesteps = self.env.spec.max_episode_steps
        self.max_action = self.env.action_space.high[0]
        self.obs_size = self.env.observation_space['observation'].shape[0]
        self.goal_size = self.env.observation_space['desired_goal'].shape[0]
        self.state_size = self.obs_size + self.goal_size
        self.action_size = self.env.action_space.shape[0]

        # input shapes
        self.actor_state_shape = (self.state_size,)
        self.critic_state_shape = ((self.state_size + self.action_size),)

        # networks
        self.actor = ActorNetwork(self.actor_state_shape, self.action_size, self.max_action)
        self.critic_1 = CriticNetwork(self.critic_state_shape)
        self.critic_2 = CriticNetwork(self.critic_state_shape)
        self.target_critic1 = CriticNetwork(self.critic_state_shape)
        self.target_critic2 = CriticNetwork(self.critic_state_shape)

        # temperature parameters
        if temperature == "auto":
            self.log_temperature = tf.Variable(tf.math.log(1.0), dtype=tf.float32)
            self.target_entropy = -tf.constant(self.action_size, dtype=tf.float32)
        elif isinstance(temperature, float):
            if 0 < temperature <= 1:
                self.log_temperature = tf.math.log(temperature)
            else:
                raise ValueError("Temperature parameter must be in range ]0,1]")
        else:
            raise TypeError("Wrong temperature coefficient. \
                            [available 'auto' or float in ]0,1] range]")

        # building critics and target critics
        """
        input_c1_state = tf.keras.Input(shape=(self.critic_state_shape), dtype=tf.float32)
        input_c2_state = tf.keras.Input(shape=(self.critic_state_shape), dtype=tf.float32)
        input_tc1_state = tf.keras.Input(shape=(self.critic_state_shape), dtype=tf.float32)
        input_tc2_state = tf.keras.Input(shape=(self.critic_state_shape), dtype=tf.float32)
        input_action = tf.expand_dims(self.env.action_space.sample(), axis=0)
        self.critic_1(input_c1_state, input_action)
        self.critic_2(input_c2_state, input_action)
        self.target_critic1(input_tc1_state, input_action)
        self.target_critic2(input_tc2_state, input_action)
        self.soft_updates(tau = 1.0)
        """

        # optimizers
        if optimizer == 'Adam':
            self.actor_optimizer = Adam(LEARNING_RATE)
            self.critic1_optimizer = Adam(LEARNING_RATE)
            self.critic2_optimizer = Adam(LEARNING_RATE)
            if temperature == "auto":
                self.temperature_optimizer = Adam(LR_TEMPERATURE)
        elif optimizer == 'Rectified_Adam':
            self.actor_optimizer = RectifiedAdam(LEARNING_RATE)
            self.critic1_optimizer = RectifiedAdam(LEARNING_RATE)
            self.critic2_optimizer = RectifiedAdam(LEARNING_RATE)
            if temperature == "auto":
                self.temperature_optimizer = RectifiedAdam(LR_TEMPERATURE)
        else:
            raise TypeError("Wrong or not supported optimizer. \
                            [availiable 'Adam' or 'Rectified_Adam']")

        # normalizers
        self.state_norm = Normalizer(size=self.obs_size, clip_range=NORM_CLIP_RANGE)
        self.goal_norm = Normalizer(size=self.goal_size, clip_range=NORM_CLIP_RANGE)

        # debug switches and params
        self.debug_state = DEBUG_STATE
        self.debug_action = DEBUG_ACTION
        self.debug_post_act = DEBUG_POST_ACT
        self.debug_last_exp = DEBUG_LAST_EXP
        self.debug_first_exp = DEBUG_FIRST_EXP
        self.debug_norm_sample = DEBUG_NORM_SAMPLE
        self.debug_rew_act_done = DEBUG_REW_ACT_DONE
        self.debug_critic_optim = DEBUG_CRITIC_OPTIM
        self.debug_actor_optim = DEBUG_ACTOR_OPTIM
        self.iterations = 0

    def getBuffer(self):
        """
        return the replay buffer of the agent
        """
        return self.her_buffer

    def getTemperature(self):
        """
        return the entropy temperature coefficient
        """
        temperature_t = tf.exp(self.log_temperature)
        temperature = temperature_t.numpy()
        return temperature

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
            if self.render:
                self.env.render()
            if criterion == "random":
                action = self.env.action_space.sample()
            elif criterion == "SAC":
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    #obs_norm = self.state_norm.normalize(state['observation'])
                    #goal_norm = self.goal_norm.normalize(state['desired_goal'])
                    #obs_goal = \
                    #    np.concatenate([obs_norm, goal_norm])
                    obs_goal = np.concatenate([state['observation'], state['desired_goal']])
                    obs_goal = np.array(obs_goal, ndmin=2)
                    if self.debug_state:
                        print("++++++++++++++++ DEBUG - STATE [AGENT.PLAY_EPISODE] ++++++++++++++++\n")
                        print("----------------------------state----------------------------")
                        print(state)
                        print("----------------------------obs_norm||goal----------------------------")
                        print(obs_goal)
                        a = input("\n\nPress Enter to continue...")
                    action, _ = self.actor(obs_goal, noisy=True)
                    action = action.numpy()[0]
                    if self.debug_action:
                        print("\n\n++++++++++++++++ DEBUG - TAKE ACTION [AGENT.PLAY_EPISODE] +++++++++++++++++\n")
                        print("----------------------------action to take----------------------------")
                        print(action)
                        print("----------------------------log probs returned----------------------------")
                        print(_)
                        a = input("Press Enter to continue...")
            else:
                raise TypeError("Wrong criterion for choosing the action. \
                                [available 'random' or 'SAC']")                
            new_state, reward, done, info = self.env.step(action)
            self.iterations += 1
            experiences.append(Experience(state, action, reward, new_state, done))
            if self.debug_post_act:
                print("----------------------------new state----------------------------")
                print(new_state)
                print("----------------------------reward----------------------------")
                print(reward)
                print("----------------------------done----------------------------")
                print(done)
                print("----------------------------experience appended----------------------------")
                print(experiences[0])
                a = input("\n\nPress Enter to continue...")
            state = new_state
        if self.debug_last_exp:
            print("\n\n++++++++++++++++ DEBUG - LAST EXPERIENCE [AGENT.PLAY_EPISODE] +++++++++++++++++\n")
            print(experiences[-1])
        if self.debug_first_exp:
            print("\n\n++++++++++++++++ DEBUG - FIRST EXPERIENCE [AGENT.PLAY_EPISODE] +++++++++++++++++\n")
            print(experiences[0])
        return experiences

    def optimization(self, minibatch=None, ere_ck=None):
        """
        Update networks in order to learn the correct policy

        Parameters
        ----------
        minibatch: sample from the her buffer for the optimization
        ere_ck: parameter ck of ERE algorithm which control sampling range

        Returns
        -------
        losses of all optimization processes
        """
        # 1° step: unzip minibatch sampled from HER
        exp_actions, rewards, dones = [], [], []
        states, new_states = [], []
        if minibatch is None:
            minibatch = self.her_buffer.sample(minibatch_size=MINIBATCH_SAMPLE_SIZE, 
                                               ere_ck=ere_ck)
        for exp in minibatch:
            exp_actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.done)
            states.append(exp.state)
            new_states.append(exp.new_state)
        #states, new_states = self.preprocess_inputs(minibatch)
        states = np.array(states, ndmin=2)
        new_states = np.array(new_states, ndmin=2)

        del minibatch
        if self.debug_norm_sample:
            print("\n\n++++++++++++++++ DEBUG - HER NORM STATES/GOAL [AGENT.OPTIMIZATION] +++++++++++++++++\n")
            print("----------------------------state 0----------------------------")
            print(states[0][0:-self.goal_size])
            print(states[0][-self.goal_size:])
            print("----------------------------state -1----------------------------")
            print(states[-1][0:-self.goal_size])
            print(states[-1][-self.goal_size:])
            print("----------------------------new_state 0----------------------------")
            print(new_states[0][0:-self.goal_size])
            print(new_states[0][-self.goal_size:])
            print("----------------------------new_state -1----------------------------")
            print(new_states[-1][0:-self.goal_size])
            print(new_states[-1][-self.goal_size:])
            print("----------------------------state andom----------------------------")
            elem = random.randint(0, len(states)-1)
            print(elem)
            print(states[elem][0:-self.goal_size])
            print(states[elem][-self.goal_size:])
            print("----------------------------new_state random----------------------------")
            elem = random.randint(0, len(new_states)-1)
            print(elem)
            print(new_states[elem][0:-self.goal_size])
            print(new_states[elem][-self.goal_size:])
            a = input("\n\nPress Enter to continue...") 

        if self.debug_rew_act_done:
            print("\n\n++++++++++++++++ DEBUG - HER SAMPLE REWARDS ACTIONS DONES [AGENT.OPTIMIZATION] +++++++++++++++++\n")
            print("----------------------------rewards----------------------------")
            print(rewards)
            print("----------------------------actions----------------------------")
            print(exp_actions)
            print("----------------------------dones----------------------------")
            print(dones)
            a = input("\n\nPress Enter to continue...")

        # 2° step: optimize critic networks
        temperature = tf.exp(self.log_temperature)
        actions, log_probs = self.actor(new_states, noisy=True)
        if self.iterations >= 100000:
            print("Actions = ", actions)
            print("log_probs = ", log_probs)
            a = input("Press Enter to continue...")
        next_q1 = self.target_critic1(new_states, actions)
        next_q2 = self.target_critic2(new_states, actions)
        next_q = tf.minimum(next_q1, next_q2) - temperature * log_probs
        q_tgt = rewards + GAMMA*([not d for d in dones]*tf.reshape(next_q, -1))
        q_tgt = tf.reshape(q_tgt, (len(rewards), 1))
        if self.debug_critic_optim:
            print("\n\n++++++++++++++++ DEBUG - CRITIC OPTIMIZATION [AGENT.OPTIMIZATION] +++++++++++++++++\n")
            print("----------------------------values----------------------------")
            print("next_q1 = ", next_q1)
            print("next_q2 = ", next_q2)
            print("next_q = ", next_q)
            print("q target = ", q_tgt)
            a = input("\n\nPress Enter to continue...")
        with tf.GradientTape() as critic1_tape:
            q1 = self.critic_1(states, exp_actions)
            if self.debug_critic_optim:
                print("----------------------------values in tape----------------------------")
                print("q1 = ", q1)
                print("q_tgt = ", q_tgt)
                print("q1 - qtgt = ", (q1-q_tgt))
                a = input("\n\nPress Enter to continue...")
            critic1_loss = 0.5 * tf.reduce_mean(tf.square(q1 - q_tgt))
        with tf.GradientTape() as critic2_tape:
            q2 = self.critic_2(states, exp_actions)
            critic2_loss = 0.5 * tf.reduce_mean(tf.square(q2 - q_tgt))
        critic1_grads = critic1_tape.gradient(critic1_loss, self.critic_1.trainable_variables)
        critic2_grads = critic2_tape.gradient(critic2_loss, self.critic_2.trainable_variables)
        self.critic1_optimizer.apply_gradients(
            zip(critic1_grads, self.critic_1.trainable_variables))
        self.critic2_optimizer.apply_gradients(
            zip(critic2_grads, self.critic_2.trainable_variables))
 
        # 3° step: optimize actor network
        with tf.GradientTape() as actor_tape:
            actions, log_probs = self.actor(states, noisy=True)
            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            q = tf.minimum(q1, q2)
            if self.debug_actor_optim:
                print("----------------------------values in tape----------------------------")
                print("q = ", q)
                print("log_probs = ", log_probs)
                print("log_probs - q = ", (log_probs-q))
                print("temp * log_probs = ", temperature*log_probs)
                print("temp * log_probs - q = ", temperature*log_probs - q)
                print("Loss argument 0.5 * tf.reduce_mean(", (temperature*log_probs - q))
                a = input("\n\nPress Enter to continue...")
            actor_loss = tf.reduce_mean(temperature*log_probs - q)
            #actor_loss = tf.reduce_mean(log_probs - q)
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))

        # 4° step: optimize temperature parameter
        if isinstance(self.log_temperature, tf.Variable):
            actions, log_probs = self.actor(states, noisy=False)
            with tf.GradientTape() as temperature_tape: 
                temperature_loss = \
                    tf.reduce_mean(-tf.exp(self.log_temperature)*
                                  (log_probs + self.target_entropy))
            temperature_grads = \
                temperature_tape.gradient(temperature_loss, [self.log_temperature])
            self.temperature_optimizer.apply_gradients(
                zip(temperature_grads, [self.log_temperature]))
        else:
            temperature_loss = None

        return critic1_loss, critic2_loss, actor_loss, temperature_loss

    def soft_updates(self, tau=TAU):
        """
        Target value soft update

        Parameters
        ----------
        tau: weight for the value parameters to do the soft update
        """
        for source, target in zip(self.critic_1.variables, self.target_critic1.variables):
            target.assign((1.0 - tau) * target + tau * source)
        for source, target in zip(self.critic_2.variables, self.target_critic2.variables):
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
