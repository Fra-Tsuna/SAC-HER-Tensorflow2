#!/usr/bin/env python3


from HER import HER_Buffer, Experience
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import time
from models import ActorNetwork, CriticNetwork, ValueNetwork
from tensorflow_addons.optimizers import RectifiedAdam
from normalizer import Normalizer


# Learning parameters
MINIBATCH_SAMPLE_SIZE = 256
OPTIMIZATION_STEPS = 1
REWARD_SCALE = 20
REWARD_SCALE_FINAL = 50
RS_INCREASE_COEFF = 1.05
LEARNING_RATE = 1e-4
GAMMA = 0.99
TAU = 0.005
NORM_CLIP_RANGE = 5
CLIP_MAX = 200


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
        self.reward_scale = REWARD_SCALE

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

    def play_episode(self, criterion="random", epsilon=0, learn=False):                          
        """
        Play an episode choosing actions according to the selected criterion
        
        Parameters
        ----------
        criterion: strategy to choose actions ('random' or 'SAC')
        epsilon: random factor for epsilon-greedy strategy
        learn: if perform a gradient step optimization

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
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    obs_norm = self.state_norm.normalize(state['observation'])
                    goal_norm = self.goal_norm.normalize(state['desired_goal'])
                    obs_goal = \
                        np.concatenate([obs_norm, goal_norm])
                    obs_goal = np.array(obs_goal, ndmin=2)
                    action, _ = self.actor(obs_goal, noisy=False)
                    action = action.numpy()[0]
            else:
                print("ERROR: Wrong criterion for choosing the action")
            new_state, reward, done, _ = self.env.step(action)
            experiences.append(Experience(state, action, reward, new_state, done))
            state = new_state
            #print("\tStep: ", step, "Reward = ", reward)        
            if learn and len(self.her_buffer) > MINIBATCH_SAMPLE_SIZE:
                for opt_step in range(OPTIMIZATION_STEPS):
                    #print("__Optimization step ", opt_step, "__")
                    #minibatch = self.her_buffer.sample(MINIBATCH_SAMPLE_SIZE)
                    states, exp_actions, rewards, new_states, dones = \
                        self.her_buffer.sample(MINIBATCH_SAMPLE_SIZE)
                    v_loss, c1_loss, c2_loss, act_loss = \
                        self.optimization(states, exp_actions, rewards, new_states, dones)
                    self.soft_update()
        return experiences

    @tf.function
    def optimization(self, states, exp_actions, rewards, new_states, not_dones):
        """
        Update networks in order to learn the correct policy
        Parameters
        ----------
        minibatch: sample from the her buffer for the optimization
        Returns
        -------
        *_loss: loss of the correspondent network
        """
        # 1° step: preprocess input tensors
        states, new_states = self.preprocess_inputs(states, new_states)

        # 2° step: optimize value network
        actions, log_probs = self.actor(states, noisy=False)
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        q = tf.minimum(q1, q2)
        with tf.GradientTape() as value_tape:
            v = self.value(states)
            value_loss = 0.5 * tf.reduce_mean((v - (q-log_probs))**2)
        value_grads = value_tape.gradient(value_loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.value.trainable_variables))

        # 3° step: optimize critic networks
        v_tgt = self.target_value(new_states)
        q_tgt = [self.reward_scale*r for r in rewards] + GAMMA*(not_dones*v_tgt)
        with tf.GradientTape() as critic1_tape:
            q1 = self.critic_1(states, exp_actions)
            critic1_loss = 0.5 * tf.reduce_mean((q1 - q_tgt)**2)
        with tf.GradientTape() as critic2_tape:
            q2 = self.critic_2(states, exp_actions)
            critic2_loss = 0.5 * tf.reduce_mean((q2 - q_tgt)**2)
        variables_c1 = self.critic_1.trainable_variables
        variables_c2 = self.critic_2.trainable_variables
        critic1_grads = critic1_tape.gradient(critic1_loss, self.critic_1.trainable_variables)
        critic2_grads = critic2_tape.gradient(critic2_loss, self.critic_2.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic_1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic_2.trainable_variables))

        # 4° step: optimize actor network
        with tf.GradientTape() as actor_tape:
            actions, log_probs = self.actor(states, noisy=True)
            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            q = tf.minimum(q1, q2)
            actor_loss = tf.reduce_mean(log_probs - q)
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
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
        self.state_norm.update(np.clip(obs, -CLIP_MAX, CLIP_MAX))
        self.goal_norm.update(np.clip(g, -CLIP_MAX, CLIP_MAX))

    def preprocess_inputs(self, states_tl, new_states_tl):
        states, new_states, goals, new_goals = [], [], [], []
        for s in range(len(states_tl)):
            states.append(states_tl[s][0:-self.goal_size])
            goals.append(states_tl[s][-self.goal_size:])
            new_states.append(new_states_tl[s][0:-self.goal_size])
            new_goals.append(new_states_tl[s][-self.goal_size:])
        states = self.state_norm.normalize(tf.clip_by_value(states, -CLIP_MAX, CLIP_MAX))
        new_states = self.state_norm.normalize(tf.clip_by_value(new_states, -CLIP_MAX, CLIP_MAX))
        goals = self.goal_norm.normalize(tf.clip_by_value(goals, -CLIP_MAX, CLIP_MAX))
        new_goals = self.goal_norm.normalize(tf.clip_by_value(new_goals, -CLIP_MAX, CLIP_MAX))
        inputs = tf.concat([states, goals], axis=1)
        new_inputs = tf.concat([new_states, new_goals], axis=1)
        return inputs, new_inputs

    def temperature_decay(self):
        self.reward_scale = \
            min(REWARD_SCALE_FINAL, self.reward_scale * RS_INCREASE_COEFF)

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
