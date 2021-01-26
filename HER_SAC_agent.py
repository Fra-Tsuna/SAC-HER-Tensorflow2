#!/usr/bin/env python3


from HER import HER_Buffer, Experience
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from models import ActorNetwork, CriticNetwork, ValueNetwork


# Learning parameters
REWARD_SCALE = 5
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005


class HER_SAC_Agent:

    def __init__(self, env, her_buffer, optimizer='Adam'):

        # env
        self.env = env
        self.her_buffer = her_buffer
        self.env.reset()

        # input shape
        self.normal_state_shape = \
            ((env.observation_space['observation'].shape[0] +
              env.observation_space['desired_goal'].shape[0]),)
        self.critic_state_shape = \
            ((env.observation_space['observation'].shape[0] +
              env.observation_space['desired_goal'].shape[0] +
              env.action_space.shape[0]),)

        # networks
        self.actor = ActorNetwork(self.normal_state_shape, env.action_space.shape[0])
        self.critic_1 = CriticNetwork(self.critic_state_shape)
        self.critic_2 = CriticNetwork(self.critic_state_shape)
        self.value = ValueNetwork(self.normal_state_shape)
        self.target_value = ValueNetwork(self.normal_state_shape)

        # optimizers
        if optimizer == 'Adam':
            self.actor_optimizer = Adam(LEARNING_RATE)
            self.critic1_optimizer = Adam(LEARNING_RATE)
            self.critic2_optimizer = Adam(LEARNING_RATE)
            self.value_optimizer = Adam(LEARNING_RATE)
        else:
            self.actor_optimizer = None
            self.critic1_optimizer = None
            self.critic2_optimizer = None
            self.value_optimizer = None
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
            if criterion == "random":
                action = self.env.action_space.sample()
            elif criterion == "SAC":
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    state_goal = \
                        np.concatenate([state['observation'], state['desired_goal']])
                    state_goal = np.array(state_goal, ndmin=2)
                    action, _ = self.actor(state_goal, noisy=False)
                    action = action.numpy()[0]
            else:
                print("ERROR: Wrong criterion for choosing the action")
            new_state, reward, done, _ = self.env.step(action)
            experiences.append(Experience(state, action, reward, new_state, done))
            state = new_state
            print("\tStep: ", step, "Reward = ", reward)
        return experiences

    def optimization(self, minibatch):
        """
        Update networks in order to learn the correct policy
        Parameters
        ----------
        minibatch: sample from the her buffer for the optimization
        Returns
        -------
        
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

        # 2° step: optimize value network
        with tf.GradientTape() as value_tape:
            actions, log_probs = self.actor(states, noisy=False)
            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            q = tf.minimum(q1, q2)
            v = self.value(states)
            value_loss = 0.5 * tf.reduce_mean((v - (q-log_probs))**2)       # correct ?
        variables = self.value.trainable_variables
        value_grads = value_tape.gradient(value_loss, variables)
        self.value_optimizer.apply_gradients(zip(value_grads, variables))

        # 3° step: optimize critic networks
        with tf.GradientTape() as critic1_tape:
            v_tgt = self.value(new_states)
            q_tgt = REWARD_SCALE*rewards + GAMMA*v_tgt
            q1 = self.critic_1(states, exp_actions)
            critic1_loss = 0.5 * tf.reduce_mean((q1 - q_tgt)**2)
        with tf.GradientTape() as critic2_tape:
            v_tgt = self.value(new_states)
            q_tgt = REWARD_SCALE*rewards + GAMMA*v_tgt
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

        # 5° step - update target network
        dummy_build = self.target_value(states)
        target_value_variables = self.target_value.trainable_variables
        value_variables = self.value.trainable_variables
        for var in range(len(value_variables)):
            value_variables[var] = \
                TAU*value_variables[var] + (1-TAU)*target_value_variables[var]
        self.target_value.net.set_weights(value_variables)

        return value_loss, critic1_loss, critic2_loss, actor_loss

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
