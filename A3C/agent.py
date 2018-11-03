import tensorflow as tf
import numpy as np



# Creating our agent, and therefore our NN : 

class Agent():
    def __init__(self, session, action_size, model='mnih',
        optimizer=tf.train.AdamOptimizer(1e-4)):
# Need to switch to param.LR 
        self.action_size = action_size
        self.optimizer = optimizer
        self.sess = session

        with tf.variable_scope('network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.target_value = tf.placeholder('float32', [None], name='target_value')
            if model == 'mnih':
                self.state, self.policy, self.value = self.build_model(100,100, 4)
            else:
                # Assume we wanted a feedforward neural network
                self.state, self.policy, self.value = self.build_model_feedforward(13)
            self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
            scope='network')
            self.advantages = tf.placeholder('float32', [None], name='advantages')

        with tf.variable_scope('optimizer'):
            # Compute the one hot vectors for each action given.
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)

            min_policy = 1e-8
            max_policy = 1.0 - 1e-8
            self.log_policy = tf.log(tf.clip_by_value(self.policy,min_policy,max_policy))

            self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, action_one_hot), reduction_indices=1)

            self.policy_loss = -tf.reduce_mean(self.log_pi_for_action * self.advantages)

            self.value_loss = tf.reduce_mean(tf.square(self.target_value - self.value))

            self.entropy = tf.reduce_sum(tf.multiply(self.policy, -self.log_policy))

            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy* 0.01

            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_vars = list(zip(grads, self.weights))

            self.train_op = optimizer.apply_gradients(grads_vars)

    def get_policy(self, state):
        return self.sess.run(self.policy, {self.state: state}).flatten()

    def get_value(self, state):
        return self.sess.run(self.value, {self.state: state}).flatten()

    def get_policy_and_value(self, state):
        policy, value = self.sess.run([self.policy, self.value], {self.state:
        state})
        return policy.flatten(), value.flatten()

    
    def train(self, states, actions, target_values, advantages):
        # Training
        self.sess.run(self.train_op, feed_dict={
            self.state: states,
            self.action: actions,
            self.target_value: target_values,
            self.advantages: advantages
        })

 
    def build_model(self, h, w, channels):
        self.layers = {}
        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')
        self.layers['state'] = state
        
        with tf.variable_scope('conv1'):
            conv1 = tf.contrib.layers.convolution2d(inputs=state,
            num_outputs=16, kernel_size=[8,8], stride=[4,4], padding="VALID",
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.zeros_initializer())
            self.layers['conv1'] = conv1

        
        with tf.variable_scope('conv2'):
            conv2 = tf.contrib.layers.convolution2d(inputs=conv1, num_outputs=32,
            kernel_size=[4,4], stride=[2,2], padding="VALID",
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.zeros_initializer())
            self.layers['conv2'] = conv2

        
        with tf.variable_scope('flatten'):
            flatten = tf.contrib.layers.flatten(inputs=conv2)
            self.layers['flatten'] = flatten

        
        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=256,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['fc1'] = fc1

        
        with tf.variable_scope('policy'):
            policy = tf.contrib.layers.fully_connected(inputs=fc1,
            num_outputs=self.action_size, activation_fn=tf.nn.softmax,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=None)
            self.layers['policy'] = policy

        
        with tf.variable_scope('value'):
            value = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=None)
            self.layers['value'] = value

        return state, policy, value

# Model used for CSB : 
    def build_model_feedforward(self, input_dim, num_hidden=30):
        self.layers = {}
        state = tf.placeholder('float32', shape=(None, input_dim), name='state')

        self.layers['state'] = state
        
        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(inputs=state,
            num_outputs=num_hidden,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['fc1'] = fc1

        
        with tf.variable_scope('fc2'):
            fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
            num_outputs=num_hidden,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['fc2'] = fc2

        with tf.variable_scope('fc3'):
            fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
            num_outputs=num_hidden,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['fc3'] = fc3

        
        with tf.variable_scope('policy'):
            policy = tf.contrib.layers.fully_connected(inputs=fc3,
            num_outputs=self.action_size, activation_fn=tf.nn.softmax,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['policy'] = policy

        
        with tf.variable_scope('value'):
            value = tf.contrib.layers.fully_connected(inputs=fc3, num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['value'] = value

        return state, policy, value
