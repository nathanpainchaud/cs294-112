"""
Author of this script: Nathan Painchaud (nathan.painchaud@usherbrooke.ca)
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import Sequence


class ExpertPolicyDataSequence(Sequence):

    def __init__(self, observations, actions, batch_size):
        self.observations = observations
        self.actions = actions
        self.batch_size = batch_size

    def __getitem__(self, index):
        batch_input = self.observations[index * self.batch_size: (index + 1) * self.batch_size]
        batch_target = np.squeeze(self.actions[index * self.batch_size: (index + 1) * self.batch_size])
        return batch_input, batch_target

    def __len__(self):
        return int(np.ceil(len(self.observations) / self.batch_size))


class DaggerExpertPolicyDataSequence(ExpertPolicyDataSequence):

    def __init__(self, observations, actions, batch_size, env, network, policy_fn, eval_period=1, **policy_eval_opts):
        super().__init__(observations, actions, batch_size)
        self.env = env
        self.model = network.model
        self.model._make_predict_function()
        self.policy_fn = policy_fn
        self.policy_eval_opts = policy_eval_opts
        self.sess = tf.get_default_session()

        self.period = eval_period
        self.epochs_since_last_query = 0

    def on_epoch_end(self):
        self.epochs_since_last_query += 1
        if self.epochs_since_last_query >= self.period:
            self.epochs_since_last_query = 0

            # Query new observations and actions by running the policy in training
            new_obs, new_actions = self._query_data_from_current_policy(self.env, **self.policy_eval_opts)

            # Rebuild the data arrays to include the newly acquired data
            self.observations = np.vstack((self.observations, new_obs))
            self.actions = np.vstack((self.actions, new_actions))

        # The following shuffling is necessary because `steps_per_epoch` cannot be change dynamically during training.
        # To be able to train over new data generated at runtime, the arrays need to be shuffled so as to expose
        # new samples within the first `steps_per_epoch * batch_size` indices the array.
        rng_state = np.random.get_state()
        np.random.shuffle(self.observations)
        np.random.set_state(rng_state)
        np.random.shuffle(self.actions)

    def _query_data_from_current_policy(self, env, num_rollouts, max_timesteps):
        print('querying new expert data from current policy\'s distribution')
        max_steps = max_timesteps or self.env.spec.timestep_limit
        observations = []
        expert_actions = []
        with self.sess.as_default():
            for i in range(num_rollouts):
                obs = env.reset()
                done = False
                steps = 0
                while not done:
                    action = self.model.predict(obs[None, :])
                    observations.append(obs)
                    expert_actions.append(self.policy_fn(obs[None, :]))
                    obs, _, done, _ = env.step(action)
                    steps += 1
                    if steps >= max_steps:
                        break
        print('collected new expert data')
        return np.array(observations), np.array(expert_actions)
