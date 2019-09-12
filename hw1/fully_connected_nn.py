"""
Author of this script: Nathan Painchaud (nathan.painchaud@usherbrooke.ca)
"""
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import elu
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.regularizers import l2
from tqdm import tqdm


class FullyConnectedNN(object):

    def __init__(self, input_dims, hidden_dims, hidden_depth, dropout, l2_reg, output_dims, optimizer, loss):
        input = Input(shape=(input_dims,))
        x = input
        for depth in range(hidden_depth):
            x = Dense(hidden_dims, kernel_regularizer=l2(l2_reg), activation=elu)(x)
            if dropout:
                x = Dropout(rate=dropout)(x)
        output = Dense(output_dims)(x)
        self.model = tf.keras.Model(inputs=input, outputs=output)
        self.model.compile(optimizer, loss=loss)

    def train(self, train_seq, valid_seq, **kwargs):
        try:
            print(("Press ctrl-c to stop the training and continue "
                   "the pipeline."))
            # Fit the model
            print("Fitting model...")
            self.model.fit_generator(train_seq, validation_data=valid_seq, **kwargs)
        except KeyboardInterrupt as ki:
            print("\nTraining cancelled!\n")

    def evaluate(self, env, num_rollouts, max_timesteps):
        max_steps = max_timesteps or env.spec.timestep_limit
        returns = []
        for _ in tqdm(range(num_rollouts), desc='Evaluating policy returns', unit='rollout'):
            obs = env.reset()
            done = False
            total = steps = 0
            while not done:
                action = self.model.predict(obs[None, :])[0]
                obs, r, done, _ = env.step(action)
                total += r
                steps += 1
                if steps >= max_steps:
                    break
            returns.append(total)
        return returns


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))
