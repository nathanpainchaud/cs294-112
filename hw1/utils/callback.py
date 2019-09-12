"""
Author of this script: Nathan Painchaud (nathan.painchaud@usherbrooke.ca)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')  # Necessary to ensure no "Could not connect to display" error when running on remote server
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.callbacks import Callback, TensorBoard, ModelCheckpoint


class ReturnsHistory(Callback):
    def __init__(self, env, network, experiment_opts, model_eval_opts, **policy_eval_opts):
        super().__init__()
        self.env = env
        self.network = network
        self.envname = experiment_opts['envname']
        self.algorithm = experiment_opts['algorithm']
        self.name = experiment_opts['name']

        self.returns = []
        self.policy_eval_opts = policy_eval_opts
        self.best_return, self.best_return_std = None, None

        self.period = model_eval_opts['eval_period']
        self.epochs_since_last_save = 0

        self.plot_expert = model_eval_opts['plot_expert']

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            epoch_returns = self.network.evaluate(self.env, **self.policy_eval_opts)
            self.returns.append(epoch_returns)

            if self.best_return is None or np.mean(epoch_returns) > self.best_return:
                self.best_return, self.best_return_std = np.mean(epoch_returns), np.std(epoch_returns)

    def on_train_end(self, logs=None):
        """
        Inspired by a public repository of solutions to the assignments that automatically generates a report.
        Available here: https://github.com/xuwd11/cs294-112_hws
        """
        experiment_name = os.path.basename(os.path.dirname(self.name))
        json_filepath = os.path.join(self.name, experiment_name + '_results.json')
        print("exporting returns history to '{}'".format(json_filepath))
        evaluated_epochs = list(range(self.period, self.params['epochs'] + 1, self.period))
        with open(json_filepath, 'w') as f:
            json.dump({'epochs': evaluated_epochs,
                       'returns': self.returns,
                       'best_return': self.best_return,
                       'best_return_std': self.best_return_std},
                      f)
        color = {'behavioral_cloning': 'b', 'dagger': 'r'}
        plt.plot(evaluated_epochs, np.mean(self.returns, axis=-1), color=color[self.algorithm], label=self.algorithm)
        plt.errorbar(evaluated_epochs, np.mean(self.returns, axis=-1), np.std(self.returns, axis=-1), fmt='.',
                     color=color[self.algorithm])
        if self.plot_expert is not None:
            from hw1.main import DATA_DIR
            with open(os.path.join(DATA_DIR, self.envname + '.json'), 'r') as f:
                expert_returns = json.load(f)
            plt.fill_between(evaluated_epochs, expert_returns['mean_return'] - expert_returns['std_return'],
                             expert_returns['mean_return'] + expert_returns['std_return'], label='expert', color='g')
        plt.xlabel('epoch')
        plt.ylabel('return')
        plt.legend(loc='best')
        plt.title(self.envname)
        plt.tight_layout()
        png_filepath = os.path.join(self.name, experiment_name + '_' + self.algorithm + '.png')
        print("exporting returns graph to '{}'".format(png_filepath))
        plt.savefig(png_filepath, bbox_inches='tight', transparent=True, pad_inches=0.1)


def get_callbacks(env, network, experiment_opts, model_eval_opts, policy_eval_opts):
    # Create callback for TensorBoard
    tensorboard = TensorBoard(log_dir=experiment_opts['name'], histogram_freq=0, write_graph=True,
                              write_images=False)

    # Create model saving callback
    details = os.path.basename(experiment_opts['name']) + '.h5'
    model_saver = ModelCheckpoint(os.path.join(experiment_opts['name'], details), monitor='val_loss',
                                  save_best_only=True)

    # Create returnsn history callback
    returns_history = ReturnsHistory(env, network, experiment_opts, model_eval_opts, **policy_eval_opts)

    return [tensorboard, model_saver, returns_history]
