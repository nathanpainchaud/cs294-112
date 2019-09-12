"""
Author of this script: Nathan Painchaud (nathan.painchaud@usherbrooke.ca)
"""
import os
import pickle

import gym
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam

from hw1.dataset import ExpertPolicyDataSequence, DaggerExpertPolicyDataSequence
from hw1.fully_connected_nn import FullyConnectedNN, root_mean_squared_error
from hw1.utils import tf_util
from hw1.utils.callback import get_callbacks
from hw1.utils.load_policy import load_policy

POLICY_DIR = 'experts'
DATA_DIR = 'expert_data'
EXPERIMENTS_DIR = 'experiments'


def run_algorithm(experiment_opts, model_opts, training_opts, model_eval_opts, policy_eval_opts):
    envname, algorithm, name = experiment_opts['envname'], experiment_opts['algorithm'], experiment_opts['name']

    # Fetch the expert policy's rollout data and build sequences to provide training data
    print('loading expert data')
    with open(os.path.join(DATA_DIR, envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)
    observations_train, observations_valid, actions_train, actions_valid \
        = train_test_split(expert_data['observations'], expert_data['actions'], train_size=0.8)
    print('loaded expert data')

    # Build a model and training tools (callbacks, environment, etc.)
    env = gym.make(envname)
    network = FullyConnectedNN(input_dims=observations_train.shape[-1], output_dims=actions_train.shape[-1],
                               **model_opts, optimizer=Adam(lr=training_opts['lr']), loss=root_mean_squared_error)

    callbacks = get_callbacks(env, network, experiment_opts, model_eval_opts, policy_eval_opts)

    with tf.Session() as sess:
        K.set_session(sess)

        # Build sequences to provide training data based on the chosen algorithm
        if algorithm == 'behavioral_cloning':
            train_seq = ExpertPolicyDataSequence(observations_train, actions_train, training_opts['batch_size'])
            valid_seq = ExpertPolicyDataSequence(observations_valid, actions_valid, training_opts['batch_size'])
        else:  # algorithm == 'dagger'
            print('loading and building expert policy')
            policy_fn = load_policy(os.path.join(POLICY_DIR, envname + '.pkl'))
            tf_util.initialize()
            print('loaded and built')

            train_seq = DaggerExpertPolicyDataSequence(observations_train, actions_train, training_opts['batch_size'],
                                                       env, network, policy_fn, model_eval_opts['eval_period'],
                                                       **policy_eval_opts)
            valid_seq = DaggerExpertPolicyDataSequence(observations_valid, actions_valid, training_opts['batch_size'],
                                                       env, network, policy_fn, model_eval_opts['eval_period'],
                                                       **policy_eval_opts)

        # Train the model to imitate the expert policy
        network.train(train_seq=train_seq, valid_seq=valid_seq, epochs=training_opts['epochs'], callbacks=callbacks)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    experiment_group = parser.add_argument_group(title="Experiment options")
    experiment_group.add_argument('--envname', type=str, required=True,
                                  choices=['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Reacher-v2',
                                           'Walker2d-v2'])
    experiment_group.add_argument('--algorithm', type=str, required=True,
                                  choices=['behavioral_cloning', 'dagger'],
                                  help="Algorithm to use to train the agent"
                                       "Results are exported to a subfolder named as the algorithm inside the target "
                                       "directory")
    experiment_group.add_argument('--name', type=str,
                                  help='Subfolder in `./experiments/` in which to save the results '
                                       'Defaults to `{envname}` if not specified')

    training_group = parser.add_argument_group(title="Training options")
    training_group.add_argument('--epochs', type=int, default=200,
                                help="Number of training epochs/DAgger iterations (depending on algorithm used)")
    training_group.add_argument('--batch_size', type=int, default=512)
    training_group.add_argument('--lr', type=float, default=1e-3)

    policy_eval_group = parser.add_argument_group(title="Policy evaluation options")
    policy_eval_group.add_argument('--max_timesteps', type=int)
    policy_eval_group.add_argument('--num_rollouts', type=int, default=20, help='Number of roll outs for evaluation')

    model_eval_group = parser.add_argument_group(title="Model evaluation options")
    model_eval_group.add_argument('--eval_period', type=int, default=10,
                                  help='Number of periods between evaluation runs/queries for new data (DAgger)')
    model_eval_group.add_argument('--plot_expert', action='store_true')

    model_group = parser.add_argument_group(title="Model options")
    model_group.add_argument('--hidden_dims', type=int, default=100)
    model_group.add_argument('--hidden_depth', type=int, default=4)
    model_group.add_argument('--l2_reg', type=float, default=1e-6)
    model_group.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    name = args.name if args.name else args.envname
    experiment_opts = {'envname': args.envname,
                       'algorithm': args.algorithm,
                       'name': os.path.join(EXPERIMENTS_DIR, name, args.algorithm)}
    training_opts = {'epochs': args.epochs,
                     'batch_size': args.batch_size,
                     'lr': args.lr}
    policy_eval_opts = {'max_timesteps': args.max_timesteps,
                        'num_rollouts': args.num_rollouts}
    model_eval_opts = {'eval_period': args.eval_period,
                       'plot_expert': args.plot_expert}
    model_opts = {'hidden_dims': args.hidden_dims,
                  'hidden_depth': args.hidden_depth,
                  'l2_reg': args.l2_reg,
                  'dropout': args.dropout}

    run_algorithm(experiment_opts, model_opts, training_opts, model_eval_opts, policy_eval_opts)


if __name__ == '__main__':
    main()
