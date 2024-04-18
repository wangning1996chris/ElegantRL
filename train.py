from elegantrl.train.run import train_agent, train_agent_multiprocessing
from elegantrl.train.config import Config, build_env, get_gym_env_args
from elegantrl.agents import AgentDQN


agent_class = AgentDQN
env_name = "CartPole-v0"

import gym
gym.logger.set_level(40)  # Block warning
env = gym.make(env_name)
env_func = gym.make
env_args = get_gym_env_args(env, if_print=True)

args = Config(agent_class, env_func, env_args)

'''reward shaping'''
args.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
args.gamma = 0.97  # discount factor of future rewards

'''network update'''
args.target_step = args.max_step * 2  # collect target_step, then update network
args.net_dim = 2 ** 7  # the middle layer dimension of Fully Connected Network
args.num_layer = 3  # the layer number of MultiLayer Perceptron, `assert num_layer >= 2`
args.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
args.repeat_times = 2 ** 0  # repeatedly update network using ReplayBuffer to keep critic's loss small
args.explore_rate = 0.25  # epsilon-greedy for exploration.

'''evaluate'''
args.eval_gap = 2 ** 5  # number of times that get episode return
args.eval_times = 2 ** 3  # number of times that get episode return
args.break_step = int(8e4)  # break training if 'total_step > break_step'


args.learner_gpus = -1

train_agent(args)