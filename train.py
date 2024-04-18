# Env humanoid
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

'''network train'''
args.max_step = 1000

'''reward shaping'''
args.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
args.gamma = 0.97  # discount factor of future rewards

'''network update'''
args.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
args.repeat_times = 2 ** 0  # repeatedly update network using ReplayBuffer to keep critic's loss small

'''evaluate'''
args.eval_times = 2 ** 3  # number of times that get episode return
args.break_step = int(8e4)  # break training if 'total_step > break_step'

'''gpu cpu'''
args.learner_gpus = -1

train_agent(args)