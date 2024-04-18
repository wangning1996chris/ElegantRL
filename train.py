# Env humanoid
from elegantrl.train.run import train_agent, train_agent_multiprocessing
from elegantrl.train.config import Config, build_env, get_gym_env_args
from elegantrl.agents import AgentPPO
from elegantrl.envs.CustomGymEnv import PendulumEnv
import gym

gym.logger.set_level(40)  # Block warning
env = PendulumEnv()
env_func = PendulumEnv
agent_class = AgentPPO
env_args = get_gym_env_args(env, if_print=True)
args = Config(agent_class, env_func, env_args)

args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
args.gamma = 0.97

'''network update'''
args.target_step = args.max_step * 8
args.net_dim = 2 ** 7
args.num_layer = 2
args.batch_size = 2 ** 8
args.repeat_times = 2 ** 5

'''evaluate'''
args.eval_gap = 2 ** 6
args.eval_times = 2 ** 3
args.break_step = int(8e5)

args.learner_gpus = -1
train_agent(args)