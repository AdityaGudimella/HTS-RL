import copy
import ctypes
import gfootball.env as football_env
import gym
import torch
import torch.multiprocessing as _mp

from a2c_ppo_acktr.base_factory import get_base
from a2c_ppo_acktr.model import MLPBase, Policy
from create_env import create_atari_mjc_env
from gym.spaces.discrete import Discrete

mp = _mp.get_context("spawn")
Value = mp.Value


def init_shared_var(
    action_space, observation_space, aug_obs_dim, num_processes, num_agents, num_actors
):
    manager = mp.Manager()
    shared_list = manager.list([False] * num_processes)
    done_list = manager.list([False] * num_processes)
    actions = torch.zeros(num_processes, num_agents, 1).long()
    action_log_probs = torch.zeros(num_processes, num_agents, 1)
    action_logits = torch.zeros(num_processes, num_agents, action_space.n)
    values = torch.zeros(num_processes, num_agents, 1)
    observations = torch.zeros(num_processes, *observation_space.shape)
    aug_observations = torch.zeros(num_processes, num_agents, aug_obs_dim)
    actions.share_memory_(), action_log_probs.share_memory_(), values.share_memory_(), observations.share_memory_()
    aug_observations.share_memory_(), action_logits.share_memory_()
    step_dones = mp.Array(ctypes.c_int32, int(num_processes))
    act_in_progs = mp.Array(ctypes.c_int32, int(num_processes))
    model_updates = mp.Array(ctypes.c_int32, int(num_actors))
    please_load_model = Value("i", 0)
    please_load_model_actor = torch.zeros(int(num_actors)).long()
    all_episode_scores = manager.list()
    return (
        shared_list,
        done_list,
        actions,
        action_log_probs,
        action_logits,
        values,
        observations,
        aug_observations,
        step_dones,
        act_in_progs,
        model_updates,
        please_load_model,
        please_load_model_actor,
        all_episode_scores,
    )


def init_policies(observation_space, action_space, base_kwargs, num_agents, base):
    actor_critics = [
        Policy(
            observation_space.shape[1:],
            action_space if num_agents == 1 else Discrete(action_space.nvec[0]),
            base=MLPBase,
            base_kwargs=base_kwargs,
        )
        for _ in range(num_agents)
    ]
    shared_cpu_actor_critics = [
        Policy(
            observation_space.shape[1:],
            action_space if num_agents == 1 else Discrete(action_space.nvec[0]),
            base=MLPBase,
            base_kwargs=base_kwargs,
        ).share_memory()
        for _ in range(num_agents)
    ]
    shared_cpu_actor_critics_env_actor = [
        Policy(
            observation_space.shape[1:],
            action_space if num_agents == 1 else Discrete(action_space.nvec[0]),
            base=MLPBase,
            base_kwargs=base_kwargs,
        ).share_memory()
        for _ in range(num_agents)
    ]
    pytorch_total_params = sum(
        p.numel() for p in actor_critics[0].parameters() if p.requires_grad
    )
    print("number of params ", pytorch_total_params)
    return actor_critics, shared_cpu_actor_critics, shared_cpu_actor_critics_env_actor


def get_policy_arg(hidden_size):
    base_kwargs = {"recurrent": False, "hidden_size": hidden_size}
    aug_obs_dim = 0
    return base_kwargs, aug_obs_dim


def get_env_info(env_name, num_agents):
    if env_name in ("CartPole-v1", "MountainCar-v0", "Pendulum-v0"):
        env = gym.make(env_name)
    else:
        env = create_atari_mjc_env(env_name)
    if num_agents == 1:
        from a2c_ppo_acktr.envs import ObsUnsqueezeWrapper

        env = ObsUnsqueezeWrapper(env)
    result = copy.deepcopy(env.observation_space), copy.deepcopy(env.action_space)
    env.close()
    return result
