import os
from types import SimpleNamespace

import torch
from torch import multiprocessing as _mp
from a2c_ppo_acktr.arguments import get_args


from main import main


def run_mujoco(value, mp):
    args = get_args()
    args.env_name = "CartPole-v1"
    args.base = "MLPBase"
    main(args, value, mp=mp)


if __name__ == "__main__":
    torch.manual_seed(0)

    mp = _mp.get_context("spawn")
    _Value = mp.Value

    os.environ["OMP_NUM_THREADS"] = "1"
    run_mujoco(_Value, mp=mp)
