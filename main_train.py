from __future__ import annotations
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from env.env import Env
from agents.tutor import SARSATutor, RandomPolicyTutor, FixedPolicyTutor
from trainer import Trainer, EpisodeResult
import utils.data_util as data_util
import utils.plotter as plot_util

import argparse

def train(
    *,
    n_rules: int = 3,
    episodes: int = 50,
    seed: int | None = 42,
    tutormode:str = 'sarsa',
    save_full_result = False
) -> List[EpisodeResult]:
    """Run a basic training loop and return per-episode results.

    Args:
        n_rules: Number of rules/items in the environment.
        episodes: Number of training episodes to run.
        seed: Random seed forwarded to the env (student model).
        use_random: If True, use RandomPolicyTutor instead of SARSA.
    """
    env = Env(N=n_rules, seed=seed)

    if tutormode == 'fixed':
        tutor = FixedPolicyTutor(
            env
        )
    elif tutormode=='random':
        tutor = RandomPolicyTutor(
            env,
            initial_epsilon=0.0,  # not used; random policy ignores it
        )
    elif tutormode=='sarsa':
        i_eps = 1.0
        f_eps = 0.1
        d_eps = (i_eps-f_eps)/(episodes/3)
        tutor = SARSATutor(
            env,
            learning_rate=0.2,
            initial_epsilon=i_eps,
            epsilon_decay=d_eps,
            final_epsilon=f_eps,
            discount_factor=0.99,
            seed=seed,
        )
    else:
        raise ValueError(f'tutormode \'{tutormode}\' does not exist')

    if save_full_result:
        ep_begin_cb = data_util.append_new_EpisodeResult
        step_cb = data_util.append_stepResult
    else:
        ep_begin_cb = None
        step_cb = None

    trainer = Trainer(env, tutor)
    results = trainer.train(
        n_episodes=episodes,
        max_steps=env.max_steps,
        decay_epsilon_each_episode=True,
        on_episode_begin=ep_begin_cb,
        on_step=step_cb,
    )

    if save_full_result:
        data_util.save_n_reset(dscr=tutormode)

    return results

def results_to_rewards_MA(results:List[EpisodeResult], window=0.05):
    episodes = len(results)
    w = int(episodes*window)
    tr = [r.total_reward for r in results]
    ma_tr = [sum(tr[i:i+w])/w for i in range(len(tr)-w)]
    return ma_tr

def results_to_S2M_MA(results:List[EpisodeResult], window=0.05):
    episodes = len(results)
    w = int(episodes*window)
    # Simple stdout summary
    steps_to_mastery = [
        r.steps for r in results
    ]
    ma_s2m = [sum(steps_to_mastery[i:i+w])/w for i in range(len(steps_to_mastery)-w)]
    return ma_s2m



if __name__ == "__main__":
    # parser
    p = argparse.ArgumentParser(
        description="PedagoReLearn Tutor Trainer.")
    p.add_argument("--n_rules", type=int, default=3)
    p.add_argument("--n_episodes",type=int, default=2000)
    p.add_argument("--n_seeds",type=int, default=50)
    p.add_argument("--ma_window",type=float, default=0.01)
    p.add_argument("--track_full_result",action='store_true')
    args = p.parse_args()

    # basic train arguments
    n_rules = args.n_rules
    n_episodes = args.n_episodes
    repeats = args.n_seeds
    ma_window = args.ma_window
    track_full_result = args.track_full_result

    w = int(n_episodes*ma_window)
    data = {}

    # train
    for tutormode in ('sarsa','random','fixed'):
        # run sarsa
        rewards = []
        steps = []
        for seed in tqdm(range(repeats)):
            results = train(
                n_rules=n_rules, episodes=n_episodes, 
                seed=seed, tutormode=tutormode,
                save_full_result = track_full_result
            )
            rewards.append(np.convolve([r.total_reward for r in results],np.ones(w)/w,'valid'))
            steps.append(np.convolve([r.steps for r in results],np.ones(w)/w,'valid'))

        
        data[f'rewards_{tutormode}'] = np.quantile(rewards,(.25,.5,.75),axis=0)
        data[f'steps_{tutormode}'] = np.quantile(steps,(.25,.5,.75),axis=0)

    # plot 
    # Total Reward
    plt.figure()
    plt.title(f'Median of Total Reward ({repeats} seeds, window={w})')
    for i, tutormode in enumerate(('sarsa','random','fixed')):
        plt.plot(
            np.arange(w-1,n_episodes),
            data[f'rewards_{tutormode}'][1], 
            label=f'{tutormode}',
            color=f'C{i}')
        plt.fill_between(
            np.arange(w-1,n_episodes),
            data[f'rewards_{tutormode}'][0],
            data[f'rewards_{tutormode}'][2],
            color = f'C{i}',
            alpha=.1
        )
    
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.legend()
    # plt.show()
    plt.savefig(f'fig_results/reward_s{repeats}w{w}.png')
    

    # Steps to Mastery
    plt.figure()
    plt.title(f'Steps to Complete Mastery ({repeats} seeds, window={w})')
    for i, tutormode in enumerate(('sarsa','random','fixed')):
        plt.plot(
            np.arange(w-1,n_episodes),
            data[f'steps_{tutormode}'][1], 
            label=f'{tutormode}',
            color=f'C{i}')
        plt.fill_between(
            np.arange(w-1,n_episodes),
            data[f'steps_{tutormode}'][0],
            data[f'steps_{tutormode}'][2],
            color = f'C{i}',
            alpha=.1
        )
    
    plt.xlabel('episodes')
    plt.ylabel('steps')
    plt.legend()
    # plt.show()
    plt.savefig(f'fig_results/steps_s{repeats}w{w}.png')

    # plot full result
    if track_full_result:
        plot_util.plot("ts_results\\sarsa*")
        plot_util.plot("ts_results\\random*")
        plot_util.plot("ts_results\\fixed*")


