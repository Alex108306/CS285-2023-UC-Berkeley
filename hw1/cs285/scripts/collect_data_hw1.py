import argparse
from collections import OrderedDict
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from glob import glob

EXPERTS = {"Ant-v4", "HalfCheetah-v4"}


def run_bc(env, params):
    hparams = OrderedDict()
    hparams["train_batch_size"] = list(range(50, 450 + 1, 100))
    seeds = list(range(5))

    cmd = "python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/{policy}.pkl \
	--env_name {env} --exp_name {exp} --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_{data}.pkl \
	--train_batch_size {train_batch_size} \
    --ep_len {ep_len} --eval_batch_size {eval_batch_size} \
    --seed {seed} --video_log_freq -1 "

    for seed in seeds:
        for train_batch_size in hparams["train_batch_size"]:
            new_cmd = cmd.format(
                env=env,
                exp="bc_"
                + "train_batch_size_"
                + str(train_batch_size)
                + "_seed_"
                + str(seed),
                policy=env.split("-")[0],
                data=env,
                train_batch_size=train_batch_size,
                ep_len=params["ep_len"],
                eval_batch_size=params["eval_batch_size"],
                seed=seed,
            )
            os.system(new_cmd)

    visualizer_bc(env, hparams["train_batch_size"], seeds)


def run_dagger(env, params):
    seeds = list(range(5))

    cmd = "python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/{policy}.pkl \
	--env_name {env} --exp_name {exp} --n_iter {n_iter} \
	--expert_data cs285/expert_data/expert_data_{data}.pkl \
	--train_batch_size {train_batch_size} \
    --ep_len {ep_len} --eval_batch_size {eval_batch_size} \
    --seed {seed} --video_log_freq -1 \
    --do_dagger"

    n_iter = 10

    for seed in seeds:
        new_cmd = cmd.format(
            env=env,
            exp="dagger_"
            + "train_batch_size_"
            + str(params["train_batch_size"])
            + "_seed_"
            + str(seed),
            policy=env.split("-")[0],
            data=env,
            n_iter=n_iter,
            train_batch_size=params["train_batch_size"],
            ep_len=params["ep_len"],
            eval_batch_size=params["eval_batch_size"],
            seed=seed,
        )
        os.system(new_cmd)

    visualizer_dagger(env, params["train_batch_size"], n_iter, seeds)


def visualizer_bc(env, hparams_values, seeds):

    scalars = [
        "Eval_AverageReturn",
        "Eval_StdReturn",
        "Initial_DataCollection_AverageReturn",
    ]

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    mean_rewards = np.zeros(shape=len(hparams_values))
    std_rewards = np.zeros(shape=len(hparams_values))
    expert_rewards = np.zeros(shape=len(hparams_values))

    for seed in seeds:
        for i, train_batch_size in enumerate(hparams_values):
            log_dir = "data/q1_bc_train_batch_size_{}_seed_{}_{}".format(
                str(train_batch_size), str(seed), str(env)
            )
            log_file = os.path.join(log_dir, "event*")
            log_file = glob(log_file)
            loaded_scalar = load_tensorboard_scalars(log_file[0], scalars)
            mean_rewards[i] += loaded_scalar[0][0][2]
            std_rewards[i] += loaded_scalar[1][0][2]
            expert_rewards[i] += loaded_scalar[2][0][2]

    mean_rewards = mean_rewards / len(seeds)
    std_rewards = std_rewards / len(seeds)
    expert_rewards = expert_rewards / len(seeds)

    ax = plot_metrics(
        ax,
        hparams_values,
        mean_rewards,
        std_rewards,
        label="bc_agent_{}".format(env),
        c="r",
    )
    ax = plot_metrics(
        ax, hparams_values, expert_rewards, label="expert_agent_{}".format(env), c="g"
    )

    ax.set_xlabel("Train Batch Size")
    ax.set_ylabel("Average Rewards")

    ax.set_title(
        "Agent with average mean and std of rewards over different train batch sizes, three seeds, and approximately 5 rollouts"
    )
    ax.legend(loc="upper right")

    fig.savefig("figures/bc_train_batch_size_{}.png".format(env))


def visualizer_dagger(env, params_values, n_iter, seeds):

    scalars = [
        "Eval_AverageReturn",
        "Eval_StdReturn",
        "Initial_DataCollection_AverageReturn",
    ]

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    dagger_mean_rewards = np.zeros(shape=n_iter)
    dagger_std_rewards = np.zeros(shape=n_iter)
    expert_rewards = np.zeros(shape=n_iter)

    bc_mean_rewards = np.zeros(shape=n_iter)
    bc_std_rewards = np.zeros(shape=n_iter)

    for seed in seeds:
        log_dir = "data/q2_dagger_train_batch_size_{}_seed_{}_{}".format(
            str(params_values), str(seed), str(env)
        )
        log_file = os.path.join(log_dir, "event*")
        log_file = glob(log_file)
        loaded_scalar = load_tensorboard_scalars(log_file[0], scalars)
        for event in loaded_scalar[0]:
            dagger_mean_rewards[event[1]] += event[2]
        for event in loaded_scalar[1]:
            dagger_std_rewards[event[1]] += event[2]

        log_dir = "data/q1_bc_train_batch_size_{}_seed_{}_{}".format(
            str(params_values), str(seed), str(env)
        )
        log_file = os.path.join(log_dir, "event*")
        log_file = glob(log_file)
        loaded_scalar = load_tensorboard_scalars(log_file[0], scalars)
        bc_mean_rewards[0] += loaded_scalar[0][0][2]
        bc_std_rewards[0] += loaded_scalar[1][0][2]
        expert_rewards[0] += loaded_scalar[2][0][2]

    dagger_mean_rewards = dagger_mean_rewards / len(seeds)
    dagger_std_rewards = dagger_std_rewards / len(seeds)

    bc_mean_rewards.fill(bc_mean_rewards[0])
    bc_std_rewards.fill(bc_std_rewards[0])
    expert_rewards.fill(expert_rewards[0])
    bc_mean_rewards = bc_mean_rewards / len(seeds)
    bc_std_rewards = bc_std_rewards / len(seeds)
    expert_rewards = expert_rewards / len(seeds)

    ax = plot_metrics(
        ax,
        list(range(1, n_iter + 1, 1)),
        dagger_mean_rewards,
        dagger_std_rewards,
        label="dagger_agent_{}".format(env),
        c="r",
    )

    ax = plot_metrics(
        ax,
        list(range(1, n_iter + 1, 1)),
        bc_mean_rewards,
        bc_std_rewards,
        label="bc_agent_{}".format(env),
        c="b",
    )

    ax = plot_metrics(
        ax,
        list(range(1, n_iter + 1, 1)),
        expert_rewards,
        label="expert_agent_{}".format(env),
        c="g",
    )

    ax.set_xlabel("Number Iterations")
    ax.set_ylabel("Average Rewards")

    ax.set_title(
        "Agent with average mean and std of rewards over three seeds, and approximately 5 rollouts using DAgger"
    )
    ax.legend(loc="upper right")

    ax.set_xticks(list(range(1, n_iter + 1, 1)))

    fig.savefig("figures/dagger_train_batch_size_{}.png".format(env))


def plot_metrics(ax, x, y, stds=None, label=None, c="y"):
    x = np.array(x)
    y = np.array(y)

    if stds is not None:
        ax.errorbar(x, y, yerr=stds, label=label, color=c)
    else:
        ax.plot(x, y, label=label, color=c)

    return ax


def load_tensorboard_scalars(log_file, scalars):
    tf_size_guidance = {
        "compressedHistograms": 500,
        "images": 0,
        "scalars": 10000,
        "histograms": 1,
    }
    event_acc = EventAccumulator(log_file, tf_size_guidance)
    event_acc.Reload()

    loaded_scalars = []
    for scalar in scalars:
        loaded_scalars.append(event_acc.Scalars(scalar))

    return loaded_scalars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_len", "-epl", type=int, default=1000)
    parser.add_argument("--eval_batch_size", "-ebs", type=int, default=5000)
    parser.add_argument("--train_batch_size", "-tbs", type=int, default=50)
    args = parser.parse_args()

    params = vars(args)

    for env in EXPERTS:
        run_bc(env, params)
        run_dagger(env, params)


if __name__ == "__main__":
    main()
