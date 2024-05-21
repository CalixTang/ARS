import matplotlib.pyplot as plt
import numpy as np
import os

def graph_training_and_eval_rewards(training_rewards, eval_rewards, log_dir, show_graphs = False):
    #assumes that eval eps are equally spaced between trainings
    training_maxes, training_mins, training_means = np.max(training_rewards, axis = 1), np.min(training_rewards, axis = 1), np.mean(training_rewards, axis = 1)
    eval_maxes, eval_mins, eval_means = np.max(eval_rewards, axis = 1), np.min(eval_rewards, axis = 1), np.mean(eval_rewards, axis = 1)
    
    n = training_rewards.shape[0]
    train_x = np.arange(n)
    eval_x = np.linspace(0, n, eval_rewards.shape[0], dtype = np.int32)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title("Training Rewards")
    ax1.plot(train_x, training_means, '-')
    ax1.fill_between(train_x, training_mins, training_maxes, alpha = 0.2)

    ax2.set_title("Evaluation Rewards")
    ax2.plot(eval_x, eval_means, '-')
    ax2.fill_between(eval_x, eval_mins, eval_maxes, alpha = 0.2)

    plt.savefig(os.path.join(log_dir, 'rewards_plot.png'))

    if show_graphs:
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_reward_file', type = str, required = True)
    parser.add_argument('--eval_reward_file', type = str, required = True)
    parser.add_argument('--log_dir', type = str, required = True)

    args = parser.parse_args()

    training_rewards, eval_rewards = np.load(args.training_reward_file), np.load(args.eval_reward_file)

    graph_training_and_eval_rewards(training_rewards, eval_rewards, args.log_dir, show_graphs = True)