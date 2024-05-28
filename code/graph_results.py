import matplotlib.pyplot as plt
import numpy as np
import os

def graph_training_and_eval_rewards(training_rewards, eval_rewards, save_dir, show_graphs = False):
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

    plt.savefig(os.path.join(save_dir, 'rewards_plot.png'))

    if show_graphs:
        plt.show()

def compare_graph_training_and_eval_rewards(log_dir_1, log_dir_2, save_dir = '.', show_graphs = False):
    train_rew_1 = np.load(os.path.join(log_dir_1, 'training_rewards.npy'), allow_pickle = True)
    train_rew_2 = np.load(os.path.join(log_dir_2, 'training_rewards.npy'), allow_pickle = True)
    eval_rew_1 = np.load(os.path.join(log_dir_1, 'eval_rewards.npy'), allow_pickle = True)
    eval_rew_2 = np.load(os.path.join(log_dir_2, 'eval_rewards.npy'), allow_pickle = True)

    training_maxes_1, training_mins_1, training_means_1 = np.max(train_rew_1, axis = 1), np.min(train_rew_1, axis = 1), np.mean(train_rew_1, axis = 1)
    eval_maxes_1, eval_mins_1, eval_means_1 = np.max(eval_rew_1, axis = 1), np.min(eval_rew_1, axis = 1), np.mean(eval_rew_1, axis = 1)
    training_maxes_2, training_mins_2, training_means_2 = np.max(train_rew_2, axis = 1), np.min(train_rew_2, axis = 1), np.mean(train_rew_2, axis = 1)
    eval_maxes_2, eval_mins_2, eval_means_2 = np.max(eval_rew_2, axis = 1), np.min(eval_rew_2, axis = 1), np.mean(eval_rew_2, axis = 1)
    
    n_1 = train_rew_1.shape[0]
    train_x_1 = np.arange(n_1)
    eval_x_1 = np.linspace(0, n_1, eval_rew_1.shape[0], dtype = np.int32)

    n_2 = train_rew_2.shape[0]
    train_x_2 = np.arange(n_2)
    eval_x_2 = np.linspace(0, n_2, eval_rew_2.shape[0], dtype = np.int32)

    name_1, name_2 = log_dir_1.split('/')[-1], log_dir_2.split('/')[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title("Training Rewards")
    ax1.plot(train_x_1, training_means_1, '-', color = 'tab:blue', label = name_1)
    ax1.fill_between(train_x_1, training_mins_1, training_maxes_1, alpha = 0.2, color = 'tab:blue')
    ax1.plot(train_x_2, training_means_2, '-', color = 'tab:orange', label = name_2)
    ax1.fill_between(train_x_2, training_mins_2, training_maxes_2, alpha = 0.2, color = 'tab:orange')

    ax2.set_title("Evaluation Rewards")
    ax2.plot(eval_x_1, eval_means_1, '-', color = 'tab:blue', label = name_1)
    ax2.fill_between(eval_x_1, eval_mins_1, eval_maxes_1, alpha = 0.2, color = 'tab:blue')
    ax2.plot(eval_x_2, eval_means_2, '-', color = 'tab:orange', label = name_2)
    ax2.fill_between(eval_x_2, eval_mins_2, eval_maxes_2, alpha = 0.2, color = 'tab:orange')

    fig.legend(frameon=False, loc='lower center', ncol=2)
    fig.suptitle(f'{name_1} vs. {name_2} rewards', fontsize = 16)

    plt.savefig(os.path.join(save_dir, 'rewards_plot.png'))

    if show_graphs:
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_reward_file', type = str, required = False)
    parser.add_argument('--eval_reward_file', type = str, required = False)
    parser.add_argument('--log_dir', type = str, required = False)
    parser.add_argument('--log_dir_2', type = str, required = False, default = None)
    parser.add_argument('--save_dir', type = str, required = True)

    args = parser.parse_args()

    if args.log_dir_2 is not None:
        compare_graph_training_and_eval_rewards(args.log_dir, args.log_dir_2, args.save_dir, show_graphs = True)
    else:
        training_rewards, eval_rewards = np.load(args.training_reward_file), np.load(args.eval_reward_file)

        graph_training_and_eval_rewards(training_rewards, eval_rewards, args.save_dir, show_graphs = True)