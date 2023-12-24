import argparse

from rchrl.train import run_rchrl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="rchrl", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=float)
    parser.add_argument("--max_timesteps", default=0.5e6, type=float)

    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--env_name", default="AntMaze", type=str)
    parser.add_argument("--load", default=False, type=bool)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--no_correction", action="store_true")
    parser.add_argument("--inner_dones", action="store_true")
    parser.add_argument("--absolute_goal", action="store_true")
    parser.add_argument("--binary_int_reward", action="store_true")
    parser.add_argument("--load_adj_net", default=False, action="store_true")

    parser.add_argument("--gid", default=0, type=int)
    parser.add_argument("--traj_buffer_size", default=50000, type=int)
    parser.add_argument("--lr_r", default=2e-4, type=float)
    parser.add_argument("--r_margin_pos", default=1.0, type=float)
    parser.add_argument("--r_margin_neg", default=1.2, type=float)
    parser.add_argument("--r_training_epochs", default=25, type=int)
    parser.add_argument("--r_batch_size", default=64, type=int)
    parser.add_argument("--r_hidden_dim", default=128, type=int)
    parser.add_argument("--r_embedding_dim", default=32, type=int)
    parser.add_argument("--goal_loss_coeff", default=20., type=float)

    parser.add_argument("--manager_propose_freq", default=10, type=int)
    parser.add_argument("--train_manager_freq", default=10, type=int)
    parser.add_argument("--man_discount", default=0.99, type=float)
    parser.add_argument("--ctrl_discount", default=0.95, type=float)

    # Manager Parameters
    parser.add_argument("--man_soft_sync_rate", default=0.005, type=float)
    parser.add_argument("--man_batch_size", default=128, type=int)
    parser.add_argument("--man_buffer_size", default=2e5, type=int)
    parser.add_argument("--man_rew_scale", default=0.1, type=float)
    parser.add_argument("--man_act_lr", default=1e-4, type=float)
    parser.add_argument("--man_crit_lr", default=1e-3, type=float)
    parser.add_argument("--candidate_goals", default=10, type=int)
    parser.add_argument("--noise_type", default="normal", type=str, choices=["normal", "ou"])
    parser.add_argument("--man_noise_sigma", default=1., type=float)
    parser.add_argument("--train_man_policy_noise", default=0.2, type=float)
    parser.add_argument("--train_man_noise_clip", default=0.5, type=float)

    # Controller Parameters
    parser.add_argument("--ctrl_soft_sync_rate", default=0.005, type=float)
    parser.add_argument("--ctrl_batch_size", default=128, type=int)
    parser.add_argument("--ctrl_buffer_size", default=2e5, type=int)
    parser.add_argument("--ctrl_rew_scale", default=1.0, type=float)
    parser.add_argument("--ctrl_act_lr", default=1e-4, type=float)
    parser.add_argument("--ctrl_crit_lr", default=1e-3, type=float)

    parser.add_argument("--ctrl_noise_sigma", default=1., type=float)
    parser.add_argument("--train_ctrl_policy_noise", default=0.2, type=float)
    parser.add_argument("--train_ctrl_noise_clip", default=0.5, type=float)

    # Demonstration Parameters
    parser.add_argument("--suboptimal_ratio", default=0, type=float)
    parser.add_argument("--coverage_k", default=50, type=int)
    parser.add_argument("--filter_factor_theta", default=2, type=float)

    # Graph Construction for Manager
    parser.add_argument("--landmark_loss_coeff", default=0.005, type=float)
    parser.add_argument("--delta", type=float, default=2)
    parser.add_argument("--adj_factor", default=0.5, type=float)

    parser.add_argument("--novelty_algo", type=str, default="none", choices=["rnd", "none"])
    parser.add_argument("--use_novelty_landmark", action="store_true")
    parser.add_argument("--close_thr", type=float, default=0.2)
    parser.add_argument("--n_landmark_novelty", type=int, default=20)
    parser.add_argument("--rnd_output_dim", type=int, default=128)
    parser.add_argument("--rnd_lr", type=float, default=1e-3)
    parser.add_argument("--rnd_batch_size", default=128, type=int)
    parser.add_argument("--use_ag_as_input", action="store_true")

    parser.add_argument("--landmark_sampling", type=str, choices=["fps", "none"])
    parser.add_argument('--clip_v', type=float, default=-38., help="clip bound for the planner")
    parser.add_argument("--n_landmark_coverage", type=int, default=20)
    parser.add_argument("--initial_sample", type=int, default=1000)
    parser.add_argument("--goal_thr", type=float, default=-10.)
    parser.add_argument("--conventional_lm_start_step", type=int, default=1.5e6)

    parser.add_argument("--no_pseudo_landmark", action="store_true")
    parser.add_argument("--discard_by_anet", action="store_true")
    parser.add_argument("--automatic_delta_pseudo", action="store_true")

    #Ablation
    parser.add_argument("--no_noise_filtering", action="store_true")
    parser.add_argument("--no_similar_filtering", action="store_true")

    # Run the algorithm
    args = parser.parse_args()

    if args.env_name in [ "AntMazeSparse"]:
        args.man_rew_scale = 1.0

    print('=' * 30)
    for key, val in vars(args).items():
        print('{}: {}'.format(key, val))

    run_rchrl(args)
