def sac_experiment(cfg):
    # configure target directory for all logs files (binary, text. models etc)
    log_dir_suffix = cfg.log_dir_suffix or strftime("%Y-%m-%d_%H-%M-%S", localtime())
    log_dir = os.path.join(cfg.log_dir or "logs", log_dir_suffix)
    cfg.full_log_dir = log_dir
    os.makedirs(cfg.full_log_dir)

    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    real_env = gym.make(cfg.env.name)

    # instantiate model for this round
    dynamics_model = utils.instantiate(cfg.dynamics_model)

    if cfg.model_env:
        if 'Cartpole' in cfg.env.name:
            m_dir = '/checkpoint/nol/runs/cartpole/cartpole-random-cem-2019-06-20_15-27-12/presets.optimizer:cem/8/trial_19.dat'
            dynamics_model = torch.load(m_dir)['dynamics_model']

        elif 'Cheetah' in cfg.env.name:
            m_dir = '/checkpoint/nol/runs/cheetah-baseline-2019-06-24_18-32-36/training.full_epochs:100/0/trial_299.dat'
            dynamics_model = torch.load(m_dir)['dynamics_model']

        env_train = ModelEnv(real_env, dynamics_model)
        env_test = ModelEnv(real_env, dynamics_model)

        raise NotImplementedError("SAC on dynamics models not yes supported")

    obs_dim = cfg.env.state_size
    action_dim = cfg.env.action_size
    target_entropy_coef = 1
    batch_size = cfg.alg.params.batch_size  # 512
    discount = cfg.alg.trainer.discount  # .99
    tau = cfg.alg.trainer.tau  # .005
    policy_freq = cfg.alg.trainer.target_update_period  # 2
    replay_buffer_size = int(cfg.alg.replay_buffer_size)  # 1000000
    start_steps = cfg.alg.params.start_steps  # 10000
    eval_freq = cfg.alg.params.eval_freq  # 10000
    max_steps = int(cfg.alg.params.max_steps)  # 2E6
    num_eval_episodes = cfg.alg.params.num_eval_episodes  # 5
    num_eval_timesteps = cfg.alg.params.num_eval_timesteps  # 1000
    num_rl_updates = 1
    model_dir = None

    replay_buffer = ReplayBuffer(obs_dim, action_dim, cfg.device, replay_buffer_size)

    policy = SAC(cfg.device, obs_dim, action_dim,
                 hidden_dim=cfg.alg.layer_size,
                 hidden_depth=cfg.alg.num_layers,
                 initial_temperature=cfg.alg.trainer.initial_temp,
                 actor_lr=cfg.alg.trainer.actor_lr,  # 1E-3,
                 critic_lr=cfg.alg.trainer.critic_lr,  # 1E-3,
                 actor_beta=cfg.alg.trainer.actor_beta,  # 0.9,
                 critic_beta=cfg.alg.trainer.critic_beta,  # 0.9,
                 log_std_min=cfg.alg.trainer.log_std_min,  # -10,
                 log_std_max=cfg.alg.trainer.log_std_max)  # 2)

    step = 0
    steps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_success = 0
    episode_step = 0
    saved_idx = 0
    done = True
    returns = None
    target_entropy = -action_dim * target_entropy_coef

    to_plot_rewards = []
    rewards = evaluate_policy(real_env, policy, step, log, num_eval_episodes, num_eval_timesteps, None)
    to_plot_rewards.append(rewards)
    start_time = time()

    env = gym.make(cfg.env.name)

    layout = dict(
        title=f"Learning Curve Reward vs Number of Steps Trials (Env: {cfg.env.name}, Alg: {cfg.alg.name})",
        xaxis={'title': f"Steps*{eval_freq}"},
        yaxis={'title': f"Avg Reward Num:{num_eval_episodes}"},
        font=dict(family='Times New Roman', size=18, color='#7f7f7f'),
        legend={'x': .83, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'}
    )
    lab = cfg.full_log_dir[cfg.full_log_dir.rfind('/') + 1:]

    vis_alg = visdom.Visdom(port=9239)

    while step < max_steps:
        # log.info(f"===================================")
        if step % 1000 == 0:
            log.info(f"Step {step}")

        if done:
            if step != 0:
                # log.info(f"train/duration: {time() - start_time}")
                start_time = time()
                # L.dump(step)

            # Evaluate episode
            if steps_since_eval >= eval_freq:
                steps_since_eval %= eval_freq
                log.info(f"eval/episode: {episode_num}")
                returns = evaluate_policy(env, policy, step, log, num_eval_episodes, num_eval_timesteps,
                                          None)
                to_plot_rewards.append(returns)
                update_plot(to_plot_rewards, vis_alg, layout, lab)

                if model_dir is not None:
                    policy.save(model_dir, step)

            # log.info(f"train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_success = 0
            episode_step = 0
            episode_num += 1

            # log.info(f"train/episode', episode_num, step)

        # Select action randomly or according to policy
        if step < start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                with eval_mode(policy):
                    action = policy.sample_action(obs)

        if step >= start_steps:
            num_updates = start_steps if step == start_steps else num_rl_updates
            for _ in range(num_updates):
                policy.update(
                    replay_buffer,
                    step,
                    log,
                    batch_size,
                    discount,
                    tau,
                    policy_freq,
                    target_entropy=target_entropy)

        next_obs, reward, done, _ = env.step(action)
        # done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        done = 1 if episode_step + 1 == num_eval_timesteps else float(done)
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs

        episode_step += 1
        step += 1
        steps_since_eval += 1
        if (step % eval_freq) == 0:
            trial_log = dict(
                env_name=cfg.env.name,
                trial_num=saved_idx,
                replay_buffer=replay_buffer if cfg.save_replay else [],
                policy=policy,
                rewards=to_plot_rewards,
            )
            save_log(cfg, step, trial_log)
            saved_idx += 1


def save_log(cfg, trial_num, trial_log):
    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(cfg.full_log_dir, name)
    log.info(f"T{trial_num} : Saving log {path}")
    torch.save(trial_log, path)


def train_model(dataset, model, cfg, log):
    if model is None:
        model = utils.instantiate(cfg.dynamics_model)
    # training data from the epsilon generate set
    training_subset, testing_subset = utils.split_dataset(dataset, cfg.training)

    # train model
    cfg.training.epochs = cfg.training.full_epochs
    train_log = model.train(training_subset, testing_subset, cfg.training)
    msg = f"Dynamics model trained"
    if 'train_loss' in train_log is not None:
        msg += f", train loss={train_log.train_loss:.4f}"
    if 'test_loss' in train_log is not None:
        msg += f", test loss={train_log.test_loss:.4f}"
    if 'total_time' in train_log and 'epochs' in train_log:
        eps = (train_log.epochs / train_log.total_time)
        msg += f" ({train_log.total_time:.2f}s, {train_log.epochs} epochs@{eps:.2f}/sec)"
    log.info(msg)
    return model


log = logging.getLogger(__name__)
@hydra.main(config_path='conf/config-base.yaml')
def experiment(cfg):
    # MBPOMain(cfg)
    sac_experiment(cfg)


if __name__ == '__main__':
    sys.exit(experiment())
    # MBPOMain()