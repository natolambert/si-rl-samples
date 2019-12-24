import torch
import random
import numpy as np


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, device, capacity):
        self.device = device
        self.capacity = capacity

        if type(obs_dim) == int:
            self.obses = np.empty((capacity, obs_dim), dtype=np.float32)
            self.next_obses = np.empty((capacity, obs_dim), dtype=np.float32)
        else:
            self.obses = np.empty((capacity, *obs_dim), dtype=np.uint8)
            self.next_obses = np.empty((capacity, *obs_dim), dtype=np.uint8)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate_policy(env, policy, step, L, num_episodes, num_eval_timesteps, video_dir=None):
    returns = []
    for i in range(num_episodes):
        # print(f"Eval episode: {i}...")
        # video = VideoRecorder(env, enabled=video_dir is not None and i == 0)
        s = 0
        obs = env.reset()
        done = False
        total_reward = 0
        while (not done) and (s < num_eval_timesteps):
            with torch.no_grad():
                with eval_mode(policy):
                    action = policy.select_action(obs)

            obs, reward, done, _ = env.step(action)
            # video.record()
            total_reward += reward
            s += 1
        returns.append(total_reward)
    L.info(f" - - Evaluated, mean reward {np.mean(returns)}, n={num_episodes}")
    return returns
