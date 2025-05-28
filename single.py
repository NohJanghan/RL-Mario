import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import numpy as np
from pathlib import Path
import datetime, time

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import matplotlib.pyplot as plt

# Constants and hyperparameters
RENDER_MODE = 'rgb_array'  # Options: 'human', 'rgb_array'
MARIO_LEVEL = "SuperMarioBros-1-1-v0"
FRAME_SKIP = 4
RESIZE_SHAPE = 84

# Training parameters
EXPLORATION_RATE_INITIAL = 1.0
EXPLORATION_RATE_DECAY = 0.99999
EXPLORATION_RATE_MIN = 0.1
GAMMA = 0.9  # Discount factor
BATCH_SIZE = 128
BURNIN = 1e4  # Min experiences before training
LEARN_EVERY = 3  # Experiences between Q_online updates
SYNC_EVERY = 1e4  # Experiences between Q_target & Q_online sync
SAVE_EVERY = 5e4  # Experiences between saving model
MEMORY_SIZE = 100000
EPISODES = 40000

# Learning parameters
LEARNING_RATE = 0.00025

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.float32)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float32)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self._build_net(c, output_dim)
        self.target = self._build_net(c, output_dim)

        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def _build_net(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = EXPLORATION_RATE_INITIAL
        self.exploration_rate_decay = EXPLORATION_RATE_DECAY
        self.exploration_rate_min = EXPLORATION_RATE_MIN
        self.curr_step = 0

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(MEMORY_SIZE, device=torch.device("cpu")))
        self.batch_size = BATCH_SIZE

        self.gamma = GAMMA
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = BURNIN
        self.learn_every = LEARN_EVERY
        self.sync_every = SYNC_EVERY
        self.save_every = SAVE_EVERY


    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            # state가 이미 관측 객체 (예: LazyFrames)라고 가정합니다.
            state_np = state.__array__()
            state_tensor = torch.tensor(state_np, device=self.device).unsqueeze(0)
            action_values = self.net(state_tensor, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        # state와 next_state가 이미 관측 객체 (예: LazyFrames)라고 가정합니다.
        state_np = state.__array__()
        next_state_np = next_state.__array__()

        state_tensor = torch.tensor(state_np)
        next_state_tensor = torch.tensor(next_state_np)
        action_tensor = torch.tensor([action])
        reward_tensor = torch.tensor([reward])
        done_tensor = torch.tensor([done])

        self.memory.add(TensorDict({"state": state_tensor, "next_state": next_state_tensor, "action": action_tensor, "reward": reward_tensor, "done": done_tensor}, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        # Ensure actions are long type for indexing
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action.long()
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0 and self.curr_step > 0: # Avoid saving at step 0
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log.txt" # Changed to .txt for clarity
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        self.writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode()
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None and q is not None : # Ensure loss and q are not None
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0.0 # 부동 소수점으로 초기화
            ep_avg_q = 0.0    # 부동 소수점으로 초기화
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def _get_mean_metric(self, metric_list, window=100):
        if not metric_list:
            return 0.0
        return np.round(np.mean(metric_list[-window:]), 3)

    def _plot_metric(self, data, metric_name_base, plot_save_path, episode_label="Episode"):
        plt.figure()
        # 레이블에서 "avg_" 제거하여 "Average Losses" 대신 "Avg Losses"로 표시
        label_metric_name = metric_name_base.replace("avg_", "avg ").replace("_", " ").title()
        plt.plot(data, label=f"Moving Avg of {label_metric_name}")
        plt.xlabel(episode_label)
        plt.ylabel(label_metric_name)
        plt.title(f"Moving Average of {label_metric_name} over Episodes")
        plt.legend()
        plt.savefig(plot_save_path)
        plt.close()

    def record(self, episode, epsilon, step):
        mean_ep_reward = self._get_mean_metric(self.ep_rewards)
        mean_ep_length = self._get_mean_metric(self.ep_lengths)
        mean_ep_loss = self._get_mean_metric(self.ep_avg_losses)
        mean_ep_q = self._get_mean_metric(self.ep_avg_qs)

        self.writer.add_scalar('Metrics/Mean Reward', mean_ep_reward, episode)
        self.writer.add_scalar('Metrics/Mean Length', mean_ep_length, episode)
        self.writer.add_scalar('Metrics/Mean Loss', mean_ep_loss, episode)
        self.writer.add_scalar('Metrics/Mean Q Value', mean_ep_q, episode)
        self.writer.add_scalar('Metrics/Epsilon', epsilon, episode)


        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon:.3f} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric_name_base in ["rewards", "lengths", "avg_losses", "avg_qs"]:
            data_list = getattr(self, f"moving_avg_ep_{metric_name_base}")
            plot_path = getattr(self, f"ep_{metric_name_base}_plot")
            self._plot_metric(data_list, metric_name_base, plot_path)

    def close(self):
        self.writer.close()


def main():
    # Initialize Super Mario environment
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(MARIO_LEVEL, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(MARIO_LEVEL, render_mode=RENDER_MODE, apply_api_compatibility=True)

    # Limit the action-space to 3 discrete actions: ["right"], ["right", "A"], ["A"]
    env = JoypadSpace(env, [["right"], ["right", "A"], ["A"]])

    # Apply Wrappers
    env = SkipFrame(env, skip=FRAME_SKIP)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=RESIZE_SHAPE)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    mario = Mario(state_dim=(4, RESIZE_SHAPE, RESIZE_SHAPE), action_dim=env.action_space.n, save_dir=save_dir)
    logger = MetricLogger(save_dir)

    print(f"Starting training for {EPISODES} episodes...")

    def _get_obs(obs_or_tuple):
        """환경에서 반환된 관측값 또는 (관측값, 정보) 튜플에서 실제 관측값을 추출합니다."""
        if isinstance(obs_or_tuple, tuple):
            return obs_or_tuple[0]
        return obs_or_tuple

    for e in range(EPISODES):
        state_tuple = env.reset()
        state = _get_obs(state_tuple)

        while True:
            env.render()
            action = mario.act(state)
            # next_state, reward, done, trunc, info 순서 확인 및 trunc 변수 사용
            step_result = env.step(action)
            next_state = _get_obs(step_result[0])
            reward = step_result[1]
            done = step_result[2]
            trunc = step_result[3] # trunc 변수 할당
            info = step_result[4]


            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()
            logger.log_step(reward, loss, q)
            state = next_state

            if done or trunc or info.get("flag_get", False): # trunc 조건 추가
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == EPISODES - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    logger.close()
    print("Training finished.")

if __name__ == "__main__":
    main()