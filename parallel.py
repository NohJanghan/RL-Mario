import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, time, os

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import matplotlib.pyplot as plt

# Constants and hyperparameters
RENDER_MODE = 'rgb_array'  # Options: 'human', 'rgb_array'
MARIO_WORLD = 1
MARIO_STAGE = 1
FRAME_SKIP = 4
RESIZE_SHAPE = 84

# Training parameters
EXPLORATION_RATE_INITIAL = 0.9
EXPLORATION_RATE_DECAY = 0.99999
EXPLORATION_RATE_MIN = 0.1
GAMMA = 0.7  # Discount factor
BATCH_SIZE = 128
BURNIN = 1e4  # Min experiences before training
LEARN_EVERY = 3  # Experiences between Q_online updates
SYNC_EVERY = 1e4  # Experiences between Q_target & Q_online sync
SAVE_EVERY = 5e4  # Experiences between saving model
MEMORY_SIZE = 100000
LEARNING_RATE = 0.00025

# Parallel training parameters
NUM_WORKERS = mp.cpu_count()
EPISODES_PER_WORKER = 200

# Environment Wrappers
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        obs, reward, done, trunk, info = None, None, None, None, None # Ensure defined in all paths
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
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
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
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)] # antialias is deprecated, use antialias=True for older torchvision
        )
        observation = transforms(observation).squeeze(0)
        return observation

# Neural Network
class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
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

        self.target = nn.Sequential(
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
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

# Agent
class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = Path(save_dir) # Ensure save_dir is a Path object
        self.save_dir.mkdir(parents=True, exist_ok=True)


        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Mario agent using device: {self.device}")

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
            state_arr = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state_tensor = torch.tensor(state_arr, device=self.device, dtype=torch.float32).unsqueeze(0) # ensure dtype
            action_values = self.net(state_tensor, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state_arr = first_if_tuple(state).__array__()
        next_state_arr = first_if_tuple(next_state).__array__()

        state_tensor = torch.tensor(state_arr, dtype=torch.uint8) # Store as uint8 if they are pixel values
        next_state_tensor = torch.tensor(next_state_arr, dtype=torch.uint8)
        action_tensor = torch.tensor([action], dtype=torch.long)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([done], dtype=torch.bool)

        self.memory.add(TensorDict({
            "state": state_tensor,
            "next_state": next_state_tensor,
            "action": action_tensor,
            "reward": reward_tensor,
            "done": done_tensor
            }, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        # Ensure state and next_state are float for the network
        return state.float(), next_state.float(), action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action):
        # state is already on self.device and float from recall()
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # next_state is already on self.device and float from recall()
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
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate, step=self.curr_step),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0 and self.curr_step > 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0 and self.curr_step > 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if len(self.memory) < self.batch_size: # Ensure enough samples in memory
             return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

# Logging
class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = Path(save_dir) / "log.txt"
        self.save_dir_path = Path(save_dir)
        self.save_dir_path.mkdir(parents=True, exist_ok=True)

        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.writer = SummaryWriter(log_dir=str(self.save_dir_path / "tensorboard"))

        self.ep_rewards_plot = self.save_dir_path / "reward_plot.jpg"
        self.ep_lengths_plot = self.save_dir_path / "length_plot.jpg"
        self.ep_avg_losses_plot = self.save_dir_path / "loss_plot.jpg"
        self.ep_avg_qs_plot = self.save_dir_path / "q_plot.jpg"

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
        if loss is not None and q is not None: # Check for None
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
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

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:] if self.ep_rewards else [0]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:] if self.ep_lengths else [0]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:] if self.ep_avg_losses else [0]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:] if self.ep_avg_qs else [0]), 3)

        self.writer.add_scalar('Metrics/Mean Reward', mean_ep_reward, episode)
        self.writer.add_scalar('Metrics/Mean Length', mean_ep_length, episode)
        self.writer.add_scalar('Metrics/Mean Loss', mean_ep_loss, episode)
        self.writer.add_scalar('Metrics/Mean Q Value', mean_ep_q, episode)
        if epsilon is not None:
             self.writer.add_scalar('Metrics/Epsilon', epsilon, episode)


        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - Step {step} - Epsilon {epsilon:.3f if epsilon is not None else 'N/A'} - Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - Mean Loss {mean_ep_loss} - Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f if epsilon is not None else 0.0:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
        for metric_name_base in ["rewards", "lengths", "avg_losses", "avg_qs"]:
            metric_list_name = f"moving_avg_ep_{metric_name_base}"
            plot_save_path = getattr(self, f"ep_{metric_name_base}_plot")
            plt.clf()
            plt.plot(getattr(self, metric_list_name), label=metric_list_name)
            plt.title(metric_list_name)
            plt.xlabel("Record Call")
            plt.ylabel("Value")
            plt.legend()
            plt.savefig(plot_save_path)

    def close(self):
        self.writer.close()

# Worker function for parallel training
def train_worker(worker_id, shared_results_queue, num_episodes_per_worker, base_save_dir, action_dim, world, stage):
    print(f"Worker {worker_id}: Starting training for {num_episodes_per_worker} episodes.")

    # Each worker has its own environment and agent
    env_id = f"SuperMarioBros-{world}-{stage}-v0"
    env = gym_super_mario_bros.make(env_id, render_mode=RENDER_MODE, apply_api_compatibility=True)
    env = JoypadSpace(env, [["right"], ["right", "A"], ["A"]])
    env = SkipFrame(env, skip=FRAME_SKIP)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=RESIZE_SHAPE)
    env = FrameStack(env, num_stack=4)

    worker_save_dir = Path(base_save_dir) / f"worker_{worker_id}"
    mario_agent = Mario(state_dim=(4, RESIZE_SHAPE, RESIZE_SHAPE), action_dim=action_dim, save_dir=worker_save_dir)

    for episode_num in range(num_episodes_per_worker):
        state_tuple = env.reset()
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple

        episode_reward = 0
        done = False

        while not done:
            action = mario_agent.act(state)
            next_state_tuple, reward, done, trunc, info = env.step(action)
            next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple

            mario_agent.cache(state, next_state, action, reward, done)
            mario_agent.learn()

            episode_reward += reward
            state = next_state

            if info.get("flag_get", False): # End episode if flag is reached
                done = True

        shared_results_queue.put({
            'worker_id': worker_id,
            'episode': episode_num, # Worker's local episode number
            'reward': episode_reward,
            'total_steps': mario_agent.curr_step,
            'final_exploration_rate': mario_agent.exploration_rate
        })
        if (episode_num + 1) % 10 == 0: # Log progress every 10 episodes per worker
             print(f"Worker {worker_id}: Episode {episode_num + 1}/{num_episodes_per_worker} finished. Reward: {episode_reward:.2f}, Total Steps: {mario_agent.curr_step}")

    env.close()
    print(f"Worker {worker_id}: Finished training.")

# Main orchestration for parallel training
def main_parallel():
    num_workers = NUM_WORKERS
    num_episodes_per_worker = EPISODES_PER_WORKER
    mario_world = MARIO_WORLD
    mario_stage = MARIO_STAGE

    print(f"Starting parallel training with {num_workers} workers.")
    print(f"Each worker will run {num_episodes_per_worker} episodes on World {mario_world}-{mario_stage}.")

    # Create a temporary env just to get action_dim safely
    _temp_env = gym_super_mario_bros.make(f"SuperMarioBros-{mario_world}-{mario_stage}-v0", render_mode=RENDER_MODE, apply_api_compatibility=True)
    _temp_env = JoypadSpace(_temp_env, [["right"], ["right", "A"], ["A"]])
    action_dim = _temp_env.action_space.n
    _temp_env.close()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    main_log_dir = Path("runs_parallel_main") / timestamp
    main_log_dir.mkdir(parents=True, exist_ok=True)

    worker_checkpoints_base_dir = Path("checkpoints_parallel_workers") / timestamp
    worker_checkpoints_base_dir.mkdir(parents=True, exist_ok=True)

    # This Mario instance in main is primarily for the logger to fetch initial/dummy values if needed.
    # It doesn't train or interact with the environment directly.
    # Its `curr_step` will remain 0, and `exploration_rate` will be the initial one.
    dummy_mario_for_main_log = Mario(state_dim=(4, RESIZE_SHAPE, RESIZE_SHAPE), action_dim=action_dim, save_dir=main_log_dir / "dummy_agent_main_checkpoints")
    logger = MetricLogger(save_dir=main_log_dir)

    shared_results_queue = mp.Queue()
    processes = []

    for i in range(num_workers):
        p = Process(target=train_worker, args=(i, shared_results_queue, num_episodes_per_worker, worker_checkpoints_base_dir, action_dim, mario_world, mario_stage))
        p.start()
        processes.append(p)

    total_episodes_processed_globally = 0
    total_worker_episodes_to_complete = num_workers * num_episodes_per_worker

    all_worker_rewards = []

    while total_episodes_processed_globally < total_worker_episodes_to_complete:
        try:
            result = shared_results_queue.get(timeout=60) # Wait for 60s for a result

            # Log results using the main logger
            # The main logger's 'episode' concept will be each worker_episode completion.
            logger.curr_ep_reward = result['reward']
            # We don't have per-step loss/q from workers, nor detailed length in a simple way for main logger.
            # So, for this main logger, we treat each worker's episode as a single data point.
            # We can fetch the worker's episode length if it were sent. For now, just reward.
            logger.curr_ep_length = 1 # Dummy length for main logger's episode
            logger.curr_ep_loss = 0
            logger.curr_ep_q = 0
            logger.curr_ep_loss_length = 0 # Will result in 0 for avg loss/q in this logger

            logger.log_episode() # Saves the above as one "episode" for the main logger

            # The 'epsilon' and 'step' for logger.record will use the dummy_mario's values
            # which are not reflective of individual workers' states. This is a known limitation.
            logger.record(
                episode=total_episodes_processed_globally, # A global counter for logging
                epsilon=result.get('final_exploration_rate', dummy_mario_for_main_log.exploration_rate), # Try to use worker's if available
                step=result.get('total_steps', dummy_mario_for_main_log.curr_step) # Try to use worker's if available
            )
            all_worker_rewards.append(result['reward'])
            total_episodes_processed_globally += 1

            if total_episodes_processed_globally % 20 == 0: # Log aggregate stats periodically
                 print(f"Main: Processed {total_episodes_processed_globally}/{total_worker_episodes_to_complete} worker episodes. "
                       f"Avg reward of last 20: {np.mean(all_worker_rewards[-20:]):.2f}")

        except mp.queues.Empty:
            print("Main: Queue empty, waiting for worker results...")
            # Check if processes are still alive; helps in debugging hangs
            alive_procs = [p.is_alive() for p in processes]
            if not any(alive_procs) and total_episodes_processed_globally < total_worker_episodes_to_complete:
                print("Main: All worker processes seem to have terminated prematurely.")
                break # Exit if all workers died and queue is empty


    print("Main: All expected worker episodes processed or workers finished.")
    for p in processes:
        p.join(timeout=30) # Wait for processes to finish
        if p.is_alive():
            print(f"Main: Process {p.pid} did not terminate, will be terminated.")
            p.terminate() # Force terminate if still alive
            p.join()


    logger.close()
    print("Parallel training finished. Logs and checkpoints saved.")
    print(f"Main logs in: {main_log_dir}")
    print(f"Worker checkpoints in: {worker_checkpoints_base_dir}")

if __name__ == "__main__":
    # For CUDA compatibility with multiprocessing, 'spawn' is often preferred or required.
    # Also good for macOS.
    mp.set_start_method('spawn', force=True)
    main_parallel()