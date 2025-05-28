import numpy as np
from pathlib import Path
import datetime

from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros


import torch.multiprocessing as mp
from torch.multiprocessing import Process

# Import only classes from single.py, keep constants here
from single import (
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    Mario,
    MetricLogger
)

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

    # Override specific parameters for parallel training
    mario_agent.exploration_rate = EXPLORATION_RATE_INITIAL
    mario_agent.exploration_rate_decay = EXPLORATION_RATE_DECAY
    mario_agent.exploration_rate_min = EXPLORATION_RATE_MIN
    mario_agent.gamma = GAMMA
    mario_agent.batch_size = BATCH_SIZE
    mario_agent.burnin = BURNIN
    mario_agent.learn_every = LEARN_EVERY
    mario_agent.sync_every = SYNC_EVERY
    mario_agent.save_every = SAVE_EVERY

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
            logger.curr_ep_reward = result['reward']
            logger.curr_ep_length = 1 # Dummy length for main logger's episode
            logger.curr_ep_loss = 0
            logger.curr_ep_q = 0
            logger.curr_ep_loss_length = 0 # Will result in 0 for avg loss/q in this logger

            logger.log_episode() # Saves the above as one "episode" for the main logger

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