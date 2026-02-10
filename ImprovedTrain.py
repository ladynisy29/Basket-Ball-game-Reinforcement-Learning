import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from tqdm import tqdm
import os
import subprocess
import sys
import time
import webbrowser

# ================= CONSTANTS =================
WIDTH, HEIGHT = 1000, 700
HOOP_X, HOOP_Y = 800, 150
HOOP_RADIUS = 20
GRAVITY = 0.3

# ================= ENVIRONMENT =================
class BasketballShooterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        # Enhanced observation space with more information
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.episode_count = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_count += 1
        
        # Curriculum learning: gradually increase difficulty
        difficulty = min(self.episode_count / 5000, 1.0)  # Faster progression
        
        # Start positions vary by difficulty with smoother progression
        if difficulty < 0.2:
            # Very easy: close range
            self.start_x = random.uniform(650, 750)
        elif difficulty < 0.4:
            # Easy: medium-close range
            self.start_x = random.uniform(550, 750)
        elif difficulty < 0.6:
            # Medium: wide range
            self.start_x = random.uniform(300, 800)
        elif difficulty < 0.8:
            # Hard: extended range
            self.start_x = random.uniform(100, 850)
        else:
            # Very hard: full court
            self.start_x = random.uniform(50, 900)
        
        self.start_y = HEIGHT - 100

        self.distance = math.hypot(self.start_x - HOOP_X, self.start_y - HOOP_Y)

        # Enhanced observation: distance, height, x_position, difficulty_level
        obs = np.array([
            self.distance / 1000,
            self.start_y / HEIGHT,
            self.start_x / WIDTH,
            difficulty
        ], dtype=np.float32)
        return obs, {}

    def step(self, action):
        angle = np.interp(action[0], [-1, 1], [25, 75])
        power = np.interp(action[1], [-1, 1], [12, 22])

        angle_rad = math.radians(angle)

        vx = power * math.cos(angle_rad)
        vy = -power * math.sin(angle_rad)

        x, y = self.start_x, self.start_y

        scored = False
        min_distance = self.distance
        trajectory_points = 0
        max_height = self.start_y
        bounce_count = 0

        for step_count in range(300):
            vy += GRAVITY
            x += vx
            y += vy
            trajectory_points += 1
            
            # Track maximum height for arc bonus
            if y < max_height:
                max_height = y

            current_distance = math.hypot(x - HOOP_X, y - HOOP_Y)
            if current_distance < min_distance:
                min_distance = current_distance

            # Check for perfect shot (direct entry)
            if abs(x - HOOP_X) < HOOP_RADIUS and abs(y - HOOP_Y) < HOOP_RADIUS:
                if y < HOOP_Y:  # Ball coming from above
                    scored = True
                    break

            if y > HEIGHT or x > WIDTH or x < 0:
                break

        # ========== PERFECT SHOT REWARD SYSTEM ==========
        if scored:
            reward = 100.0
            
            # PERFECT SHOT BONUS: Direct entry from above (swish!)
            perfect_entry_bonus = 0
            if y < HOOP_Y:  # Ball came from above
                perfect_entry_bonus = 75.0  # Massive bonus for perfect entry
            
            # DISTANCE BONUS: Long-range shots are cooler
            # Exponential bonus for distance
            distance_bonus = (self.distance / 800) ** 1.3 * 50  # Up to +50 for far shots
            
            # ARC BONUS: Beautiful high arc trajectories
            arc_height = self.start_y - max_height
            arc_bonus = min((arc_height / 120) * 30, 30)  # Up to +30 for high arcs
            
            # EFFICIENCY BONUS: Quick, smooth shots
            efficiency_bonus = max(0, 20 - trajectory_points / 15)  # Up to +20
            
            # ACCURACY BONUS: How close to perfect angle
            accuracy_bonus = max(0, (HOOP_RADIUS - min_distance) / HOOP_RADIUS * 15)
            
            # CONSISTENCY BONUS: Reward consistent good shots
            consistency_bonus = 5
            
            reward += perfect_entry_bonus + distance_bonus + arc_bonus + efficiency_bonus + accuracy_bonus + consistency_bonus
            
        else:
            # Miss penalties and encouragement
            distance_penalty = -(min_distance / 35.0)  # Stricter penalty for accuracy
            
            # Encourage long-range attempts
            long_range_effort = 0
            if self.distance > 600:
                long_range_effort = 10 * (self.distance / 900)  # Strong bonus for long shots
            
            # Efficiency penalty (lower)
            efficiency_penalty = -0.003 * trajectory_points
            
            # Close miss bonus - very important!
            close_miss_bonus = 0
            if min_distance < 30:
                close_miss_bonus = 20  # Strong reward for near misses
            elif min_distance < 60:
                close_miss_bonus = 10
            
            reward = distance_penalty + long_range_effort + efficiency_penalty + close_miss_bonus

        difficulty = min(self.episode_count / 10000, 1.0)
        
        obs = np.array([
            self.distance / 1000,
            self.start_y / HEIGHT,
            self.start_x / WIDTH,
            difficulty
        ], dtype=np.float32)
        
        self.step_count += 1
        return obs, reward, True, False, {}

# ================= TRAIN =================
if __name__ == "__main__":
    MODEL_PATH = "./models/ppo_basketball"
    SAVE_PATH = "./models/"
    
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Start TensorBoard in background
    tensorboard_dir = "./tensorboard/"
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    print("🚀 Starting TensorBoard server...")
    tb_process = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", tensorboard_dir, "--port", "6006"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(3)
    print("✅ TensorBoard started at http://localhost:6006")
    print("📊 Opening in browser...\n")
    
    try:
        webbrowser.open("http://localhost:6006")
    except:
        print("   Tip: Open http://localhost:6006 in your browser\n")
    
    env = make_vec_env(BasketballShooterEnv, n_envs=8)  # 8 parallel environments for faster training
    
    # Check if saved model exists
    if os.path.exists(f"{MODEL_PATH}.zip"):
        print("📦 Old model found. Creating fresh new model with improved features...")
        # Don't load old model since observation space changed
        # Delete old model to avoid confusion
        os.remove(f"{MODEL_PATH}.zip")
        print("🆕 Creating new improved model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,  # Increased for more exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.2,
            verbose=0,
            tensorboard_log="./tensorboard/"
        )
    else:
        print("🆕 Creating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=2e-4,  # Increased for faster learning
            n_steps=4096,  # Increased for better stability
            batch_size=128,  # Increased for better gradient estimates
            gamma=0.995,  # Higher discount factor
            gae_lambda=0.98,  # Better advantage estimation
            ent_coef=0.03,  # Higher entropy for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.3,  # Slightly higher clip range
            verbose=0,
            tensorboard_log="./tensorboard/"
        )

    # Callbacks for better training
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=SAVE_PATH,
        name_prefix="ppo_basketball_checkpoint"
    )

    TOTAL_STEPS = 1_500_000  # Increased for perfect shot learning
    CHUNK = 10_000

    print("🚀 Training started")
    print(f"Total steps: {TOTAL_STEPS:,}")
    print("📊 View progress at: http://localhost:6006\n")
    print("🎯 Model learning:")
    print("   • Perfect shots (swish from above)")
    print("   • Long-range three-pointers")
    print("   • High arc trajectories")
    print("   • Efficient, smooth shots")
    print("   • Close miss bonuses\n")

    try:
        for i in tqdm(range(TOTAL_STEPS // CHUNK), desc="Training PPO"):
            model.learn(
                total_timesteps=CHUNK,
                reset_num_timesteps=False,
                callback=checkpoint_callback if i % 10 == 0 else None
            )
            
            # Save model periodically
            if (i + 1) % 50 == 0:
                model.save(MODEL_PATH)
                print(f"\n✅ Model saved at step {(i+1)*CHUNK}")

        # Final save
        model.save(MODEL_PATH)
        print("\n✅ Training completed! Model saved.")
        
    finally:
        # Keep TensorBoard running after training
        print("\n📊 TensorBoard still running at http://localhost:6006")
        print("Press Ctrl+C in this terminal to stop TensorBoard and exit.")
        try:
            tb_process.wait()
        except KeyboardInterrupt:
            print("\nStopping TensorBoard...")
            tb_process.terminate()
            tb_process.wait()
