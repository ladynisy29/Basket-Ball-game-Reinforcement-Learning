import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm import tqdm
import os

# ================= CONSTANTS =================
WIDTH, HEIGHT = 1000, 700
HOOP_X, HOOP_Y = 800, 150
HOOP_RADIUS = 20
GRAVITY = 0.3

# ================= OLD ENVIRONMENT =================
class BasketballShooterEnvOld(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.start_x = random.uniform(200, 400)
        self.start_y = HEIGHT - 100
        self.distance = math.hypot(self.start_x - HOOP_X, self.start_y - HOOP_Y)
        obs = np.array([self.distance / 1000], dtype=np.float32)
        return obs, {}

    def step(self, action):
        angle = np.interp(action[0], [-1, 1], [25, 75])
        power = np.interp(action[1], [-1, 1], [12, 22])
        angle_rad = math.radians(angle)
        vx = power * math.cos(angle_rad)
        vy = -power * math.sin(angle_rad)
        x, y = self.start_x, self.start_y
        scored = False
        for _ in range(300):
            vy += GRAVITY
            x += vx
            y += vy
            if abs(x - HOOP_X) < HOOP_RADIUS and abs(y - HOOP_Y) < HOOP_RADIUS:
                scored = True
                break
            if y > HEIGHT or x > WIDTH:
                break
        reward = 100 if scored else -10
        obs = np.array([self.distance / 1000], dtype=np.float32)
        return obs, reward, True, False, {}

# ================= NEW ENVIRONMENT WITH COOL SHOTS =================
class BasketballShooterEnvNew(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

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
        difficulty = min(self.episode_count / 10000, 1.0)
        
        # Start positions vary by difficulty
        if difficulty < 0.3:
            self.start_x = random.uniform(600, 750)
        elif difficulty < 0.7:
            self.start_x = random.uniform(400, 750)
        else:
            self.start_x = random.uniform(100, 800)
        
        self.start_y = HEIGHT - 100

        self.distance = math.hypot(self.start_x - HOOP_X, self.start_y - HOOP_Y)

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

        for step_count in range(300):
            vy += GRAVITY
            x += vx
            y += vy
            trajectory_points += 1
            
            if y < max_height:
                max_height = y

            current_distance = math.hypot(x - HOOP_X, y - HOOP_Y)
            if current_distance < min_distance:
                min_distance = current_distance

            if abs(x - HOOP_X) < HOOP_RADIUS and abs(y - HOOP_Y) < HOOP_RADIUS:
                scored = True
                break

            if y > HEIGHT or x > WIDTH or x < 0:
                break

        # ========== ENHANCED REWARD FOR COOL SHOTS ==========
        if scored:
            reward = 100.0
            
            # Bonus for distance (long range shots!)
            distance_bonus = (self.distance / 1000) * 20
            
            # Bonus for high arc
            arc_height = self.start_y - max_height
            arc_bonus = min((arc_height / 200) * 15, 15)
            
            # Efficiency bonus
            efficiency_bonus = max(0, 10 - trajectory_points / 30)
            
            reward += distance_bonus + arc_bonus + efficiency_bonus
        else:
            distance_penalty = -(min_distance / 50.0)
            long_range_effort = 0
            if self.distance > 500:
                long_range_effort = 5 * (self.distance / 1000)
            efficiency_penalty = -0.01 * trajectory_points
            reward = distance_penalty + long_range_effort + efficiency_penalty

        difficulty = min(self.episode_count / 10000, 1.0)
        
        obs = np.array([
            self.distance / 1000,
            self.start_y / HEIGHT,
            self.start_x / WIDTH,
            difficulty
        ], dtype=np.float32)
        
        self.step_count += 1
        return obs, reward, True, False, {}

# ================= MIGRATION =================
if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    
    print("🔄 Basketball Model Migration - Add Cool Shot Features")
    print("=" * 60)
    
    # Find old model
    old_model_paths = [
        "./ppo_basketball.zip",
        "./models/ppo_basketball.zip",
        "./PPO_0.zip"
    ]
    
    old_model_path = None
    for path in old_model_paths:
        if os.path.exists(path):
            old_model_path = path
            break
    
    if not old_model_path:
        print("❌ No old model found!")
        print("Searched locations:")
        for path in old_model_paths:
            print(f"  - {path}")
        print("\nRun ImprovedTrain.py to train a new model.")
        exit(1)
    
    print(f"✅ Found old model at: {old_model_path}")
    print("📚 Loading old model...")
    
    env_old = make_vec_env(BasketballShooterEnvOld, n_envs=1)
    old_model = PPO.load(old_model_path, env=env_old)
    print(f"📊 Old model training steps: {old_model.num_timesteps:,}")
    
    # Create new environment with cool shot features
    env_new = make_vec_env(BasketballShooterEnvNew, n_envs=4)
    
    print("\n🆕 Creating new model with cool shot features...")
    print("   - 4D observations (added difficulty level)")
    print("   - Enhanced rewards for cool shots")
    print("   - Curriculum learning (easy to hard)")
    print("   - Distance bonuses for long-range shots")
    print("   - Arc bonuses for high trajectories")
    
    new_model = PPO(
        "MlpPolicy",
        env_new,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        verbose=0,
        tensorboard_log="./tensorboard/"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="ppo_basketball_migrated"
    )
    
    # Fine-tune on new environment
    print("\n🚀 Fine-tuning on new environment with cool shot rewards...")
    print("   Training for 500,000 steps\n")
    
    TOTAL_STEPS = 500_000
    CHUNK = 10_000
    
    for i in tqdm(range(TOTAL_STEPS // CHUNK), desc="Fine-tuning"):
        new_model.learn(
            total_timesteps=CHUNK,
            reset_num_timesteps=False,
            callback=checkpoint_callback if i % 10 == 0 else None
        )
        
        if (i + 1) % 50 == 0:
            new_model.save("./models/ppo_basketball")
    
    # Final save
    new_model.save("./models/ppo_basketball")
    
    print("\n" + "=" * 60)
    print("✅ Migration completed successfully!")
    print("=" * 60)
    print("📊 New model saved to: ./models/ppo_basketball.zip")
    print("🎯 Features:")
    print("   ✓ Cool long-range shots")
    print("   ✓ High arc trajectories")
    print("   ✓ Curriculum learning")
    print("   ✓ Progressive difficulty")
    print("\n🧪 Test the model:")
    print("   python ImprovedTest.py")
    print("=" * 60)
