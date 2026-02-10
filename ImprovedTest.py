import pygame
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from tqdm import trange
import os

# ================= CONSTANTS =================
WIDTH, HEIGHT = 1000, 700
HOOP_X, HOOP_Y = 800, 150
HOOP_RADIUS = 20
BALL_RADIUS = 8
GRAVITY = 0.3
FPS = 60

# ================= COLORS =================
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
ORANGE = (255, 165, 0)
RED = (255, 0, 0)
BROWN = (139, 69, 19)
BLACK = (0, 0, 0)

# ================= ENVIRONMENT =================
class BasketballShooterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

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

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Basketball PPO Agent - Improved")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        self.total_shots = 0
        self.total_hits = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.start_x = random.uniform(200, 400)
        self.start_y = HEIGHT - 100

        self.ball_x = self.start_x
        self.ball_y = self.start_y

        self.distance = math.hypot(self.start_x - HOOP_X, self.start_y - HOOP_Y)

        obs = np.array([
            self.distance / 1000,
            self.start_y / HEIGHT,
            self.start_x / WIDTH,
            0.0  # difficulty starts at 0
        ], dtype=np.float32)
        return obs, {}

    def step(self, action):
        angle = np.interp(action[0], [-1, 1], [25, 75])
        power = np.interp(action[1], [-1, 1], [12, 22])

        angle_rad = math.radians(angle)

        vx = power * math.cos(angle_rad)
        vy = -power * math.sin(angle_rad)

        scored = False
        min_distance = self.distance
        window_closed = False

        for _ in range(300):
            vy += GRAVITY
            self.ball_x += vx
            self.ball_y += vy

            if not self.render():  # Check if window was closed
                window_closed = True
                break

            current_distance = math.hypot(self.ball_x - HOOP_X, self.ball_y - HOOP_Y)
            if current_distance < min_distance:
                min_distance = current_distance

            if abs(self.ball_x - HOOP_X) < HOOP_RADIUS and abs(self.ball_y - HOOP_Y) < HOOP_RADIUS:
                scored = True
                break

            if self.ball_y > HEIGHT or self.ball_x > WIDTH:
                break

        if scored:
            reward = 100.0
            self.total_hits += 1
        else:
            distance_penalty = -(min_distance / 50.0)
            reward = distance_penalty

        self.total_shots += 1
        
        obs = np.array([
            self.distance / 1000,
            self.start_y / HEIGHT,
            self.start_x / WIDTH,
            0.0  # difficulty starts at 0
        ], dtype=np.float32)
        return obs, reward, True, False, {"window_closed": window_closed}

    def render(self):
        # Check for quit events to prevent window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        self.screen.fill(GREEN)

        pygame.draw.rect(self.screen, BROWN, (0, HEIGHT - 50, WIDTH, 50))
        pygame.draw.circle(self.screen, RED, (HOOP_X, HOOP_Y), HOOP_RADIUS, 3)
        pygame.draw.circle(self.screen, ORANGE, (int(self.ball_x), int(self.ball_y)), BALL_RADIUS)
        
        # Display stats
        hit_rate = (self.total_hits / self.total_shots * 100) if self.total_shots > 0 else 0
        stats_text = f"Shots: {self.total_shots} | Hits: {self.total_hits} | Hit Rate: {hit_rate:.1f}%"
        text_surface = self.font.render(stats_text, True, WHITE)
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)
        return True

# ================= TEST =================
if __name__ == "__main__":
    MODEL_PATH = "./models/ppo_basketball"
    
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print("Model not found! Please train the model first using ImprovedTrain.py")
        exit(1)
    
    print("Loading trained model...")
    env = BasketballShooterEnv(render_mode="human")
    model = PPO.load(MODEL_PATH)
    
    print("Testing the model...")
    print("Close the window to stop testing\n")
    
    total_episodes = 100
    total_reward = 0
    window_open = True
    
    try:
        for episode in range(total_episodes):
            if not window_open:
                break
                
            obs, _ = env.reset()
            episode_reward = 0
            
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            
            # Check if window was closed
            if info.get("window_closed", False):
                window_open = False
                break
            
            total_reward += episode_reward
            
            if (episode + 1) % 10 == 0:
                avg_reward = total_reward / (episode + 1)
                hit_rate = (env.total_hits / env.total_shots * 100) if env.total_shots > 0 else 0
                print(f"Episode {episode + 1}/{total_episodes} | Avg Reward: {avg_reward:.2f} | Hit Rate: {hit_rate:.1f}%")
    
    except KeyboardInterrupt:
        print("\n\nTesting stopped by user")
    
    # Final stats
    hit_rate = (env.total_hits / env.total_shots * 100) if env.total_shots > 0 else 0
    print(f"\nFinal Results:")
    print(f"   Total Shots: {env.total_shots}")
    print(f"   Total Hits: {env.total_hits}")
    print(f"   Hit Rate: {hit_rate:.1f}%")
    if env.total_shots > 0:
        print(f"   Average Reward: {total_reward / env.total_shots:.2f}")
    
    pygame.quit()
    print("\nTest completed!")
