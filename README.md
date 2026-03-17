# Basketball Shooter AI - Improved Training Guide

## Overview
This repository contains an improved PPO (Proximal Policy Optimization) agent for a basketball shooting simulator. The improvements include:

### Key Enhancements:
1. **Enhanced Reward Shaping**: Instead of binary rewards (-10/+100), the agent now gets:
   - 100 reward for scoring
   - Penalty based on how close the ball got to the hoop (encourages better aiming)
   - Small efficiency penalty (encourages efficient shots)

2. **Better Observation Space**: 
   - Old: Only distance to hoop (1D)
   - New: Distance to hoop + ball height + ball x-position + episode progress (4D) → Better learning

3. **Improved Hyperparameters**:
   - Parallel environments (4 parallel trainings for faster learning)
   - Better gamma (0.99) and GAE lambda (0.95)
   - Entropy coefficient and value function coefficient tuning

4. **Better Training Infrastructure**:
   - Checkpoint saving during training
   - Learning rate reduction for fine-tuning
   - Statistics tracking

5. **Comprehensive Testing**:
   - Real-time visualization with hit rate statistics
   - Detailed performance metrics

---

## Files

### Training
- **ImprovedTrain.py** - Main training script
  - Loads saved model if exists (for fine-tuning)
  - Creates new model if not found
  - Trains for 1,500,000 steps
  - Saves checkpoints every 5,000 steps

- **TrainForCoolShots.py** - Cool shots training script
  - Trains agent with cool shots environment
  - Fine-tunes on the improved environment

### Testing
- **ImprovedTest.py** - Evaluation script
  - Loads trained model
  - Shows real-time rendering
  - Displays hit rate and performance stats
  - Runs for 100 episodes

### Reference (Original Files)
- **Train.py** - Original training script
- **Test.py** - Original testing script
- **new.py** - Alternative environment with visual enhancements

---

## Quick Start

### Step 1: Train the Model
```bash
python ImprovedTrain.py
```
This will:
- Load existing model (if available) for fine-tuning
- Or create a new model from scratch
- Train for 1,500,000 steps
- Save checkpoints every 5,000 steps to `./models/`

### Step 2: Test the Model
```bash
python ImprovedTest.py
```
This will:
- Load the trained model
- Run 100 episodes with visualization
- Display hit rate and performance statistics

### Step 3 (Optional): Train with Cool Shots
To train the agent with cool shots mechanics, run:
```bash
python TrainForCoolShots.py
```
This will:
- Load existing model or create a new one
- Train with cool shots environment
- Fine-tune on the improved environment with additional features

---

## Training Details

### Reward Function (Improved)
```
If ball entered hoop:
  reward = 100.0

Else:
  distance_penalty = -(min_distance / 50.0)  # Closer to hoop = smaller penalty
  efficiency_penalty = -0.01 * trajectory_points  # Shorter shots = smaller penalty
  reward = distance_penalty + efficiency_penalty
```

This encourages the agent to:
1. Get the ball into the hoop (100 reward)
2. Get closer to the hoop even if it misses
3. Make more efficient shots

### Observation Space
The agent observes:
1. **Normalized Distance to Hoop**: `distance / 1000`
2. **Ball Height**: `ball_y / HEIGHT`
3. **Ball X Position**: `ball_x / WIDTH`
4. **Episode Progress**: `step_count / max_steps`

This gives the neural network better context about ball position relative to the hoop and training progress.

### Hyperparameters
- **Learning Rate**: 2e-4 (new model) or 1e-4 (existing model)
- **n_steps**: 4096 (new model) or 2048 (existing model) (steps per update)
- **batch_size**: 128 (new model) or 64 (existing model)
- **gamma**: 0.995 (new model) or 0.99 (existing model) (discount factor)
- **gae_lambda**: 0.98 (new model) or 0.95 (existing model) (generalized advantage estimation)
- **ent_coef**: 0.03 (new model) or 0.02 (existing model) (entropy coefficient for exploration)
- **clip_range**: 0.3 (new model) or 0.2 (existing model) (PPO clip range)
- **n_envs**: 8 (parallel environments)

---

## Performance Tips

### To Improve Training Speed:
1. Increase `n_envs` in ImprovedTrain.py (4 → 8 or 16 if you have GPU)
2. Increase `CHUNK` size for faster saves
3. Use GPU acceleration if available

### To Improve Model Performance:
1. Increase total training steps (1.5M → 2M+)
2. Adjust reward shaping (modify distance_penalty divisor)
3. Train for longer with reduced learning rate
4. Use curriculum learning (gradually increase difficulty)

### To Debug Training:
1. Check TensorBoard logs: `tensorboard --logdir=./tensorboard/`
2. Monitor rewards in the terminal output
3. Test intermediate models in the `./models/` folder

---

## Directory Structure
```
BasketBall Game_Nisy/
├── ImprovedTrain.py         # Main training script
├── ImprovedTest.py          # Testing script
├── TrainForCoolShots.py      # Cool shots training
├── ppo_basketball.zip       # Saved model
├── models/                  # Saved models directory
│   ├── ppo_basketball.zip
│   ├── ppo_basketball_checkpoint_*.zip
│   └── ...
├── tensorboard/             # TensorBoard logs
│   └── PPO_0/
│       └── events.out.tfevents...
└── IMPROVEMENTS_README.md   # This file
```

---

## Expected Performance

### Initial Training (0-300K steps)
- Hit Rate: 5-20%
- The agent is learning the basics

### Mid Training (300K-800K steps)
- Hit Rate: 20-45%
- Better consistency in aiming

### Advanced Training (800K-1.2M steps)
- Hit Rate: 45-65%
- Strong performance on consistent shots

### Fine-tuned Model (1.2M+ steps)
- Hit Rate: 60-75%+
- Excellent shot consistency and accuracy

---

## Troubleshooting

### Model not found error
Run `ImprovedTrain.py` first to create and train a model

### Low performance
- Increase training steps
- Adjust reward shaping
- Check if model is loading correctly

### Out of memory
- Reduce `n_envs` in ImprovedTrain.py
- Reduce `batch_size`
- Reduce `n_steps`

### Training too slow
- Increase `n_envs` (parallel environments)
- Use GPU if available
- Reduce `tensorboard` logging frequency

---

## Advanced Customization

### Change Physics
Modify in `ImprovedTrain.py`:
```python
GRAVITY = 0.3  # Increase for harder physics
HOOP_RADIUS = 20  # Increase to make scoring easier
```

### Adjust Reward Shaping
In the `step()` method:
```python
distance_penalty = -(min_distance / 50.0)  # Change 50 to adjust sensitivity
efficiency_penalty = -0.01 * trajectory_points  # Change 0.01 to adjust
```

### Change Training Duration
```python
TOTAL_STEPS = 1_500_000  # Increase to train longer
CHUNK = 10_000  # Adjust checkpoint frequency (5000 for frequent saves)
```

---

## Next Steps

1. **Train the model**: `python ImprovedTrain.py`
2. **Test performance**: `python ImprovedTest.py`
3. **Fine-tune**: Run training again to improve further
4. **Customize**: Modify reward shaping or physics
5. **Deploy**: Use model for other applications

---

## Requirements
- Python 3.8+
- stable-baselines3
- gymnasium
- pygame
- numpy
- tqdm

Install with:
```bash
pip install stable-baselines3 gymnasium pygame numpy tqdm tensorboard
```
