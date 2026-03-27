# Adv-AI-Project

## DQN 
Train:
python train.py --algo dqn

Eval:
python evaluate.py --algo dqn -c results/dqn/model.pt

Results over 100 episodes:
  Mean reward : 247.98
  Std  reward : 56.01
  Min  reward : 38.40
  Max  reward : 317.24

## PPO 
Train: 
python train.py --algo ppo

Eval:
python evaluate.py --algo ppo -c results/ppo/model.pt

Results over 100 episodes:
  Mean reward : 261.52
  Std  reward : 24.74
  Min  reward : 131.41
  Max  reward : 316.12