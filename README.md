# Discrete-Control-for-Atari-Game-by-using-DDQN

## End-to-end (Input to Output)
- State (Input)  
  - Consist Consecutive Samples  
```
    config.state_shape = env.observation_space.shape # Single Sample
```

<p align="center">
  <img src="/README/PongNoFrameskip-v4-history0.jpg" alt="Description" width="100" height="100" border="0" />
  <img src="/README/PongNoFrameskip-v4-history1.jpg" alt="Description" width="100" README="100" border="0" />
  <img src="/README/PongNoFrameskip-v4-history2.jpg" alt="Description" width="100" README="100" border="0" />
  <img src="/README/PongNoFrameskip-v4-history3.jpg" alt="Description" width="100" height="100" border="0" />
</p>
<p align="center">
  <img src="/README/BreakoutNoFrameskip-v4-history0.jpg" alt="Description" width="100" height="100" border="0" />
  <img src="/README/BreakoutNoFrameskip-v4-history1.jpg" alt="Description" width="100" README="100" border="0" />
  <img src="/README/BreakoutNoFrameskip-v4-history2.jpg" alt="Description" width="100" README="100" border="0" />
  <img src="/README/BreakoutNoFrameskip-v4-history3.jpg" alt="Description" width="100" height="100" border="0" />
</p>
<p align="center">
  Figure 1: Consecutive Samples (Pong and Breakout) 
</p>

- Action (Output)  
  - **Discrete** (Select one action)  
```
    config.action_dim = env.action_space.n # Number of Action
```

## Reinforcement Learning DDQN
### Train
```
python DDQN.py --train --env PongNoFrameskip-v4

python DDQN.py --train --env BreakoutNoFrameskip-v4
```

### Test
```
python DDQN.py --test --env PongNoFrameskip-v4 --model_path out/PongNoFrameskip-v4-runx/model_xxxx.pkl

python DDQN.py --test --env BreakoutNoFrameskip-v4 --model_path out/BreakoutNoFrameskip-v4-runx/model_xxxx.pkl
```

### Retrain
```
python DDQN.py --retrain --env PongNoFrameskip-v4 --model_path out/PongNoFrameskip-v4-runx/checkpoint_model/checkpoint_fr_xxxxx.tar

python DDQN.py --retrain --env BreakoutNoFrameskip-v4 --model_path out/BreakoutNoFrameskip-v4-runx/checkpoint_model/checkpoint_fr_xxxxx.tar
```

## Result

PongNoFrameskip-v4 (Discrete)  | BreakoutNoFrameskip-v4 (Discrete)
:-----------------------------:|:-----------------------------:
![](/README/PongNoFrameskip-v4.gif) |  ![](/README/BreakoutNoFrameskip-v4.gif)
<p align="center">
  Figure 2: Reinforcement Learning DDQN on Atari Game
</p>

## Reference
https://github.com/blackredscarf/pytorch-DQN  
[Human-level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236)  
[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)  