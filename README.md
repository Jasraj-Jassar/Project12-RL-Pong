# Beginner RL Pong (No External Dependencies)

This is a tiny, beginner-friendly reinforcement learning project. A Q-learning agent learns to keep the Pong ball in play. Everything runs with the Python standard library only.

## Files
- `pong_env.py`: Simple Pong environment (grid-based)
- `q_agent.py`: Q-learning agent
- `train.py`: Training loop
- `ui_app.py`: Tkinter UI to watch training/play
- `config.py`: Small constants

## Quick start
Train the agent:

```bash
python train.py --episodes 500
```

Open the GUI (no terminal rendering):

```bash
python ui_app.py
```

## How it works (short version)
- State is a small tuple: ball position/velocity and paddle position.
- Actions are simple: move paddle up, stay, or move down.
- Rewards: +1 for hitting the ball, -1 for missing.
- The agent learns a Q-table over time.

## Tips to tweak
- Increase `GRID_WIDTH`/`GRID_HEIGHT` in `config.py` for a larger board.
- Increase `MAX_STEPS` for longer episodes.
- Adjust `CELL_SIZE` or `FRAME_DELAY_MS` in `config.py` for UI size/speed.
- Lower `epsilon_decay` in `q_agent.py` for more exploration.
