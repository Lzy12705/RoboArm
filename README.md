# RoboArm

A Panda robotic arm learning to open a door via reinforcement learning (TD3), simulated in [robosuite](https://robosuite.ai/) / MuJoCo.

The agent controls the arm's joints (`JOINT_VELOCITY` control) to reach the door handle, turn it, and swing the door open. Training uses a custom environment (`CurriculumDoor`) that extends robosuite's built-in `Door` task with extra reward shaping for the door's hinge angle, so the agent is rewarded for actually pushing the door open — not just reaching and holding the handle.

## Setup

All code lives in `tmp/`.

1. Create and activate a virtual environment (Python 3.11 recommended):

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\Activate.ps1
   # macOS / Linux
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   `torch` is pinned loosely in `requirements.txt` — if you need GPU acceleration, install the CUDA build matching your system from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the default CPU wheel.

3. Always run scripts from inside `tmp/` — checkpoints and logs use paths relative to that directory:

   ```bash
   cd tmp
   ```

## Running

### Train

```bash
python main.py
```

- Runs headless (no render window) for speed.
- Automatically resumes from the last saved checkpoint (`tmp/ppo/`) if one exists — prints `Successfully loaded models` on success, or `Failed to load models, starting from scratch` if none is found or the network shapes don't match.
- Saves a checkpoint every 10 episodes.
- Logs scores to TensorBoard under a new timestamped folder per run (`logs/run_<timestamp>/`).
- Runs up to 10,000 episodes; safe to stop anytime (`Ctrl+C`) and resume later — you'll lose at most the last 9 unsaved episodes.

### Watch the trained policy

```bash
python test.py
```

- Opens a render window and plays the current checkpoint with no exploration noise, so you see the policy's real learned behavior.
- Loads from the same `tmp/ppo/` checkpoint `main.py` writes to — always reflects your most recent training.

### View training curves

```bash
tensorboard --logdir=logs --port=6006
```

Run this from inside `tmp/` (or point `--logdir` at `tmp/logs` from the repo root). Then open `http://localhost:6006` in a browser, go to the **Scalars** tab, and turn up the smoothing slider — raw per-episode scores are noisy. Each training run appears as its own separate, toggleable line.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Training entry point |
| `test.py` | Loads the trained model and visualizes it in a render window |
| `td3_torch.py` | TD3 agent: actor/critic networks, learning update, checkpoint save/load |
| `networks.py` | Actor and Critic network definitions (PyTorch) |
| `buffer.py` | Replay buffer used for off-policy training |
| `curriculum_door.py` | Custom `CurriculumDoor` environment — adds hinge-angle reward shaping on top of robosuite's `Door` task |
| `ppo/` | Saved model checkpoints (created automatically) |
| `logs/` | TensorBoard event logs, one subfolder per run (created automatically) |

## Notes

- If you ever change the robot's control mode or action space in `main.py`, update `test.py` to match — otherwise checkpoint loading will fail silently (network shape mismatch) and it'll fall back to an untrained model.

## Credit
Credit to youtube guide: https://youtube.com/playlist?list=PLOkmXPXHDP22VQmr37DFuJr6k30setQ2w&si=aP8A-uFhNXj3BcZ2
