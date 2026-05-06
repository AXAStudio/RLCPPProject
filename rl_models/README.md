# RL model saves

The game writes trained RL model weights here at runtime.

- `autosave_<model>.rlmodel` is updated automatically while training.
- `<model>_<timestamp>.rlmodel` is a manual snapshot from the trainer UI.

The generated `.rlmodel` files are ignored by git.
