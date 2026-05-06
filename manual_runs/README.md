# Manual run saves

The game writes completed manual player runs here at runtime.

- Only the two fastest manual runs are kept.
- `manual_<time>_<timestamp>.manualrun` stores the player's first-person camera frames, trail, and RL guidance samples.
- Saved runs can be replayed from the player perspective in manual mode.

The generated `.manualrun` files are ignored by git.
