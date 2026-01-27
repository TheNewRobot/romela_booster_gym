# External Policy Integration

This folder contains adapters for integrating trained policies from external repositories.

## Design Philosophy

- **Inference only** - No training dependencies (Isaac Gym, legged_gym, etc.)
- **Minimal extraction** - Copy only: network architecture, observation specs, checkpoint
- **No upstream sync** - Trained policies are frozen artifacts

## Structure

```
external/
├── README.md
├── base_policy.py          # Abstract adapter interface
├── dribblebot/
│   ├── adapter.py          # Policy adapter implementation
│   ├── network.py          # Network architecture (copied from source)
│   └── checkpoints/
│       └── dribbling.pt
└── goalkeeper/
    └── ...
```

## Adding a New Policy

1. **Extract from source repo:**
   - Network class (`nn.Module`)
   - Observation normalization constants
   - Action scaling/defaults
   - Trained checkpoint (`.pt` file)

2. **Create adapter** (subclass `BasePolicyAdapter`):
   - `build_network()` - Match training architecture exactly
   - `build_observation()` - Map sensor data → input tensor
   - `decode_action()` - Map output → joint targets

## Potential Sources

| Policy | Source | Robot | Notes |
|--------|--------|-------|-------|
| Dribblebot | [Improbable-AI/dribblebot](https://github.com/Improbable-AI/dribblebot) | Go1 (quadruped) | Requires retrain for T1 |
| Goalkeeper | [InternRobotics/Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper) | TBD | TBD |