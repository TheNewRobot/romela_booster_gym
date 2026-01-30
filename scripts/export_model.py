import os
import yaml
import argparse
import torch
from policies.actor_critic import ActorCritic
from utils.config_loader import find_experiment_config, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=None, help="Explicit config path (auto-detected from checkpoint if not provided)")
    parser.add_argument("--task", type=str, default=None, help="Fallback: task name to load config from envs/locomotion/{task}.yaml")
    args = parser.parse_args()

    # Config resolution: explicit --config > auto-detect from checkpoint > fallback to --task
    cfg_file = None
    if args.config:
        cfg_file = args.config
        print(f"Using explicit config: {cfg_file}")
    else:
        cfg_file = find_experiment_config(args.checkpoint)
        if cfg_file:
            print(f"Auto-detected config: {cfg_file}")
        elif args.task:
            cfg_file = os.path.join("envs", "locomotion", f"{args.task}.yaml")
            print(f"Fallback to task config: {cfg_file}")
        else:
            raise ValueError("Could not find config. Provide --config or --task, or ensure config.yaml exists in experiment folder.")
    cfg = load_config(cfg_file)
    model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
    cfg["basic"]["checkpoint"] = args.checkpoint

    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(model_dict["model"])

    model.eval()
    script_module = torch.jit.script(model.actor)
    
    # Save to models/exported/{task}/{checkpoint_name}.pt
    task_name = cfg["basic"]["task"]
    checkpoint_path = os.path.abspath(cfg["basic"]["checkpoint"])
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(checkpoint_dir) == "nn":
        experiment_name = os.path.basename(os.path.dirname(checkpoint_dir))
    else:
        experiment_name = os.path.basename(checkpoint_dir)
    export_dir = os.path.join("deploy", "models", task_name)
    os.makedirs(export_dir, exist_ok=True)
    save_path = os.path.join(export_dir, f"{experiment_name}.pt")
    
    script_module.save(save_path)
    print(f"Saved model to {save_path}")
