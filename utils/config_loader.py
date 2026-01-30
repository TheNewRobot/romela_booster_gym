import os
import yaml


def find_experiment_config(policy_path):
    """Auto-detect config.yaml from policy's experiment folder."""
    policy_dir = os.path.dirname(os.path.abspath(policy_path))
    
    # If policy is in nn/ subfolder, go up one level
    if os.path.basename(policy_dir) == "nn":
        experiment_dir = os.path.dirname(policy_dir)
        config_path = os.path.join(experiment_dir, "config.yaml")
        if os.path.exists(config_path):
            return config_path
    return None


def load_config(cfg_file):
    """Load YAML config file."""
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)