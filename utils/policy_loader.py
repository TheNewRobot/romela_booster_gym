import torch
from policies.actor_critic import ActorCritic


def load_policy(policy_path, cfg):
    """Load policy - supports both .pth (ActorCritic) and .pt (JIT).
    
    Returns:
        tuple: (policy, is_jit) where is_jit indicates if it's a JIT model
    """
    is_jit = policy_path.endswith(".pt")
    
    if is_jit:
        policy = torch.jit.load(policy_path, map_location="cpu")
        policy.eval()
        return policy, True
    else:
        model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
        model_dict = torch.load(policy_path, map_location="cpu", weights_only=True)
        model.load_state_dict(model_dict["model"])
        model.eval()
        return model, False