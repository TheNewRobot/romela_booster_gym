import torch
from policies.actor_critic import ActorCritic, get_network_dims


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
        model_dict = torch.load(policy_path, map_location="cpu", weights_only=True)
        actor_dims, critic_dims = get_network_dims(cfg, model_dict["model"])
        model = ActorCritic(
            cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"],
            actor_hidden_dims=actor_dims, critic_hidden_dims=critic_dims,
        )
        model.load_state_dict(model_dict["model"])
        model.eval()
        return model, False