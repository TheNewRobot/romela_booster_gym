import torch
import torch.nn.functional as F


def _build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ELU())
        prev = h
    layers.append(torch.nn.Linear(prev, output_dim))
    return torch.nn.Sequential(*layers)


class ActorCritic(torch.nn.Module):

    def __init__(self, num_act, num_obs, num_privileged_obs,
                 actor_hidden_dims=None, critic_hidden_dims=None):
        super().__init__()
        assert actor_hidden_dims is not None, \
            "actor_hidden_dims must be specified in config (network.actor_hidden_dims)"
        assert critic_hidden_dims is not None, \
            "critic_hidden_dims must be specified in config (network.critic_hidden_dims)"
        self.critic = _build_mlp(
            num_obs + num_privileged_obs, critic_hidden_dims, 1)
        self.actor = _build_mlp(num_obs, actor_hidden_dims, num_act)
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=-2.0), requires_grad=True)

    def act(self, obs):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)


def _infer_mlp_hidden_dims(state_dict, prefix):
    dims = []
    i = 0
    while f"{prefix}.{i}.weight" in state_dict:
        dims.append(state_dict[f"{prefix}.{i}.weight"].shape[0])
        i += 2
    return dims[:-1]


def get_network_dims(cfg, state_dict=None):
    if "network" in cfg:
        return cfg["network"]["actor_hidden_dims"], cfg["network"]["critic_hidden_dims"]
    if state_dict is not None:
        actor_dims = _infer_mlp_hidden_dims(state_dict, "actor")
        critic_dims = _infer_mlp_hidden_dims(state_dict, "critic")
        return actor_dims, critic_dims
    raise KeyError(
        "Missing 'network' section in config and no state_dict to infer from"
    )
