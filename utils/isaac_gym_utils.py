def get_friction(env):
    """Get friction from first shape of first environment."""
    props = env.gym.get_actor_rigid_shape_properties(env.envs[0], 0)
    return props[0].friction


def set_friction(env, value):
    """Set friction for all shapes in all environments."""
    for env_id in range(env.num_envs):
        props = env.gym.get_actor_rigid_shape_properties(env.envs[env_id], 0)
        for s in range(len(props)):
            props[s].friction = value
        env.gym.set_actor_rigid_shape_properties(env.envs[env_id], 0, props)


def print_friction(env, overridden=False):
    """Print current friction value."""
    friction = get_friction(env)
    status = "(overridden)" if overridden else "(default)"
    print(f"Foot friction: {friction:.3f} {status}")


def print_play_status(state, cmd):
    """Print single-line status update for play mode."""
    mode = "PLAY" if state["is_playing"] else "PAUSE"
    print(f"\r[{mode}] Cmd: vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  vyaw={cmd[2]:+.2f}    ", end="", flush=True)