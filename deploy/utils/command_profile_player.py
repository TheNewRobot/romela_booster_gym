import threading
import time


# Ramp rate: 0.2 m/s (or rad/s) per second
# e.g., 0→0.3 m/s takes 1.5s, 0.3→0 takes 1.5s
RAMP_RATE = 0.2
RAMP_UPDATE_HZ = 50  # How often to update velocity during ramps


class CommandProfilePlayer:
    """Plays back velocity command profiles by writing to RemoteControlService attributes.

    Runs in a daemon thread. The existing control loop reads vx/vy/vyaw from
    RemoteControlService getters — this class just overrides the values.
    After all segments play, ramps down to zero and stays there.

    Transitions between segments are automatically ramped at RAMP_RATE to avoid
    sharp velocity steps.
    """

    def __init__(self, remote_control_service, profile_segments):
        self._rcs = remote_control_service
        self._segments = profile_segments
        self._cur = [0.0, 0.0, 0.0]  # current vx, vy, vyaw

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _set_velocity(self, vx, vy, vyaw):
        with self._rcs._lock:
            self._rcs.vx = vx
            self._rcs.vy = vy
            self._rcs.vyaw = vyaw
        self._cur = [vx, vy, vyaw]

    def _ramp_to(self, target_vx, target_vy, target_vyaw):
        """Linearly ramp from current velocity to target. Returns ramp duration."""
        target = [target_vx, target_vy, target_vyaw]
        max_delta = max(abs(t - c) for t, c in zip(target, self._cur))
        if max_delta < 1e-4:
            self._set_velocity(target_vx, target_vy, target_vyaw)
            return 0.0

        ramp_time = max_delta / RAMP_RATE
        dt = 1.0 / RAMP_UPDATE_HZ
        n_steps = max(1, int(ramp_time * RAMP_UPDATE_HZ))
        start = list(self._cur)

        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            vx = start[0] + alpha * (target_vx - start[0])
            vy = start[1] + alpha * (target_vy - start[1])
            vyaw = start[2] + alpha * (target_vyaw - start[2])
            self._set_velocity(vx, vy, vyaw)
            time.sleep(dt)

        return ramp_time

    def _run(self):
        for i, seg in enumerate(self._segments):
            vx = seg.get("vx", 0.0)
            vy = seg.get("vy", 0.0)
            vyaw = seg.get("vyaw", 0.0)
            duration = seg["duration"]

            print(f"[Profile] Segment {i+1}/{len(self._segments)}: "
                  f"vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f} for {duration:.1f}s")

            ramp_time = self._ramp_to(vx, vy, vyaw)
            hold_time = max(0.0, duration - ramp_time)
            if hold_time > 0:
                time.sleep(hold_time)

        # Ramp down to zero and stay there
        self._ramp_to(0.0, 0.0, 0.0)
        print("[Profile] Done. Holding zero velocity.")
