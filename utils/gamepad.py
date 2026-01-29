"""
Xbox Controller Support for Play Mode

Provides analog stick input for velocity commands.
Falls back to keyboard if no controller detected.
"""
import threading

try:
    from inputs import get_gamepad, devices
    INPUTS_AVAILABLE = True
except ImportError:
    INPUTS_AVAILABLE = False


class GamepadController:
    """Xbox controller interface for velocity commands."""
    
    # Stick mapping configuration
    AXIS_MAP = {
        'ABS_Y': 'vx',      # Left stick Y → forward/back
        'ABS_X': 'vy',      # Left stick X → left/right strafe
        'ABS_RX': 'vyaw',   # Right stick X → turn
    }
    
    # Axis normalization (Xbox sticks report -32768 to 32767)
    AXIS_SCALE = 32768.0
    
    # Print threshold (only print when change exceeds this)
    PRINT_THRESHOLD = 0.05
    
    def __init__(self, max_vx=1.0, max_vy=1.0, max_vyaw=1.0, deadzone=0.15, verbose=True):
        self._connected = False
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0
        self._last_print_vx = 0.0
        self._last_print_vy = 0.0
        self._last_print_vyaw = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        
        # Configurable limits
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_vyaw = max_vyaw
        self.deadzone = deadzone
        self.verbose = verbose
        
        if not INPUTS_AVAILABLE:
            print("[Gamepad] 'inputs' library not installed. Run: pip install inputs")
            return
        
        gamepads = [d for d in devices.gamepads]
        if not gamepads:
            print("[Gamepad] No controller detected")
            return
        
        self._connected = True
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"[Gamepad] Connected: {gamepads[0].name}")
    
    def is_connected(self):
        return self._connected
    
    def _apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)
    
    def _poll_loop(self):
        while self._running:
            try:
                events = get_gamepad()
                for event in events:
                    if event.ev_type == 'Absolute' and event.code in self.AXIS_MAP:
                        raw = event.state / self.AXIS_SCALE
                        raw = max(-1.0, min(1.0, raw))
                        value = self._apply_deadzone(raw)
                        
                        axis = self.AXIS_MAP[event.code]
                        with self._lock:
                            if axis == 'vx':
                                self._vx = -value * self.max_vx  # Invert: stick up = forward
                            elif axis == 'vy':
                                self._vy = -value * self.max_vy  # Invert: stick left = +vy
                            elif axis == 'vyaw':
                                self._vyaw = -value * self.max_vyaw  # Invert: stick left = +vyaw
            except Exception:
                self._connected = False
                break
    
    def poll(self):
        """Returns current (vx, vy, vyaw) from controller."""
        if not self._connected:
            return 0.0, 0.0, 0.0
        with self._lock:
            vx, vy, vyaw = self._vx, self._vy, self._vyaw
        
        # Print if changed significantly
        if self.verbose and (abs(vx - self._last_print_vx) > self.PRINT_THRESHOLD or
            abs(vy - self._last_print_vy) > self.PRINT_THRESHOLD or
            abs(vyaw - self._last_print_vyaw) > self.PRINT_THRESHOLD):
            print(f"Velocity: vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}")
            self._last_print_vx = vx
            self._last_print_vy = vy
            self._last_print_vyaw = vyaw
        
        return vx, vy, vyaw
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)