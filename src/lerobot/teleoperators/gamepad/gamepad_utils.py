#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import struct
import threading
import time

from ..utils import TeleopEvents


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = TeleopEvents.FAILURE
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z


class GamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info(f"Initialized gamepad: {self.joystick.get_name()}")

        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print("  B/Circle button: Exit")
        print("  Y/Triangle button: End episode with SUCCESS")
        print("  A/Cross button: End episode with FAILURE")
        print("  X/Square button: Rerecord episode")

    def stop(self):
        """Clean up pygame resources."""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 3:
                    self.episode_end_status = TeleopEvents.SUCCESS
                # A button (1) for failure
                elif event.button == 1:
                    self.episode_end_status = TeleopEvents.FAILURE
                # X button (0) for rerecord
                elif event.button == 0:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE

                # RB button (6) for closing gripper
                elif event.button == 6:
                    self.close_gripper_command = True

                # LT button (7) for opening gripper
                elif event.button == 7:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [0, 2, 3]:
                    self.episode_end_status = None

                elif event.button == 6:
                    self.close_gripper_command = False

                elif event.button == 7:
                    self.open_gripper_command = False

            # Check for RB button (typically button 5) for intervention flag
            if self.joystick.get_button(5):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        try:
            # Read joystick axes
            # Left stick X and Y (typically axes 0 and 1)
            y_input = self.joystick.get_axis(0)  # Up/Down (often inverted)
            x_input = self.joystick.get_axis(1)  # Left/Right

            # Right stick Y (typically axis 3 or 4)
            z_input = self.joystick.get_axis(3)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Calculate deltas (note: may need to invert axes depending on controller)
            delta_x = -x_input * self.x_step_size  # Forward/backward
            delta_y = -y_input * self.y_step_size  # Left/right
            delta_z = -z_input * self.z_step_size  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states
        self.buttons = {}

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        logging.error(
            "No gamepad found, check the connection and the product string in HID to add your gamepad"
        )
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        import hid

        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            logging.info(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            logging.info(f"Connected to {manufacturer} {product}")

            logging.info("Gamepad controls (HID mode):")
            logging.info("  Left analog stick: Move in X-Y plane")
            logging.info("  Right analog stick: Move in Z axis (vertical)")
            logging.info("  Button 1/B/Circle: Exit")
            logging.info("  Button 2/A/Cross: End episode with SUCCESS")
            logging.info("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            logging.error(f"Error opening gamepad: {e}")
            logging.error("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            # Interpret gamepad data - this will vary by controller model
            # These offsets are for the Logitech RumblePad 2
            if data and len(data) >= 8:
                # Normalize joystick values from 0-255 to -1.0-1.0
                self.left_y = (data[1] - 128) / 128.0
                self.left_x = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # Apply deadzone
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                # Parse button states (byte 5 in the Logitech RumblePad 2)
                buttons = data[5]

                # Check if RB is pressed then the intervention flag should be set
                self.intervention_flag = data[6] in [2, 6, 10, 14]

                # Check if RT is pressed
                self.open_gripper_command = data[6] in [8, 10, 12]

                # Check if LT is pressed
                self.close_gripper_command = data[6] in [4, 6, 12]

                # Check if Y/Triangle button (bit 7) is pressed for saving
                # Check if X/Square button (bit 5) is pressed for failure
                # Check if A/Cross button (bit 4) is pressed for rerecording
                if buttons & 1 << 7:
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif buttons & 1 << 5:
                    self.episode_end_status = TeleopEvents.FAILURE
                elif buttons & 1 << 4:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE
                else:
                    self.episode_end_status = None

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_x * self.x_step_size  # Forward/backward
        delta_y = -self.left_y * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z


class PS5JoystickController:
    """
    PS5 DualSense (or PS4 DualShock 4) controller for direct joint-space teleoperation
    of SO-100/101 robot arms via USB HID.

    Adapted from the PS4JoystickController reference implementation. Uses inverse kinematics
    to translate left-stick X/Y (Cartesian space) into shoulder_lift and elbow_flex angles,
    while right stick directly controls wrist_flex/wrist_roll, and triggers control the gripper.

    Control mapping:
      Left stick X        → shoulder_pan (arm rotation left/right)
      Left stick Y        → shoulder_lift (forward/backward reach)
      Right stick Y       → elbow_flex (arm elevation)
      Right stick X       → wrist_roll
      D-pad Up/Down       → wrist_flex
      R2 analog           → gripper close
      L2 analog           → gripper open
      Circle (O) held     → return to startup position (gradual)
      PS button           → toggle gyro mode (accelerometer-based wrist control)
      Circle (O)          → reset to initial position
      Triangle (T)        → end episode: SUCCESS
      Cross (X)           → end episode: FAILURE
      Square (S)          → rerecord episode
      L1 + R1             → intervention flag

    Returns positions as {"motor_name.pos": value} matching SO-100/101 action format.
    """

    # Sony vendor ID
    VENDOR_ID = 0x054C

    # Supported product IDs (tried in order)
    PRODUCT_IDS = {
        0x0CE6: "PS5 DualSense",
        0x09CC: "PS4 DualShock 4",
        0x05C4: "PS4 DualShock 4 (v1)",
    }

    # Motor names matching SO-100/101 follower
    MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(
        self,
        motor_names=None,
        l1=117.0,  # Length of first arm segment in mm (kept for potential future IK use)
        l2=136.0,  # Length of second arm segment in mm
    ):
        self.motor_names = motor_names if motor_names else self.MOTOR_NAMES
        # Positions are initialized to zero and overwritten by initialize_from_obs()
        self.current_positions = {name: 0.0 for name in self.motor_names}
        self.new_positions = self.current_positions.copy()
        self._startup_positions = None  # set in initialize_from_obs()

        self.l1 = l1
        self.l2 = l2

        # Gamepad axis state
        self.axes = {"RX": 0.0, "RY": 0.0, "LX": 0.0, "LY": 0.0, "L2": 0.0, "R2": 0.0}

        # Button state
        self.buttons = {
            "L2": 0, "R2": 0,
            "DPAD_LEFT": 0, "DPAD_RIGHT": 0, "DPAD_UP": 0, "DPAD_DOWN": 0,
            "X": 0, "O": 0, "T": 0, "S": 0,
            "L1": 0, "R1": 0,
            "SHARE": 0, "OPTIONS": 0,
            "PS": 0, "L3": 0, "R3": 0,
        }
        self.previous_buttons = self.buttons.copy()

        # D-pad direction lookup
        self.DPAD_DIRECTIONS = {
            0x00: "DPAD_UP", 0x01: "DPAD_UP_RIGHT", 0x02: "DPAD_RIGHT",
            0x03: "DPAD_DOWN_RIGHT", 0x04: "DPAD_DOWN", 0x05: "DPAD_DOWN_LEFT",
            0x06: "DPAD_LEFT", 0x07: "DPAD_UP_LEFT", 0x08: "DPAD_NEUTRAL", 0x0F: "DPAD_NEUTRAL",
        }

        # HID device and threading
        self.device = None
        self.pid = None  # detected product ID
        self.running = False
        self.light_bar_color = (0, 0, 255)  # default blue

        # Episode end status (set by button presses, consumed by get_episode_end_status)
        self._episode_status = None

        # Gyro / accelerometer state
        self.gyro_mode = False
        self.gyro_reference = {"pitch": 0.0, "roll": 0.0}
        self.pitch_deg = 0.0
        self.roll_deg = 0.0

        # Whether current_positions has been synced to the robot's actual state.
        # Until True, get_command() is blocked and the read loop skips position updates.
        self._positions_initialized = False

        # Per-joint degree limits derived from the robot's calibration.
        # Populated by initialize_from_obs(); None means no per-joint check.
        self._joint_limits: dict | None = None

        self.lock = threading.Lock()

        self.connect()

        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

        self._set_light_bar(*self.light_bar_color)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        import hid

        for pid, name in self.PRODUCT_IDS.items():
            try:
                self.device = hid.Device(self.VENDOR_ID, pid)
                self.pid = pid
                logging.info(f"Connected to {name}: {self.device.manufacturer} {self.device.product}")
                self.running = True
                print(f"Connected to {name}")
                print("Controls:")
                print("  Left stick X     → shoulder_pan (rotation)")
                print("  Left stick Y     → shoulder_lift (forward/backward)")
                print("  Right stick Y    → elbow_flex (elevation)")
                print("  Right stick X    → wrist_roll")
                print("  D-pad Up/Down    → wrist_flex")
                print("  Circle (O) held  → return to startup position")
                print("  R2 / L2          → gripper close / open")
                print("  PS button        → toggle gyro wrist mode")
                print("  Circle (O)       → reset to initial position")
                print("  Triangle (T)     → SUCCESS")
                print("  Cross (X)        → FAILURE")
                print("  Square (S)       → rerecord")
                return
            except OSError:
                continue

        logging.error("No PS4/PS5 controller found. Connect via USB.")
        raise RuntimeError(
            "No PS4/PS5 controller found. Connect via USB and ensure 'hid' is installed: pip install hid"
        )

    def disconnect(self):
        self.running = False
        if self.device:
            self.device.close()
            self.device = None
            logging.info("Controller disconnected.")

    def stop(self):
        self.disconnect()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Background read loop
    # ------------------------------------------------------------------

    def read_loop(self):
        while self.running:
            try:
                data = self.device.read(64, timeout=100)
                if data:
                    self._process_gamepad_input(data)
            except Exception as e:
                logging.error(f"Controller read error: {e}")
                time.sleep(1)
                try:
                    self.connect()
                except RuntimeError:
                    pass

    def _process_gamepad_input(self, data):
        """Parse a raw HID report and update axes/buttons, then update positions."""
        with self.lock:
            if self.pid == 0x0CE6:
                # PS5 DualSense USB (report ID 0x01)
                # Byte layout:
                #  [1-4]  LX, LY, RX, RY
                #  [5-6]  L2 analog, R2 analog
                #  [7]    sequence counter
                #  [8]    dpad (bits 0-3) | Square(4) | Cross(5) | Circle(6) | Triangle(7)
                #  [9]    L1(0) | R1(1) | L2dig(2) | R2dig(3) | Create(4) | Options(5) | L3(6) | R3(7)
                #  [10]   PS(0) | Touchpad(1) | Mute(2)
                #  [27-38] gyro x/y/z + accel x/y/z (int16 LE)
                lx = data[1] - 128
                ly = data[2] - 128
                rx = data[3] - 128
                ry = data[4] - 128

                self.axes["LX"] = self._filter_deadzone(lx / 128.0)
                self.axes["LY"] = self._filter_deadzone(-ly / 128.0)
                self.axes["RX"] = self._filter_deadzone(rx / 128.0)
                self.axes["RY"] = self._filter_deadzone(-ry / 128.0)

                self.axes["L2"] = data[5] / 255.0
                self.axes["R2"] = data[6] / 255.0

                # D-pad and face buttons (byte 8)
                btn8 = data[8]
                dpad_bits = btn8 & 0x0F
                dpad_direction = self.DPAD_DIRECTIONS.get(dpad_bits, "DPAD_NEUTRAL")
                self.buttons["DPAD_UP"] = 1 if "UP" in dpad_direction else 0
                self.buttons["DPAD_DOWN"] = 1 if "DOWN" in dpad_direction else 0
                self.buttons["DPAD_LEFT"] = 1 if "LEFT" in dpad_direction else 0
                self.buttons["DPAD_RIGHT"] = 1 if "RIGHT" in dpad_direction else 0
                # Square=4, Cross=5, Circle=6, Triangle=7
                self.buttons["S"] = 1 if btn8 & 0x10 else 0
                self.buttons["X"] = 1 if btn8 & 0x20 else 0
                self.buttons["O"] = 1 if btn8 & 0x40 else 0
                self.buttons["T"] = 1 if btn8 & 0x80 else 0

                # Shoulder/menu buttons (byte 9)
                btn9 = data[9]
                self.buttons["L1"]      = 1 if btn9 & 0x01 else 0
                self.buttons["R1"]      = 1 if btn9 & 0x02 else 0
                self.buttons["L2"]      = 1 if btn9 & 0x04 else 0
                self.buttons["R2"]      = 1 if btn9 & 0x08 else 0
                self.buttons["SHARE"]   = 1 if btn9 & 0x10 else 0
                self.buttons["OPTIONS"] = 1 if btn9 & 0x20 else 0
                self.buttons["L3"]      = 1 if btn9 & 0x40 else 0
                self.buttons["R3"]      = 1 if btn9 & 0x80 else 0

                # Special buttons (byte 10)
                btn10 = data[10]
                self.buttons["PS"] = 1 if btn10 & 0x01 else 0

                # Accelerometer (bytes 33-38, int16 LE): ax, ay, az
                if len(data) >= 39:
                    accel_x_raw = struct.unpack("<h", bytes(data[33:35]))[0]
                    accel_y_raw = struct.unpack("<h", bytes(data[35:37]))[0]
                    accel_z_raw = struct.unpack("<h", bytes(data[37:39]))[0]
                    self._update_tilt(accel_x_raw, accel_y_raw, accel_z_raw)

            else:
                # PS4 DualShock 4 USB (report ID 0x01)
                # Byte layout identical to reference PS4JoystickController:
                #  [1-4]  LX, LY, RX, RY
                #  [5]    dpad (bits 0-3) | S(4) | X(5) | O(6) | T(7)
                #  [6]    L1(0) | R1(1) | L2(2) | R2(3) | SHARE(4) | OPTIONS(5) | L3(6) | R3(7)
                #  [7]    PS(0) | TOUCHPAD(1)
                #  [8]    L2 analog
                #  [9]    R2 analog
                #  [19-24] accel x/y/z (int16 LE)
                lx = data[1] - 128
                ly = data[2] - 128
                rx = data[3] - 128
                ry = data[4] - 128

                self.axes["LX"] = self._filter_deadzone(lx / 128.0)
                self.axes["LY"] = self._filter_deadzone(-ly / 128.0)
                self.axes["RX"] = self._filter_deadzone(rx / 128.0)
                self.axes["RY"] = self._filter_deadzone(-ry / 128.0)

                btn5 = data[5]
                dpad_bits = btn5 & 0x0F
                dpad_direction = self.DPAD_DIRECTIONS.get(dpad_bits, "DPAD_NEUTRAL")
                self.buttons["DPAD_UP"] = 1 if "UP" in dpad_direction else 0
                self.buttons["DPAD_DOWN"] = 1 if "DOWN" in dpad_direction else 0
                self.buttons["DPAD_LEFT"] = 1 if "LEFT" in dpad_direction else 0
                self.buttons["DPAD_RIGHT"] = 1 if "RIGHT" in dpad_direction else 0
                self.buttons["S"] = 1 if btn5 & 0x10 else 0
                self.buttons["X"] = 1 if btn5 & 0x20 else 0
                self.buttons["O"] = 1 if btn5 & 0x40 else 0
                self.buttons["T"] = 1 if btn5 & 0x80 else 0

                btn6 = data[6]
                self.buttons["L1"]      = 1 if btn6 & 0x01 else 0
                self.buttons["R1"]      = 1 if btn6 & 0x02 else 0
                self.buttons["L2"]      = 1 if btn6 & 0x04 else 0
                self.buttons["R2"]      = 1 if btn6 & 0x08 else 0
                self.buttons["SHARE"]   = 1 if btn6 & 0x10 else 0
                self.buttons["OPTIONS"] = 1 if btn6 & 0x20 else 0
                self.buttons["L3"]      = 1 if btn6 & 0x40 else 0
                self.buttons["R3"]      = 1 if btn6 & 0x80 else 0

                btn7 = data[7]
                self.buttons["PS"] = 1 if btn7 & 0x01 else 0

                self.axes["L2"] = data[8] / 255.0
                self.axes["R2"] = data[9] / 255.0

                if len(data) >= 25:
                    accel_x_raw = struct.unpack("<h", bytes(data[19:21]))[0]
                    accel_y_raw = struct.unpack("<h", bytes(data[21:23]))[0]
                    accel_z_raw = struct.unpack("<h", bytes(data[23:25]))[0]
                    self._update_tilt(accel_x_raw, accel_y_raw, accel_z_raw)

            # PS button → toggle gyro mode
            if self.buttons["PS"] == 1 and self.previous_buttons["PS"] == 0:
                self.toggle_gyro_mode()

            # Episode control buttons (rising edge)
            if self.buttons["T"] == 1 and self.previous_buttons["T"] == 0:
                self._episode_status = TeleopEvents.SUCCESS
            elif self.buttons["X"] == 1 and self.previous_buttons["X"] == 0:
                self._episode_status = TeleopEvents.FAILURE
            elif self.buttons["S"] == 1 and self.previous_buttons["S"] == 0:
                self._episode_status = TeleopEvents.RERECORD_EPISODE

            self.previous_buttons = self.buttons.copy()
            axes = self.axes.copy()
            buttons = self.buttons.copy()

        if self._positions_initialized:
            self._update_positions(axes, buttons)

    def initialize_from_obs(self, obs: dict, joint_limits: dict | None = None):
        """
        Seed current_positions from the robot's live observation dict.
        Must be called before get_command() will return anything.

        Args:
            obs: Robot observation dict with keys 'motor_name.pos'.
            joint_limits: Optional dict {motor_name: (min_deg, max_deg)} derived
                          from the robot's calibration. If provided, replaces the
                          Cartesian-only validation with per-joint degree limits.
        """
        with self.lock:
            for name in self.motor_names:
                key = f"{name}.pos"
                if key in obs:
                    self.current_positions[name] = float(obs[key])
            if joint_limits is not None:
                self._joint_limits = joint_limits
                logging.info(f"PS5 joint limits from calibration: {joint_limits}")
            # Clamp initial positions to valid range before saving as startup reference
            self._clamp_to_limits(self.current_positions)
            self._startup_positions = self.current_positions.copy()
            self._positions_initialized = True
        logging.info(f"PS5 controller initialized from robot observation: {self.current_positions}")

    def _filter_deadzone(self, value, threshold=0.1):
        """Apply a deadzone to joystick input to avoid drift."""
        if abs(value) < threshold:
            return 0.0
        return value

    def _update_tilt(self, accel_x_raw, accel_y_raw, accel_z_raw):
        """Compute smoothed pitch/roll from raw accelerometer data."""
        roll_rad = -math.atan2(accel_x_raw, math.sqrt(accel_y_raw**2 + accel_z_raw**2))
        pitch_rad = math.atan2(accel_y_raw, math.sqrt(accel_x_raw**2 + accel_z_raw**2))
        exp_smooth = 0.05
        self.pitch_deg = self.pitch_deg * (1 - exp_smooth) + math.degrees(pitch_rad) * exp_smooth
        self.roll_deg = self.roll_deg * (1 - exp_smooth) + math.degrees(roll_rad) * exp_smooth

    # ------------------------------------------------------------------
    # Gyro mode
    # ------------------------------------------------------------------

    def toggle_gyro_mode(self):
        self.gyro_mode = not self.gyro_mode
        if self.gyro_mode:
            self.light_bar_color = (0, 255, 0)
            self._set_light_bar(0, 255, 0)
            self.gyro_reference = {"pitch": self.pitch_deg, "roll": self.roll_deg}
            logging.info("Gyro control mode activated")
        else:
            self.light_bar_color = (0, 0, 255)
            self._set_light_bar(0, 0, 255)
            logging.info("Gyro control mode deactivated")

    # ------------------------------------------------------------------
    # Position update (IK + direct joint control)
    # ------------------------------------------------------------------

    def _update_positions(self, axes, buttons):
        speed = 0.2        # degrees (or normalized units) per update at 60 Hz = 12 deg/s for body joints
        gripper_speed = 0.5  # gripper uses 0-100 range, needs higher rate
        temp_positions = self.current_positions.copy()

        # O button: step gradually toward startup position
        if buttons.get("O") and self._startup_positions:
            step = 1.5  # degrees per frame toward startup
            for name in self.motor_names:
                if name in self._startup_positions:
                    delta = self._startup_positions[name] - temp_positions[name]
                    temp_positions[name] += max(-step, min(step, delta))
        else:
            # Gyro mode overrides right stick for wrist control
            if self.gyro_mode:
                delta_pitch = self.pitch_deg - self.gyro_reference["pitch"]
                delta_roll = self.roll_deg - self.gyro_reference["roll"]
                temp_positions["wrist_flex"] += delta_pitch * 0.5
                temp_positions["wrist_roll"] += delta_roll * 0.5
                self.gyro_reference = {"pitch": self.pitch_deg, "roll": self.roll_deg}

            # Left stick X → shoulder_pan (arm rotation left/right)
            temp_positions["shoulder_pan"] += axes["LX"] * speed

            # Left stick Y → shoulder_lift (forward/backward reach)
            temp_positions["shoulder_lift"] += axes["LY"] * speed

            # Right stick Y → elbow_flex (arm elevation/height)
            temp_positions["elbow_flex"] += axes["RY"] * speed

            # Right stick X → wrist_roll (negated to match physical direction)
            temp_positions["wrist_roll"] -= axes["RX"] * speed

            # D-pad up/down → wrist_flex (negated to match physical direction)
            temp_positions["wrist_flex"] -= (buttons["DPAD_UP"] - buttons["DPAD_DOWN"]) * speed

            # Triggers → gripper
            temp_positions["gripper"] -= gripper_speed * axes["R2"]  # close
            temp_positions["gripper"] += gripper_speed * axes["L2"]  # open

        self._clamp_to_limits(temp_positions)
        self.current_positions = temp_positions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_command(self):
        """Return current joint positions as {motor_name.pos: value} dict, or None if not yet initialized."""
        if not self._positions_initialized:
            return None
        with self.lock:
            return {f"{name}.pos": float(val) for name, val in self.current_positions.items()}

    def get_episode_end_status(self):
        with self.lock:
            status = self._episode_status
            self._episode_status = None
        return status

    def should_intervene(self):
        """Return True while L1+R1 are both held."""
        return bool(self.buttons.get("L1") and self.buttons.get("R1"))

    # ------------------------------------------------------------------
    # Feedback (light bar + rumble)
    # ------------------------------------------------------------------

    def indicate_error(self):
        self._set_light_bar(255, 0, 0, weak_rumble=128, strong_rumble=128)
        threading.Thread(target=self._reset_light_after_delay, args=(0.2,), daemon=True).start()

    def _reset_light_after_delay(self, delay):
        time.sleep(delay)
        self._set_light_bar(*self.light_bar_color)

    def _set_light_bar(self, red, green, blue, weak_rumble=0, strong_rumble=0):
        if not self.device:
            return
        try:
            if self.pid == 0x0CE6:
                # PS5 DualSense output report 0x02
                report = [0x02] + [0x00] * 47
                report[1] = 0xFF  # feature flags 1
                report[2] = 0xF7  # feature flags 2
                report[3] = 0x04  # light bar enable
                report[4] = weak_rumble
                report[5] = strong_rumble
                report[40] = red
                report[41] = green
                report[42] = blue
                self.device.write(bytes(report[:48]))
            else:
                # PS4 DualShock 4 output report 0x05
                report = [0x05, 0xFF, 0x00, 0x00, weak_rumble, strong_rumble, red, green, blue]
                report += [0x00] * 23
                self.device.write(bytes(report[:32]))
        except Exception as e:
            logging.error(f"Error sending output report: {e}")

    # ------------------------------------------------------------------
    # Safety / validation
    # ------------------------------------------------------------------

    def _clamp_to_limits(self, positions: dict) -> None:
        """Clamp all joint positions to their calibrated limits in-place."""
        if not self._joint_limits:
            return
        for motor, (lo, hi) in self._joint_limits.items():
            if motor in positions:
                positions[motor] = max(lo, min(hi, positions[motor]))

    # ------------------------------------------------------------------
    # Macros
    # ------------------------------------------------------------------

    def _execute_macro(self, button, positions):
        macros = {
            "O": self.initial_position[:6],     # home / initial position
            "T": [90, 130, 150, 70, 90, 80],   # top-down gripper
            "X": [90, 50, 130, -90, 90, 80],   # low horizontal gripper
            "S": [90, 160, 140, 20, 0, 0],      # looking forward
        }
        if button in macros:
            motor_positions = macros[button][:6]
            for name, pos in zip(self.motor_names, motor_positions, strict=False):
                positions[name] = pos
            logging.info(f"Macro '{button}': {motor_positions}")
        return positions

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def _compute_inverse_kinematics(self, x, y):
        """
        Compute shoulder_lift and elbow_flex from desired (x, y) endpoint in mm.

        A two-link planar arm has two solutions (elbow-up and elbow-down).
        We compute both and return the one closest to the current joint positions
        to prevent the IK from jumping between configurations.
        """
        l1, l2 = self.l1, self.l2
        distance = math.hypot(x, y)

        if distance > (l1 + l2) or distance < abs(l1 - l2):
            raise ValueError(f"Point ({x:.1f}, {y:.1f}) is out of reach.")

        offset = math.degrees(math.asin(32 / l1))

        cos_theta2 = max(-1.0, min(1.0, (l1**2 + l2**2 - distance**2) / (2 * l1 * l2)))
        theta2_deg = math.degrees(math.acos(cos_theta2))

        cos_theta1 = max(-1.0, min(1.0, (l1**2 + distance**2 - l2**2) / (2 * l1 * distance)))
        theta1_deg = math.degrees(math.acos(cos_theta1))

        alpha_deg = math.degrees(math.atan2(y, x))

        # Two solutions: motor2 = alpha ± theta1 + offset
        #                motor3 = 180 - theta2 + offset  (solution 1)
        #                motor3 = theta2 + offset - 180   (solution 2)
        sl1 = alpha_deg + theta1_deg + offset
        ef1 = 180.0 - theta2_deg + offset

        sl2 = alpha_deg - theta1_deg + offset
        ef2 = theta2_deg + offset - 180.0

        # Pick the solution closest to the current joint angles to avoid configuration jumps
        cur_sl = self.current_positions.get("shoulder_lift", sl1)
        cur_ef = self.current_positions.get("elbow_flex", ef1)

        dist1 = (sl1 - cur_sl) ** 2 + (ef1 - cur_ef) ** 2
        dist2 = (sl2 - cur_sl) ** 2 + (ef2 - cur_ef) ** 2

        return (sl1, ef1) if dist1 <= dist2 else (sl2, ef2)

    def _compute_position(self, motor2_angle, motor3_angle):
        """Compute (x, y) endpoint from shoulder_lift and elbow_flex angles."""
        l1, l2 = self.l1, self.l2
        offset = math.degrees(math.asin(32 / l1))

        beta_rad = math.radians(180 - motor2_angle + offset)
        theta2_rad = math.radians(180 - motor3_angle + offset)

        y = l1 * math.sin(beta_rad) - l2 * math.sin(beta_rad - theta2_rad)
        x = -l1 * math.cos(beta_rad) + l2 * math.cos(beta_rad - theta2_rad)
        return x, y
