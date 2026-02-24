# !/usr/bin/env python

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

import sys
from enum import IntEnum
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadTeleopConfig




class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None

    @property
    def _is_ps5_mode(self) -> bool:
        return self.config.id == "ps5_controller"

    @property
    def action_features(self) -> dict:
        if self._is_ps5_mode:
            motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": {f"{m}.pos": i for i, m in enumerate(motor_names)},
            }
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        if self._is_ps5_mode:
            from .gamepad_utils import PS5JoystickController
            self.gamepad = PS5JoystickController()
            return

        # use HidApi for macos
        if sys.platform == "darwin":
            # NOTE: On macOS, pygame doesn’t reliably detect input from some controllers so we fall back to hidapi
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if self._is_ps5_mode:
            cmd = self.gamepad.get_command()
            if cmd is None:
                # Not yet initialized — read robot obs and initialize now
                raise RuntimeError(
                    "PS5 controller not initialized. Call send_feedback(robot.get_observation()) first."
                )
            return cmd

        # Update the controller to get fresh inputs
        self.gamepad.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.gamepad.get_deltas()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict = {
            "delta_x": gamepad_action[0],
            "delta_y": gamepad_action[1],
            "delta_z": gamepad_action[2],
        }

        # Default gripper action is to stay
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the gamepad such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        if self._is_ps5_mode:
            # PS5JoystickController updates positions in background thread;
            # episode status and intervention are tracked internally.
            is_intervention = self.gamepad.should_intervene()
            episode_end_status = self.gamepad.get_episode_end_status()
        else:
            # Update gamepad state to get fresh inputs
            self.gamepad.update()
            is_intervention = self.gamepad.should_intervene()
            episode_end_status = self.gamepad.get_episode_end_status()

        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    @property
    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        if self._is_ps5_mode and self.gamepad is not None:
            if not self.gamepad._positions_initialized:
                joint_limits = feedback.pop("__joint_limits__", None)
                self.gamepad.initialize_from_obs(feedback, joint_limits=joint_limits)
