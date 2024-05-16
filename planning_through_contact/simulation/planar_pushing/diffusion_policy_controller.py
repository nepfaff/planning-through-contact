import logging
import pathlib
import time as pytime
from collections import deque
from typing import List, Optional, Tuple

import cv2
import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np

# Diffusion Policy imports
import torch
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Pydrake imports
from pydrake.common.value import AbstractValue, Value
from pydrake.math import RigidTransform
from pydrake.systems.framework import Context, LeafSystem
from pydrake.systems.sensors import Image, PixelType

from planning_through_contact.geometry.planar.planar_pose import PlanarPose

logger = logging.getLogger(__name__)

# Set the print precision to 4 decimal places
np.set_printoptions(precision=4)


class DiffusionPolicyController(LeafSystem):
    def __init__(
        self,
        checkpoint: str,
        initial_pusher_pose: PlanarPose,
        target_slider_pose: PlanarPose,
        diffusion_policy_path: str = "/home/adam/workspace/gcs-diffusion",
        freq: float = 10.0,
        delay: float = 1.0,
        device="cuda:0",
        debug: bool = False,
        cfg_overrides: dict = {},
    ):
        super().__init__()
        self._checkpoint = pathlib.Path(checkpoint)
        self._diffusion_policy_path = pathlib.Path(diffusion_policy_path)
        self._initial_pusher_pose = initial_pusher_pose
        self._target_slider_pose = target_slider_pose
        self._freq = freq
        self._dt = 1.0 / freq
        self._delay = delay
        self._debug = debug
        self._device = torch.device(device)
        self._load_policy_from_checkpoint(self._checkpoint)
        # Override diffusion policy config
        for key, value in cfg_overrides.items():
            self._cfg[key] = value

        # get parameters
        self._obs_horizon = self._cfg.n_obs_steps
        self._action_steps = self._cfg.n_action_steps
        self._state_dim = self._cfg.shape_meta.obs.agent_pos.shape[0]
        self._action_dim = self._cfg.shape_meta.action.shape[0]
        self._target_dim = self._cfg.policy.target_dim
        self._num_image_channels = self._cfg.shape_meta.obs.image.shape[0]
        self._image_height = self._cfg.shape_meta.obs.image.shape[1]
        self._image_width = self._cfg.shape_meta.obs.image.shape[2]
        self._B = 1  # batch size is 1

        # indexing parameters for action predictions
        self._start = self._obs_horizon - 1
        self._start += 1
        if "push_tee_v2" in checkpoint:  # for backwards compatibility
            print("Using push_tee_v2 slicing for action predictions")
            self._start += 1
        self._end = self._start + self._action_steps

        # observation histories
        self._pusher_pose_deque = deque(
            [self._initial_pusher_pose.vector() for _ in range(self._obs_horizon)],
            maxlen=self._obs_horizon,
        )
        self._image_deque = deque([], maxlen=self._obs_horizon)

        # variables for DoCalcOutput
        self._actions = deque([], maxlen=self._action_steps)
        self._current_action = np.array(
            [
                self._initial_pusher_pose.x,
                self._initial_pusher_pose.y,
            ]
        )

        # Input port for pusher pose
        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )
        self.camera_port = self.DeclareAbstractInputPort(
            "camera",
            Value[Image[PixelType.kRgba8U]].Make(
                Image[PixelType.kRgba8U](self._image_width, self._image_height)
            ),
        )

        self.output = self.DeclareVectorOutputPort(
            "planar_position_command", 2, self.DoCalcOutput
        )

    def _load_policy_from_checkpoint(self, checkpoint: str):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        self._cfg = payload["cfg"]
        # self._cfg.training.device = self._device
        cls = hydra.utils.get_class(self._cfg._target_)
        workspace: BaseWorkspace
        workspace = cls(self._cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get normalizer: this might be expensive for larger datasets
        zarr_configs = self._cfg.task.dataset.zarr_configs
        for config in zarr_configs:
            config["path"] = self._diffusion_policy_path.joinpath(config["path"])
        dataset: BaseImageDataset = hydra.utils.instantiate(self._cfg.task.dataset)
        self._normalizer = dataset.get_normalizer()

        # get policy from workspace
        self._policy = workspace.model
        self._policy.set_normalizer(self._normalizer)
        if self._cfg.training.use_ema:
            self._policy = workspace.ema_model
            self._policy.set_normalizer(self._normalizer)
        self._policy.to(self._device)
        self._policy.eval()

    def DoCalcOutput(self, context: Context, output):
        time = context.get_time()

        # Continually update ports until delay is over
        if time < self._delay:
            self._update_history(context)
            output.set_value(self._current_action)
            return
        # Accumulate new observations after reset
        if (
            len(self._pusher_pose_deque) < self._obs_horizon
            or len(self._image_deque) < self._obs_horizon
        ):
            self._update_history(context)
            output.set_value(self._current_action)
            return

        # Update observation history
        self._update_history(context)

        obs_dict = self._deque_to_dict(
            self._pusher_pose_deque,
            self._image_deque,
            # self._target_slider_pose.vector()
            self._initial_pusher_pose.vector(),  # Doing this because of bug TODO: fix this
        )

        if len(self._actions) == 0:
            # Compute new actions
            start_time = pytime.time()
            with torch.no_grad():
                action_prediction = self._policy.predict_action(obs_dict)[
                    "action_pred"
                ][0]
            actions = action_prediction[self._start : self._end]
            for action in actions:
                self._actions.append(action.cpu().numpy())

            if self._debug:
                print(
                    f"[TIME: {time:.3f}] Computed new actions in {pytime.time() - start_time:.3f}s\n"
                )
                print("Observations:")
                for state in self._pusher_pose_deque:
                    print(state)
                for img in self._image_deque:
                    plt.imshow(img)
                    plt.show()
                print("\nAction Predictions:")
                print(action_prediction)
                print("\nActions")
                print(actions)

        # get next action
        assert len(self._actions) > 0
        prev_action = self._current_action
        self._current_action = self._actions.popleft()
        output.set_value(self._current_action)

        # debug print statements
        if self._debug:
            print(
                f"Time: {time:.3f}, action delta: {np.linalg.norm(self._current_action - prev_action)}"
            )
            print(f"Time: {time:.3f}, action: {self._current_action}")

    def reset(self, reset_position: np.ndarray):
        self._current_action = reset_position
        self._actions.clear()
        self._pusher_pose_deque.clear()
        self._image_deque.clear()

    def _deque_to_dict(self, obs_deque: deque, img_deque: deque, target: np.ndarray):
        state_tensor = torch.cat(
            [torch.from_numpy(obs) for obs in obs_deque], dim=0
        ).reshape(self._B, self._obs_horizon, self._state_dim)
        img_tensor = torch.cat(
            [torch.from_numpy(np.moveaxis(img, -1, -3) / 255.0) for img in img_deque],
            dim=0,
        ).reshape(
            self._B,
            self._obs_horizon,
            self._num_image_channels,
            self._image_width,
            self._image_height,
        )
        target_tensor = torch.from_numpy(target).reshape(1, self._target_dim)  # 1, D_t
        return {
            "obs": {
                "image": img_tensor.to(self._device),  # 1, T_obs, C, H, W
                "agent_pos": state_tensor.to(self._device),  # 1, T_obs, D_x
            },
            "target": target_tensor.to(self._device),  # 1, D_t
        }

    def _update_history(self, context):
        """Update state and image observation history"""
        pusher_pose: RigidTransform = self.pusher_pose_measured.Eval(context)  # type: ignore
        image = self.camera_port.Eval(context)
        pusher_planer_pose = PlanarPose.from_pose(pusher_pose).vector()
        self._pusher_pose_deque.append(pusher_planer_pose)
        if image.shape[0] != self._image_height or image.shape[1] != self._image_width:
            image = cv2.resize(
                image.data[:, :, :-1], (self._image_height, self._image_width)
            )
            self._image_deque.append(image)
        else:
            self._image_deque.append(image.data[:, :, :-1])
