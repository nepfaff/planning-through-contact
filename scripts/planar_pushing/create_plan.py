from pathlib import Path
from typing import Literal

import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    SliderPusherSystemConfig,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.planar import (
    visualize_planar_pushing_trajectory,
)


def get_slider_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.15, height=0.15)
    slider = RigidBody("box", box_geometry, mass)
    return slider


def get_tee() -> RigidBody:
    mass = 0.1
    body = RigidBody("t_pusher", TPusher2d(), mass)
    return body


def create_plan(
    debug: bool = False,
    body_to_use: Literal["box", "t_pusher"] = "box",
    traj_number: int = 1,
):
    if body_to_use == "box":
        body = get_slider_box()
    elif body_to_use == "t_pusher":
        body = get_tee()

    dynamics_config = SliderPusherSystemConfig(pusher_radius=0.04, slider=body)

    config = PlanarPlanConfig(
        time_non_collision=4.0,
        time_in_contact=4.0,
        num_knot_points_contact=3,
        num_knot_points_non_collision=4,
        avoid_object=True,
        avoidance_cost="quadratic",
        no_cycles=True,
        dynamics_config=dynamics_config,
    )

    planner = PlanarPushingPlanner(
        config,
    )

    if traj_number == 1:  # gives a kind of weird small touch
        slider_initial_pose = PlanarPose(x=-0.2, y=0.65, theta=0.0)
        slider_target_pose = PlanarPose(x=0.2, y=0.65, theta=-0.5)
        finger_initial_pose = PlanarPose(x=0.2, y=-0.2, theta=0.0)
        finger_target_pose = PlanarPose(x=0.0, y=-0.2, theta=0.0)
    elif traj_number == 2:
        slider_initial_pose = PlanarPose(x=0.2, y=0.65, theta=0.0)
        slider_target_pose = PlanarPose(x=-0.2, y=0.65, theta=0.5)
        finger_initial_pose = PlanarPose(x=-0.0, y=-0.2, theta=0.0)
        finger_target_pose = PlanarPose(x=0.2, y=-0.2, theta=0.0)
    elif traj_number == 3:
        slider_initial_pose = PlanarPose(x=0.0, y=0.65, theta=0.5)
        slider_target_pose = PlanarPose(x=-0.3, y=0.55, theta=-0.2)
        finger_initial_pose = PlanarPose(x=-0.0, y=-0.2, theta=0.0)
        finger_target_pose = PlanarPose(x=0.2, y=-0.2, theta=0.0)
    elif traj_number == 4:
        slider_initial_pose = PlanarPose(x=0.1, y=0.60, theta=-0.2)
        slider_target_pose = PlanarPose(x=-0.2, y=0.70, theta=0.5)
        finger_initial_pose = PlanarPose(x=-0.0, y=-0.2, theta=0.0)
        finger_target_pose = PlanarPose(x=0.2, y=-0.2, theta=0.0)
    elif traj_number == 5:
        slider_initial_pose = PlanarPose(x=0.55, y=0.0, theta=np.pi / 2)
        slider_target_pose = PlanarPose(x=0.80, y=0.0, theta=np.pi / 2)
        finger_initial_pose = PlanarPose(x=-0.2, y=0.15, theta=0.0)
        finger_target_pose = PlanarPose(x=-0.2, y=0.15, theta=0.0)
    elif traj_number == 6:
        slider_initial_pose = PlanarPose(x=0.55, y=0.0, theta=np.pi / 2)
        slider_target_pose = PlanarPose(x=0.80, y=0.0, theta=np.pi / 6)
        finger_initial_pose = PlanarPose(x=-0.2, y=0.15, theta=0.0)
        finger_target_pose = PlanarPose(x=-0.2, y=0.15, theta=0.0)
    else:
        raise NotImplementedError()

    planner.set_initial_poses(finger_initial_pose, slider_initial_pose)
    planner.set_target_poses(finger_target_pose, slider_target_pose)

    if debug:
        planner.save_graph_diagram(Path("graph.svg"))

    traj = planner.plan_trajectory(
        round_trajectory=True, print_output=debug, measure_time=debug, print_path=debug
    )
    traj_name = f"trajectories/{body_to_use}_pushing_{traj_number}.pkl"
    traj.save(traj_name)
    breakpoint()

    if debug:
        visualize_planar_pushing_trajectory(
            traj.to_old_format(),
            body.geometry,
            config.pusher_radius,
            visualize_robot_base=True,
        )


if __name__ == "__main__":
    create_plan(body_to_use="t_pusher", traj_number=5, debug=True)
