import numpy as np
from pydrake.math import eq, ge, le

from geometry.bezier import BezierCurve
from geometry.contact import CollisionPair, ContactModeType, PositionModeType, RigidBody
from planning.gcs import GcsContactPlanner, GcsPlanner
from planning.graph_builder import GraphBuilder, ModeConfig
from visualize.visualize import animate_positions, plot_positions_and_forces

# TODO remove
# flake8: noqa

# TODO add a guard that makes sure all bodies in all pairs have same dimension?


def plan_w_graph_builder():
    # Bezier curve params
    problem_dim = 2
    bezier_curve_order = 2

    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    finger_1 = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="f1",
        geometry="point",
        actuated=True,
    )
    finger_2 = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="f2",
        geometry="point",
        actuated=True,
    )
    box = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="b",
        geometry="box",
        width=box_width,
        height=box_height,
        actuated=False,
    )
    ground = RigidBody(
        dim=problem_dim,
        position_curve_order=bezier_curve_order,
        name="g",
        geometry="box",
        width=20,
        height=box_height,
        actuated=True,
    )
    rigid_bodies = [finger_1, finger_2, box, ground]

    x_f_1 = finger_1.pos_x
    y_f_1 = finger_1.pos_y
    x_f_2 = finger_2.pos_x
    y_f_2 = finger_2.pos_y
    x_b = box.pos_x
    y_b = box.pos_y
    x_g = ground.pos_x
    y_g = ground.pos_y

    # TODO these collision pairs will soon be generated automatically
    p1 = CollisionPair(
        finger_1,
        box,
        friction_coeff,
        position_mode=PositionModeType.LEFT,
    )
    p2 = CollisionPair(
        finger_2,
        box,
        friction_coeff,
        position_mode=PositionModeType.RIGHT,
    )
    p3 = CollisionPair(
        box,
        ground,
        friction_coeff,
        position_mode=PositionModeType.TOP,
    )
    p4 = CollisionPair(
        finger_1,
        ground,
        friction_coeff,
        position_mode=PositionModeType.TOP,
    )
    p5 = CollisionPair(
        finger_2,
        ground,
        friction_coeff,
        position_mode=PositionModeType.TOP,
    )
    collision_pairs = [p1, p2, p3, p4, p5]

    # Specify problem
    no_ground_motion = [eq(x_g, 0), eq(y_g, -1)]
    additional_constraints = [
        *no_ground_motion,
    ]
    source_config = ModeConfig(
        modes={
            p1.name: ContactModeType.NO_CONTACT,
            p2.name: ContactModeType.NO_CONTACT,
            p3.name: ContactModeType.ROLLING,
            p4.name: ContactModeType.NO_CONTACT,
            p5.name: ContactModeType.NO_CONTACT,
        },
        additional_constraints=[
            eq(x_f_1, 0),
            eq(y_f_1, 0.6),
            eq(x_f_2, 10.0),
            eq(y_f_2, 0.6),
            eq(x_b, 6.0),
            eq(y_b, box_height),
        ],
    )
    # TODO make it such that all non-specified modes are automatically not in contact
    target_config = ModeConfig(
        modes={
            p1.name: ContactModeType.ROLLING,
            p2.name: ContactModeType.ROLLING,
            p3.name: ContactModeType.NO_CONTACT,
            p4.name: ContactModeType.NO_CONTACT,
            p5.name: ContactModeType.NO_CONTACT,
        },
        additional_constraints=[eq(x_b, 10.0), eq(y_b, 4.0)],
    )

    # TODO:
    # Things to clean up:
    # - [ ] External forces
    # - [ ] Weights for costs
    # - [X] Unactuated bodies
    # - [ ] GCSContactPlanner should be removed and replaced
    # - [X] Rigid bodies collection
    # - [ ] Position variables, decision variables, force variables

    # - [ ] Position modes
    # - [ ] Specifying some mode constraints for source and target config (wait with this until I have fixed position modes too)
    # - [ ] Automatic collision_pair generation (wait with this until I have fixed position modes)

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    graph_builder = GraphBuilder(
        rigid_bodies,
        collision_pairs,
        external_forces,
        additional_constraints,
    )
    graph_builder.add_source_config(source_config)
    graph_builder.add_target_config(target_config)
    graph = graph_builder.build_graph("BFS")

    planner = GcsPlanner(graph, collision_pairs)
    planner.save_graph_diagram("pruned_graph.svg")
    planner.allow_revisits_to_vertices(1)
    planner.save_graph_diagram("pruned_graph_w_revisits.svg")

    # TODO add weights here
    planner.add_position_continuity_constraints()
    planner.add_position_path_length_cost()
    planner.add_force_path_length_cost()
    planner.add_num_visited_vertices_cost(100)
    planner.add_force_strength_cost()

    result = planner.solve()
    vertex_values = planner.get_vertex_values(result)

    normal_forces, friction_forces = planner.get_force_ctrl_points(vertex_values)
    positions = {
        body_name: planner.get_pos_ctrl_points(vertex_values, body_name)
        for body_name in planner.rigid_bodies_names
    }

    pos_curves = {
        body: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(c).eval_entire_interval()
                for c in ctrl_points
            ]
        )
        for body, ctrl_points in positions.items()
    }

    normal_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in normal_forces.items()
    }

    friction_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in friction_forces.items()
    }

    plot_positions_and_forces(pos_curves, normal_force_curves, friction_force_curves)
    animate_positions(pos_curves, rigid_bodies)
    return


def plan_for_two_fingers():
    # Bezier curve params
    dim = 2
    order = 2

    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    finger_1 = RigidBody(
        dim=dim, position_curve_order=order, name="finger_1", geometry="point"
    )
    finger_2 = RigidBody(
        dim=dim, position_curve_order=order, name="finger_2", geometry="point"
    )
    box = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="box",
        geometry="box",
        width=box_width,
        height=box_height,
    )
    ground = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="ground",
        geometry="box",
        width=100,
        height=1,
    )

    x_f_1 = finger_1.pos_x
    y_f_1 = finger_1.pos_y
    x_f_2 = finger_2.pos_x
    y_f_2 = finger_2.pos_y
    x_b = box.pos_x
    y_b = box.pos_y
    x_g = ground.pos_x
    y_g = ground.pos_y

    pair_finger_1_box = CollisionPair(
        finger_1,
        box,
        friction_coeff,
        position_mode=PositionModeType.LEFT,
    )
    pair_finger_2_box = CollisionPair(
        finger_2,
        box,
        friction_coeff,
        position_mode=PositionModeType.RIGHT,
    )
    pair_box_ground = CollisionPair(
        box, ground, friction_coeff, position_mode=PositionModeType.TOP
    )

    bodies = [finger_1, finger_2, box, ground]
    all_pairs = [
        pair_finger_1_box,
        pair_finger_2_box,
        pair_box_ground,
    ]

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    no_ground_motion = [eq(x_g, 0), eq(y_g, -1)]
    finger_1_pos_below_box_height = le(y_f_1, y_b + box_height)
    finger_1_pos_above_box_bottom = ge(y_f_1, y_b - box_height)
    finger_2_pos_below_box_height = le(y_f_2, y_b + box_height)
    finger_2_pos_above_box_bottom = ge(y_f_2, y_b - box_height)
    additional_constraints = [
        *no_ground_motion,
        finger_1_pos_below_box_height,
        finger_2_pos_below_box_height,
        finger_1_pos_above_box_bottom,
        finger_2_pos_above_box_bottom,
    ]

    source_constraints = [
        eq(x_f_1, 0),
        eq(y_f_1, 0.6),
        eq(x_f_2, 10.0),
        eq(y_f_2, 0.6),
        eq(x_b, 6.0),
        eq(y_b, box_height),
    ]
    target_constraints = [eq(x_b, 10.0), eq(y_b, 4.0)]

    planner = GcsContactPlanner(
        all_pairs,
        additional_constraints,
        external_forces,
        unactuated_bodies,
        allow_sliding=False,  # if set to True the problem will blow up!
    )

    planner.add_source(source_constraints)
    planner.add_target(target_constraints)

    planner.save_graph_diagram("graph_without_revisits.svg")
    print("Saved graph as diagram")
    planner.allow_revisits_to_vertices(1)  # TODO not optimal to call this here

    # TODO add weights here
    planner.add_position_continuity_constraints()
    planner.add_position_path_length_cost()
    planner.add_force_path_length_cost()
    planner.add_num_visited_vertices_cost(100)
    planner.add_force_strength_cost()

    result = planner.solve()
    vertex_values = planner.get_vertex_values(result)

    normal_forces, friction_forces = planner.get_force_ctrl_points(vertex_values)
    positions = {
        body: planner.get_pos_ctrl_points(vertex_values, body)
        for body in planner.all_bodies
    }

    pos_curves = {
        body: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(c).eval_entire_interval()
                for c in ctrl_points
            ]
        )
        for body, ctrl_points in positions.items()
    }

    normal_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in normal_forces.items()
    }

    friction_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in friction_forces.items()
    }

    plot_positions_and_forces(pos_curves, normal_force_curves, friction_force_curves)
    animate_positions(pos_curves, bodies)
    return


def plan_for_one_box_one_finger():
    # Bezier curve params
    dim = 2
    order = 2

    mass = 1  # kg
    g = 9.81  # m/s^2
    mg = mass * g
    box_width = 2
    box_height = 1
    friction_coeff = 0.5

    finger = RigidBody(
        dim=dim, position_curve_order=order, name="finger", geometry="point"
    )
    box = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="box",
        geometry="box",
        width=box_width,
        height=box_height,
    )
    ground = RigidBody(
        dim=dim,
        position_curve_order=order,
        name="ground",
        geometry="box",
        width=100,
        height=1,
    )

    bodies = [finger, box, ground]
    pair_finger_box = CollisionPair(
        finger,
        box,
        friction_coeff,
        position_mode=PositionModeType.LEFT,
    )
    pair_box_ground = CollisionPair(
        box, ground, friction_coeff, position_mode=PositionModeType.TOP
    )
    all_pairs = [pair_finger_box, pair_box_ground]

    # TODO this is very hardcoded
    gravitational_jacobian = np.array([[0, -1, 0, -1, 0, -1]]).T
    external_forces = gravitational_jacobian.dot(mg)

    unactuated_bodies = ["box"]

    x_f = finger.pos_x
    y_f = finger.pos_y
    x_b = box.pos_x
    y_b = box.pos_y
    x_g = ground.pos_x
    y_g = ground.pos_y

    no_ground_motion = [eq(x_g, 0), eq(y_g, -1)]
    additional_constraints = [
        *no_ground_motion,
        eq(pair_box_ground.lam_n, mg),
    ]

    source_constraints = [
        eq(x_f, 0),
        eq(y_f, 0.6),
        eq(x_b, 4.0),
        eq(y_b, box_height),
    ]
    target_constraints = [eq(x_f, 0.0), eq(x_b, 5)]

    planner = GcsContactPlanner(
        all_pairs,
        additional_constraints,
        external_forces,
        unactuated_bodies,
        allow_sliding=True,
    )

    planner.add_source(source_constraints)
    planner.add_target(target_constraints)

    planner.save_graph_diagram("graph_without_revisits.svg")
    print("Saved graph as diagram")
    planner.allow_revisits_to_vertices(1)  # TODO not optimal to call this here

    # TODO add weights here
    planner.add_position_continuity_constraints()
    planner.add_position_path_length_cost()
    planner.add_num_visited_vertices_cost(100)

    result = planner.solve()
    vertex_values = planner.get_vertex_values(result)

    normal_forces, friction_forces = planner.get_force_ctrl_points(vertex_values)
    positions = {
        body: planner.get_pos_ctrl_points(vertex_values, body)
        for body in planner.all_bodies
    }

    pos_curves = {
        body: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(c).eval_entire_interval()
                for c in ctrl_points
            ]
        )
        for body, ctrl_points in positions.items()
    }

    normal_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in normal_forces.items()
    }

    friction_force_curves = {
        pair: np.concatenate(
            [
                BezierCurve.create_from_ctrl_points(
                    points.reshape((1, -1))
                ).eval_entire_interval()
                for points in control_points
            ]
        )
        for pair, control_points in friction_forces.items()
    }

    plot_positions_and_forces(pos_curves, normal_force_curves, friction_force_curves)
    animate_positions(pos_curves, bodies)
    return
