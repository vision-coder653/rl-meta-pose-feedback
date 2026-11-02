"""Automatic reference bindings for the modular HOD generation grammar.

Presence comes from the actual fingertip visual geom, not empty bodies, ghost
joint counts, filenames or a requested training group. Module-local rest joint
coordinates retain the original CAD direction even when upstream compensation
translations move a removed middle hinge close to the end hinge.
"""
from __future__ import annotations

import mujoco
import numpy as np

from simulation.morphology import HandMorphology
from .reference_points import fingertip_mesh_reference, palm_mesh_reference
from .robot import FingerFrameSpec, PointSpec


def hod_fingertip_references(hand: HandMorphology) -> dict[int, PointSpec]:
    """Return one distal-surface point for each existing f1..f5 finger.

    Supports HOD's retained ghost hinges and arbitrary grammar lengths. The
    pinned generator keeps middle/end joint coordinates in parallel module CAD
    frames (P2/P3 in generate_rand_gram_joint_finger.py). Subtract these intrinsic
    coordinates, not world anchors: grammar and servo-removal translations must
    not tilt or reverse the distal module's longitudinal direction.

    Unknown/missing module geometry raises; generic manually authored robots
    continue to use explicit PointSpec/FingerFrameSpec bindings.
    """
    model = hand.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = {}
    for finger in range(1, 6):
        prefix = f"f{finger}_finger_"
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + "tip_vis_geom")
        if geom < 0:
            if hand.active_mask[(finger-1)*4:finger*4].any():
                raise ValueError(f"Active HOD finger {finger} has no fingertip visual mesh")
            continue  # A missing HOD finger retains empty bodies and ghost joints.
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "tip")
        if body < 0 or model.geom_bodyid[geom] != body:
            raise ValueError(f"HOD finger {finger} tip mesh is attached to an unexpected body")
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + role + "_joint")
                  for role in ("middle", "end")]
        if any(joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
               for joint in joints):
            raise ValueError(f"HOD finger {finger} needs its middle/end module hinges, including ghost placeholders")
        middle, end = joints
        tip_from_world = data.xmat[body].reshape(3, 3).T
        tip_from_middle = tip_from_world @ data.xmat[model.jnt_bodyid[middle]].reshape(3, 3)
        tip_from_end = tip_from_world @ data.xmat[model.jnt_bodyid[end]].reshape(3, 3)
        if not np.allclose(tip_from_middle, tip_from_end, atol=1e-7, rtol=0):
            raise ValueError(f"HOD finger {finger} module CAD axes are not parallel; supply explicit references")
        longitudinal = tip_from_end @ (model.jnt_pos[end] - model.jnt_pos[middle])
        flexion = tip_from_end @ model.jnt_axis[end]
        point = fingertip_mesh_reference(model, body=prefix + "tip", geom=prefix + "tip_vis_geom",
                                         longitudinal_axis=longitudinal, flexion_axis=flexion)
        # Check that the selected mesh really belongs to the distal rigid link.
        FingerFrameSpec.from_joint(hand, point, joint=prefix + "end_joint")
        result[finger] = point
    if not result:
        raise ValueError("No HOD fingertip visual meshes found")
    return result


def hod_landmark_config(hand: HandMorphology) -> dict:
    """Build the existing RobotLandmarks JSON schema without hand-count groups."""
    tips = hod_fingertip_references(hand)
    palm = palm_mesh_reference(hand.model, body="palm", geom="palm_vis_geom")
    return {
        "palm_frame": "palm",
        "palm": {"body": palm.body, "offset_m": list(palm.offset_m)},
        "fingertips": {str(i): {"body": point.body, "offset_m": list(point.offset_m)}
                       for i, point in tips.items()},
        "fingertip_frames": {str(i): {"joint": f"f{i}_finger_end_joint"} for i in tips},
    }


def add_hod_reference_sites(spec: mujoco.MjSpec, config: dict) -> None:
    """Add visible, nonphysical sites to an in-memory model before compiling.

    The JSON's body offsets remain authoritative for reward bindings. Adding a
    site changes no joint, actuator, mass or contact geometry. Identical saved
    sites are accepted; a conflicting marker is never silently overwritten.
    """
    points = [("palm_reference", config["palm"], (.2, .9, 1., 1.))]
    points += [(f"f{i}_fingertip_reference", value, (1., .2, .15, 1.))
               for i, value in config["fingertips"].items()]
    bodies = {body.name: body for body in spec.bodies}
    sites = {site.name: (body.name, site) for body in spec.bodies for site in body.sites}
    pending = []
    for name, value, color in points:
        point = PointSpec(**value)
        if point.body not in bodies:
            raise ValueError(f"Reference marker {name} needs an existing body-local point")
        if name in sites:
            owner, site = sites[name]
            if owner != point.body or not np.allclose(site.pos, point.offset_m, atol=1e-9, rtol=0):
                raise ValueError(f"Existing reference marker conflicts with selected point: {name}")
        else:
            pending.append((bodies[point.body], name, point.offset_m, color))
    # Validate every binding before mutating the spec.
    for body, name, position, color in pending:
        body.add_site(name=name, pos=position, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[.002, 0, 0], rgba=color, group=0)


def mark_hod_references(spec: mujoco.MjSpec) -> dict:
    """Automation entry: select this candidate's points and attach visible sites.

    Compile the modified spec for subsequent simulation. The returned JSON-ready
    bindings also work with the original unmarked model. No file is overwritten.
    """
    config = hod_landmark_config(HandMorphology.from_model(spec.compile()))
    add_hod_reference_sites(spec, config)
    return config
