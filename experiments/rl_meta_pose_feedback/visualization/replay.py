"""Export an offline, synchronized UmeTrack/robot motion viewer.

Default: replay saved evaluation states. --checkpoint recomputes deterministic
actions and real physics for the selected windows, without updating parameters.
All frames are the same step-end reference times used by the reward.
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from rl_meta_pose_feedback.training.sampling import EpisodeSampler
from kinematics.mesh import UmeTrackMesh
from rl_meta_pose_feedback.training.config import EXPERIMENT_DIR, TrainingConfig
from rl_meta_pose_feedback.training.runtime import create_runtime, write_json


# Common rigid display rotation: fingers (+semantic X) point up on screen.
DISPLAY_ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def packed(values, *, integer=False):
    values = np.asarray(values, dtype="<u4" if integer else "<f4")
    if not np.isfinite(values).all():
        raise ValueError("Viewer arrays must be finite")
    return {"type": "u32" if integer else "f32", "shape": list(values.shape),
            "data": base64.b64encode(values.tobytes(order="C")).decode("ascii")}


def display_points(points, origin, basis):
    """Re-express metre geometry in the frozen semantic palm basis, then rotate."""
    return ((np.asarray(points)-np.asarray(origin)) @ np.asarray(basis)) @ DISPLAY_ROTATION.T


def replay_initial_angles(trace, rollout):
    if "initial_joint_angles" in trace:
        return rollout.environment.hand.validate_angles(np.asarray(trace["initial_joint_angles"]))
    if rollout.config.episode_initialization == "retarget":
        raise ValueError("Aligned evaluation trace is missing initial_joint_angles")
    return np.asarray(rollout.config.initial_angles)


def validate_trace(episode, trace, rollout):
    replay_initial_angles(trace, rollout)
    steps = rollout.config.episode_steps
    required = {"actual_joint_angles", "simulation_time", "reference_query_time", "reference_sample_time", "costs"}
    if not required <= set(trace):
        raise ValueError("Evaluation trace is missing replay fields")
    if (trace["actual_joint_angles"].shape != (steps, 20) or trace["costs"].shape != (steps,)
            or any(not np.isfinite(trace[key]).all() for key in required)):
        raise ValueError("Evaluation trace has invalid shapes or nonfinite values")
    checks = ((trace["simulation_time"], (np.arange(steps)+1)*rollout.period, 1e-10),
              (trace["reference_query_time"], episode.query_times[1:].numpy(), 1e-6),
              (trace["reference_sample_time"], episode.sampled_times[1:].numpy(), 1e-6))
    for actual, expected, tolerance in checks:
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=tolerance):
            raise ValueError("Human reference times do not match the recorded robot trajectory")


def scene_meshes(hand):
    """Keep visual hand geoms; collision hulls, floor and other scene objects are hidden."""
    model = hand.model
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if root < 0:
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if root < 0:
        raise ValueError("Viewer needs a named hand or palm root")
    meshes, objects, lookup = [], [], {}
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body not in (0, root):
            body = int(model.body_parentid[body])
        if body != root or model.geom_group[geom] != 0 or model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        kind = int(model.geom_type[geom])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            key = (kind, mesh)
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            face, faces = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            vertices = model.mesh_vert[start:start+count].copy()
            triangles = model.mesh_face[face:face+faces].copy()
        else:
            import trimesh
            size = model.geom_size[geom]
            key = (kind, *size.tolist())
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                primitive = trimesh.creation.box(extents=2*size)
            elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                primitive = trimesh.creation.icosphere(subdivisions=2, radius=1.)
                primitive.vertices *= size[0] if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                primitive = trimesh.creation.cylinder(radius=size[0], height=2*size[1], sections=24)
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                primitive = trimesh.creation.capsule(radius=size[0], height=2*size[1], count=[12, 12])
                primitive.vertices -= primitive.bounds.mean(axis=0)
            else:
                raise ValueError(f"Unsupported visual geom type {kind}")
            vertices, triangles = np.asarray(primitive.vertices), np.asarray(primitive.faces)
        material = int(model.geom_matid[geom])
        color = model.mat_rgba[material].copy() if material >= 0 else model.geom_rgba[geom].copy()
        if color[3] <= 0:
            continue
        if key not in lookup:
            lookup[key] = len(meshes)
            corners = np.array(list(itertools.product(*zip(vertices.min(0), vertices.max(0)))))
            meshes.append({"vertices": vertices, "triangles": triangles, "corners": corners})
        objects.append({"geom": geom, "mesh": lookup[key], "color": color.tolist()})
    if not objects:
        raise ValueError("No visible hand geometry found")
    return meshes, objects


def robot_poses(rollout, joint_angles, meshes, objects):
    """Forward only the recorded actual positions; never clip them to target limits."""
    hand, landmarks = rollout.environment.hand, rollout.landmarks
    model, data = hand.model, mujoco.MjData(hand.model)
    active = hand.active_mask
    basis = landmarks.palm_rest_frame(reference_finger=rollout.palm_finger)
    matrices, markers, lows, highs = [], [], [], []
    for angles in joint_angles:
        data.qpos[hand.qpos_ids[active]] = angles[active]
        mujoco.mj_forward(model, data)
        native_from_world = data.xmat[landmarks.frame_id].reshape(3, 3).T
        rotation = DISPLAY_ROTATION @ basis.T @ native_from_world
        kind, index, offset = landmarks.points[0]
        origin = (data.site_xpos[index] if kind == "site" else
                  data.xpos[index]+data.xmat[index].reshape(3, 3) @ offset)
        poses, tips = [], []
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for obj in objects:
            geom = obj["geom"]
            transform = np.eye(4)
            transform[:3, :3] = rotation @ data.geom_xmat[geom].reshape(3, 3)
            transform[:3, 3] = rotation @ (data.geom_xpos[geom]-origin)
            poses.append(transform.T.reshape(16))  # WebGL column-major matrices.
            corners = meshes[obj["mesh"]]["corners"] @ transform[:3, :3].T+transform[:3, 3]
            lo, hi = np.minimum(lo, corners.min(0)), np.maximum(hi, corners.max(0))
        for point_kind, point_id, point_offset in landmarks.points:
            point = (data.site_xpos[point_id] if point_kind == "site" else
                     data.xpos[point_id]+data.xmat[point_id].reshape(3, 3) @ point_offset)
            tips.append(rotation @ (point-origin))
        matrices.append(poses)
        markers.append(tips)
        lows.append(lo)
        highs.append(hi)
    return np.asarray(matrices), np.asarray(markers), [np.min(lows, 0).tolist(), np.max(highs, 0).tolist()]


@torch.no_grad()
def export_replay(evaluation: Path, output: Path, episode_ids, *, checkpoint_path=None, device="cpu"):
    evaluation = Path(evaluation).resolve()
    metrics_path = evaluation / "metrics.json" if evaluation.is_dir() else evaluation
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (not episode_ids or len(set(episode_ids)) != len(episode_ids)
            or any(type(i) is not int or not 0 <= i < len(metrics["window_indices"]) for i in episode_ids)):
        raise ValueError("Episode IDs must be distinct indices into the selected evaluation")
    if output.exists():
        raise FileExistsError("Refusing to overwrite replay output")
    recompute = checkpoint_path is not None
    controller = metrics.get("controller", "decoder")
    if recompute and controller == "reference_retarget":
        raise ValueError("Reference-controller replay requires saved trajectories; cannot recompute with a decoder")
    checkpoint_path = Path(metrics["checkpoint"] if checkpoint_path is None else checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not recompute and checkpoint["update"] != metrics["update"]:
        raise ValueError("Cached evaluation and checkpoint updates differ; supply --checkpoint to recompute")
    from dataclasses import replace
    config = replace(TrainingConfig.from_dict(checkpoint["config"]), device=device)
    actor, _, preprocessor, rollout = create_runtime(config, checkpoint)
    source = EpisodeSampler.from_config(config, actor.encoder, rollout, metrics["split"], initialize=recompute)
    episodes, selected, _ = source.draw(len(episode_ids),
        selected_indices=[metrics["window_indices"][i] for i in episode_ids])
    metadata = {i.recording.path.name: i.recording.metadata for i in source.dataset.indexes}
    traces = []
    if recompute:
        if metrics["emg_control"] != "matched" or metrics["zero_actions"]:
            raise ValueError("Recompute mode expects a matched-input evaluation selection")
        actions, _ = rollout.actions(actor, preprocessor, episodes, stochastic=False)
        traces = [rollout.execute(episode, action).trace for episode, action in zip(episodes, actions)]
    else:
        for index in episode_ids:
            with np.load(metrics_path.parent / f"episode_{index:03d}.npz", allow_pickle=False) as saved:
                traces.append({key: saved[key].copy() for key in saved.files})
    human = UmeTrackMesh(side="right").float().eval()
    basis = rollout.metric.human_palm_frame.numpy()
    reference = rollout.metric.human_reference_offset.numpy()
    meshes, objects = scene_meshes(rollout.environment.hand)
    payload = {"schemaVersion": 1, "split": metrics["split"], "checkpointUpdate": checkpoint["update"],
               "controller": controller,
               "costLabel": {"tracking": "跟踪代价", "retarget": "总目标代价",
                             "legacy_vector": "局部向量代价"}[config.reward_objective],
               "emgControl": metrics["emg_control"], "zeroActions": metrics["zero_actions"],
               "group": config.group, "humanScale": rollout.group.human_scale,
               "humanProvenance": human.provenance,
               "robotMeshes": [{"vertices": packed(m["vertices"]), "triangles": packed(m["triangles"], integer=True)} for m in meshes],
               "robotObjects": [{"mesh": obj["mesh"], "color": obj["color"]} for obj in objects],
               "humanTriangles": packed(human.faces.numpy(), integer=True), "clips": []}
    report = {"schema_version": 1, "evaluation": str(metrics_path), "checkpoint": str(checkpoint_path),
              "controller": controller,
              "checkpoint_role": "configuration_only" if controller == "reference_retarget" else "policy",
               "reward_objective": rollout.reward_settings,
              "checkpoint_update": checkpoint["update"], "source": "recomputed_mean_policy" if recompute else "saved_evaluation_trace",
              "human_mesh_provenance": human.provenance, "human_side": "right", "units": "metres",
              "frame_alignment": "initial state then identical step-end references and actual robot joint states",
              "display_coordinates": "frozen semantic palm basis, same rigid screen rotation on both sides",
              "default_camera_scale": "independent fit; equal metres per pixel toggle available",
              "clips": []}
    for index, window_index, episode, trace in zip(episode_ids, selected, episodes, traces):
        window = metrics["windows"][index]
        if any(episode.sample[key] != window[key] for key in ("filename", "start", "stop", "segment_id")):
            raise ValueError("Evaluation window layout differs from this checkpoint's data configuration")
        validate_trace(episode, trace, rollout)
        human_vertices = display_points(human(episode.reference_angles).numpy(), reference, basis)*rollout.group.human_scale
        points = display_points(episode.human_points.numpy(), reference, basis)*rollout.group.human_scale
        human_markers = np.concatenate((np.zeros((len(points), 1, 3)), points[:, rollout.metric.human_indices[1:].numpy()]), axis=1)
        angles = np.vstack((replay_initial_angles(trace, rollout), trace["actual_joint_angles"]))
        transforms, robot_markers, bounds = robot_poses(rollout, angles, meshes, objects)
        frames = len(human_vertices)
        if frames != len(transforms):
            raise ValueError("Human and robot frame counts differ")
        costs = [None]+trace["costs"].tolist()
        label = metadata[window["filename"]].get("stage", "")
        payload["clips"].append({"id": index, "stage": label, "filename": window["filename"],
            "sourceHandSide": metadata[window["filename"]]["side"], "displayHandSide": "right",
            "times": [0.]+trace["simulation_time"].tolist(), "costs": costs,
            "humanVertices": packed(human_vertices), "humanMarkers": packed(human_markers),
            "robotTransforms": packed(transforms), "robotMarkers": packed(robot_markers),
            "humanBounds": [human_vertices.min((0, 1)).tolist(), human_vertices.max((0, 1)).tolist()],
            "robotBounds": bounds})
        report["clips"].append({"episode_id": index, "window_index": window_index, **window, "stage": label,
            "source_hand_side": metadata[window["filename"]]["side"], "display_hand_side": "right",
            "frames": frames, "duration_seconds": float(trace["simulation_time"][-1]),
            "mean_cost": float(trace["costs"].mean()),
            "mean_geometric_cost": float(trace.get("geometric_costs", trace["costs"]).mean()),
            "mean_regularization_cost": float(trace.get("regularization_costs", np.zeros_like(trace["costs"])).mean()),
            "initial_joint_angles": angles[0].tolist(), "episode_initialization": config.episode_initialization,
            "human_reference_query_times": episode.query_times.tolist(), "human_sample_times": episode.sampled_times.tolist(),
            "human_frame_shape": list(human_vertices.shape), "robot_transform_shape": list(transforms.shape)})
    template = Path(__file__).with_name("replay.html").read_text(encoding="utf-8")
    if template.count("__REPLAY_DATA__") != 1:
        raise ValueError("Viewer template must contain one data placeholder")
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    output.mkdir(parents=True)
    (output / "replay.html").write_text(template.replace("__REPLAY_DATA__", data), encoding="utf-8")
    write_json(output / "replay_info.json", report)
    print(json.dumps({"html": str((output / "replay.html").resolve()), "episodes": episode_ids,
                      "frames_per_clip": [c["frames"] for c in report["clips"]], "source": report["source"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True, help="Evaluation directory or metrics.json")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0], help="0-based positions in the evaluation window list")
    parser.add_argument("--checkpoint", type=Path, help="Recompute the selected EMG windows using this checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the standalone offline HTML")
    args = parser.parse_args()
    export_replay(args.evaluation, args.output.resolve(), args.episodes, checkpoint_path=args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
