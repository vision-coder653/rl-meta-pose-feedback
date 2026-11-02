"""Select fixed palm and distal-surface references from model geometry.

Human: UmeTrack landmark 20 at zero angles, frozen in the wrist frame.
Robot: uniform-volume centroid of an explicitly named closed palm mesh,
expressed in the palm body frame. Never fit a point to a motion trajectory.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from .human import UmeTrackFK
from .frames import finger_frame
from .robot import PointSpec, _rotation


def human_rest_palm_offset(*, side: str) -> tuple[float, float, float]:
    """Return a constant wrist-local point in metres; do not sample it per frame."""
    with torch.no_grad():
        point = UmeTrackFK(side=side)(torch.zeros(20, dtype=torch.float64))[20]
    return tuple(point.tolist())


def fingertip_mesh_reference(model: mujoco.MjModel, *, body: str, geom: str,
                            longitudinal_axis, flexion_axis) -> PointSpec:
    """Select the most distal surface point in the finger's width midplane.

    Axes are explicit body-local directions: longitudinal points distally and
    flexion defines finger width. Intersect the actual triangle mesh with the
    plane halfway across that width, then maximize longitudinal position. A
    flat distal segment uses its midpoint. This selects the tip rather than a
    lateral corner, finger-pad centre, mesh centre, or hinge/body origin.

    The result is a fixed body-local offset, valid at every joint pose. Only
    the named geom is considered; no collision hull or other link is used.
    Disconnected equally distal surfaces require an explicit PointSpec.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if body_id < 0 or geom_id < 0:
        raise ValueError("Fingertip reference needs an existing body and named mesh geom")
    if (model.geom_bodyid[geom_id] != body_id
            or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH):
        raise ValueError("Fingertip reference geom must be a mesh attached directly to its body")
    basis = finger_frame(longitudinal_axis, flexion_axis)
    forward, width = basis[:, 0], basis[:, 1]
    mesh = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start:start + count].astype(np.float64)
    vertices = vertices @ _rotation(model.geom_quat[geom_id]).T + model.geom_pos[geom_id]
    start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
    faces = model.mesh_face[start:start + count]
    # Numerical tolerance follows the precision of compiled float32 vertices;
    # it is not a band that changes the physical definition of the tip.
    tolerance = max(float(np.ptp(vertices, axis=0).max()) * 1e-7, 1e-12)
    lateral = vertices @ width
    midplane = .5 * (lateral.min() + lateral.max())
    distances = lateral - midplane
    sections = []
    for face in faces:
        points = [vertices[index] for index in face if abs(distances[index]) <= tolerance]
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if ((distances[a] < -tolerance and distances[b] > tolerance)
                    or (distances[b] < -tolerance and distances[a] > tolerance)):
                fraction = distances[a] / (distances[a] - distances[b])
                points.append(vertices[a] + fraction * (vertices[b] - vertices[a]))
        if points:
            sections.append(np.stack(points))
    if not sections:
        raise ValueError("Fingertip width midplane does not intersect the mesh")
    farthest = max(float((section @ forward).max()) for section in sections)
    distal = [section[(section @ forward) >= farthest - tolerance] for section in sections]
    distal = [points for points in distal if len(points)]
    # Retain triangle connectivity: averaging separated peaks can put a point
    # inside the mesh or in empty space. For a flat cap its cut is one segment.
    reached = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for other in range(len(distal)):
            if other not in reached and np.any(np.linalg.norm(
                    distal[current][:, None] - distal[other][None], axis=-1) <= 2*tolerance):
                reached.add(other)
                pending.append(other)
    if len(reached) != len(distal):
        raise ValueError("Disconnected distal peaks need an explicit fingertip PointSpec")
    points = np.concatenate(distal)
    depth = points @ basis[:, 2]
    point = .5 * (points[depth.argmin()] + points[depth.argmax()])
    return PointSpec(body=body, offset_m=tuple(point.tolist()))


def palm_mesh_reference(model: mujoco.MjModel, *, body: str, geom: str) -> PointSpec:
    """Select the solid palm's geometric centroid, excluding fingers and motors.

    The geometry must belong directly to the specified palm body and contain
    one closed orientable triangle surface. Face winding is made consistent in
    a temporary copy, including HOD palms whose side faces face inward. Invalid
    geometry raises rather than falling back to another origin or bounding box.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if body_id < 0 or geom_id < 0:
        raise ValueError("Palm reference needs an existing body and named mesh geom")
    if (model.geom_bodyid[geom_id] != body_id
            or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH):
        raise ValueError("Palm reference geom must be a mesh attached directly to the palm body")
    mesh = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start:start + count].astype(np.float64)
    start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
    faces = model.mesh_face[start:start + count]
    edge_faces = {}
    for index, face in enumerate(faces):
        if len(set(face.tolist())) != 3:
            raise ValueError("Palm mesh has a degenerate face")
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault((min(a, b), max(a, b)), []).append((index, 1 if a < b else -1))
    if not edge_faces or any(len(owners) != 2 for owners in edge_faces.values()):
        raise ValueError("Palm mesh must be closed with two faces per edge")
    neighbors = [[] for _ in faces]
    for (a, direction_a), (b, direction_b) in edge_faces.values():
        relation = -direction_a * direction_b  # Shared edges must run in opposite directions.
        neighbors[a].append((b, relation))
        neighbors[b].append((a, relation))
    winding = np.zeros(len(faces), dtype=np.int8)
    winding[0] = 1
    pending = [0]
    while pending:
        current = pending.pop()
        for other, relation in neighbors[current]:
            desired = winding[current] * relation
            if winding[other] == 0:
                winding[other] = desired
                pending.append(other)
            elif winding[other] != desired:
                raise ValueError("Palm mesh is not orientable")
    if np.any(winding == 0):
        raise ValueError("Palm mesh must contain one connected closed surface")
    triangles = vertices[faces].copy()
    triangles[winding < 0] = triangles[winding < 0][:, [0, 2, 1]]
    # Use a nearby tetrahedron origin to avoid cancellation for translated meshes.
    origin = vertices.mean(axis=0)
    relative = triangles - origin
    volume6 = np.einsum("ij,ij->i", relative[:, 0], np.cross(relative[:, 1], relative[:, 2]))
    total = volume6.sum()
    extent = np.ptp(vertices, axis=0).max()
    if not np.isfinite(total) or abs(total) <= np.finfo(float).eps * extent**3:
        raise ValueError("Palm mesh has no finite nonzero enclosed volume")
    centroid = origin + (relative.sum(axis=1) * volume6[:, None]).sum(axis=0) / (4 * total)
    offset = _rotation(model.geom_quat[geom_id]) @ centroid + model.geom_pos[geom_id]
    return PointSpec(body=body, offset_m=tuple(offset.tolist()))
