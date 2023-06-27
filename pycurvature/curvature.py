from typing import Tuple

import numpy as np
import numpy.typing as npt


def gaussian_mean_curvatures(
    vertices: npt.NDArray[np.floating],
    triangles: npt.NDArray[np.integer],
    vertex_normals: npt.NDArray[np.floating],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    n = vertices.shape[0]
    gaussian_curvature = np.full(n, 2 * np.pi)
    area_mixed = np.zeros(n)
    e01v = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    e20v = vertices[triangles[:, 0]] - vertices[triangles[:, 2]]
    angle0 = _angle(e01v, -e20v)

    valid_triangles_mask = angle0 != 0.0
    triangles = triangles[valid_triangles_mask]
    angle0 = angle0[valid_triangles_mask]
    e01v = e01v[valid_triangles_mask]
    e20v = e20v[valid_triangles_mask]

    e12v = vertices[triangles[:, 2]] - vertices[triangles[:, 1]]
    angle1 = _angle(-e01v, e12v)
    angle2 = np.pi - angle0 - angle1
    angles = np.column_stack([angle0, angle1, angle2])
    cotan_angles = 1.0 / np.tan(angles)
    acute_mask = np.all(angles < 0.5 * np.pi, axis=-1)
    e01 = np.sum(e01v[acute_mask] ** 2, axis=-1)
    e12 = np.sum(e12v[acute_mask] ** 2, axis=-1)
    e20 = np.sum(e20v[acute_mask] ** 2, axis=-1)
    assert np.all(np.isfinite(cotan_angles))
    area0 = (
        e20 * cotan_angles[acute_mask, 1] + e01 * cotan_angles[acute_mask, 2]
    ) / 8.0
    area1 = (
        e01 * cotan_angles[acute_mask, 2] + e12 * cotan_angles[acute_mask, 0]
    ) / 8.0
    area2 = (
        e12 * cotan_angles[acute_mask, 0] + e20 * cotan_angles[acute_mask, 1]
    ) / 8.0
    np.add.at(area_mixed, triangles[acute_mask, 0], area0)
    np.add.at(area_mixed, triangles[acute_mask, 1], area1)
    np.add.at(area_mixed, triangles[acute_mask, 2], area2)
    obtuse_mask = np.logical_not(acute_mask)
    double_area = np.linalg.norm(
        np.cross(e01v[obtuse_mask], -e20v[obtuse_mask]), axis=-1, keepdims=True
    )
    coefficient = np.full_like(angles[obtuse_mask], 8.0)
    np.put_along_axis(
        coefficient,
        np.argmax(angles[obtuse_mask], axis=-1, keepdims=True),
        4.0,
        axis=-1,
    )
    np.add.at(area_mixed, triangles[obtuse_mask], double_area / coefficient)

    total_curvature = np.zeros_like(vertices)
    np.add.at(
        total_curvature,
        triangles[:, 0],
        (e20v * cotan_angles[:, [1]] - e01v * cotan_angles[:, [2]]) / 4.0,
    )
    np.add.at(
        total_curvature,
        triangles[:, 1],
        (e01v * cotan_angles[:, [2]] - e12v * cotan_angles[:, [0]]) / 4.0,
    )
    np.add.at(
        total_curvature,
        triangles[:, 2],
        (e12v * cotan_angles[:, [0]] - e20v * cotan_angles[:, [1]]) / 4.0,
    )

    np.subtract.at(gaussian_curvature, triangles, angles)

    wrapped_triangles = np.pad(triangles, ((0, 0), (0, 1)), mode="wrap")
    edges = np.lib.stride_tricks.sliding_window_view(wrapped_triangles, 2, axis=1)
    edges_list = np.reshape(np.sort(edges, axis=-1), (-1, 2))
    _, edge_inverse, edge_counts = np.unique(
        edges_list, return_inverse=True, return_counts=True, axis=0
    )
    border_indices = np.flatnonzero(edge_counts == 1)
    is_border = np.reshape(np.isin(edge_inverse, border_indices), (-1, 3))

    border_edges = edges[is_border]
    double_border_edges = np.row_stack([border_edges, np.fliplr(border_edges)])
    order = np.lexsort(np.rot90(double_border_edges))
    original_mask = order < border_edges.shape[0]
    double_border_edges = double_border_edges[order]
    start = double_border_edges[::2, 0]
    end1 = double_border_edges[::2, 1]
    end2 = double_border_edges[1::2, 1]
    e1 = vertices[end1] - vertices[start]
    e2 = vertices[end2] - vertices[start]

    consistent_border_edges = np.logical_xor(original_mask[::2], original_mask[1::2])
    e2[np.logical_not(consistent_border_edges)] *= -1
    gaussian_curvature[start] -= _angle(e1, e2)

    non_null_area = area_mixed > np.finfo(np.float64).eps
    gaussian_curvature = np.divide(
        gaussian_curvature,
        area_mixed,
        out=np.zeros_like(gaussian_curvature),
        where=non_null_area,
    )
    # sign seems a much better choice instead of where
    mean_curvature_sign = np.where(
        np.sum(total_curvature * vertex_normals, axis=-1) > 0, 1.0, -1.0
    )
    mean_curvature = mean_curvature_sign * np.divide(
        np.linalg.norm(total_curvature, axis=-1),
        area_mixed,
        out=np.zeros_like(gaussian_curvature),
        where=non_null_area,
    )
    return gaussian_curvature, mean_curvature


def _angle(
    u: npt.NDArray[np.floating], v: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    u_norm = np.linalg.norm(u, axis=-1)
    v_norm = np.linalg.norm(v, axis=-1)
    norm_product = u_norm * v_norm
    cos_angle: npt.NDArray[np.floating] = np.divide(
        np.sum(u * v, axis=-1),
        norm_product,
        out=np.ones_like(norm_product),
        where=norm_product > np.finfo(np.float64).eps,
    )
    cos_angle = np.clip(cos_angle, -1, 1)
    arccos: npt.NDArray[np.floating] = np.arccos(cos_angle)
    return arccos
