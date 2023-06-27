from typing import Tuple

import numpy as np
import numpy.typing as npt
import pymeshlab as pm
from pytest_cases import parametrize_with_cases

from pycurvature import gaussian_mean_curvatures


def meshlab_gaussian_mean_curvatures(
    vertices: npt.NDArray[np.floating],
    triangles: npt.NDArray[np.integer],
    vertex_normals: npt.NDArray[np.floating],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    mesh_set = pm.MeshSet()
    mesh_set.add_mesh(pm.Mesh(vertices, triangles, v_normals_matrix=vertex_normals))
    mesh = mesh_set.current_mesh()
    mesh_set.apply_filter(
        "compute_scalar_by_discrete_curvature_per_vertex", curvaturetype=0
    )
    mean_curvature = mesh.vertex_scalar_array()
    mesh_set.apply_filter(
        "compute_scalar_by_discrete_curvature_per_vertex", curvaturetype=1
    )
    gaussian_curvature = mesh.vertex_scalar_array()
    mesh_vertices = mesh.vertex_matrix()
    if mesh_vertices.shape[0] != vertices.shape[0]:
        n = vertices.shape[0]
        disconnected_vertex_indices = np.setdiff1d(
            np.arange(n), np.unique(triangles), assume_unique=True
        )
        gaussian_curvature = np.insert(
            gaussian_curvature, disconnected_vertex_indices, 0.0
        )
        mean_curvature = np.insert(mean_curvature, disconnected_vertex_indices, 0.0)
    return gaussian_curvature, mean_curvature


def case_plane_match() -> (
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.floating]]
):
    mesh_set = pm.MeshSet()
    mesh_set.apply_filter(
        "create_grid", numvertx=5, numverty=4, absscalex=0.5, absscaley=1.0
    )
    mesh = mesh_set.current_mesh()
    return mesh.vertex_matrix(), mesh.face_matrix(), mesh.vertex_normal_matrix()


def case_plane_with_disconnected_point_match() -> (
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.floating]]
):
    mesh_set = pm.MeshSet()
    mesh_set.apply_filter(
        "create_grid", numvertx=5, numverty=4, absscalex=0.5, absscaley=1.0
    )
    mesh = mesh_set.current_mesh()
    plane_vertices = mesh.vertex_matrix()
    plane_faces = mesh.face_matrix()
    plane_vertex_normals = mesh.vertex_normal_matrix()
    plane_vertices = np.append(plane_vertices, np.full((1, 3), -10), axis=0)
    plane_vertex_normals = np.append(plane_vertex_normals, np.ones((1, 3)), axis=0)
    return plane_vertices, plane_faces, plane_vertex_normals


def case_plane_with_inconsistent_orientation_match() -> (
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.floating]]
):
    mesh_set = pm.MeshSet()
    mesh_set.apply_filter(
        "create_grid", numvertx=4, numverty=4, absscalex=0.5, absscaley=1.0
    )
    mesh = mesh_set.current_mesh()
    plane_vertices = mesh.vertex_matrix()
    plane_faces = mesh.face_matrix()
    plane_faces[8:10, 1:] = np.fliplr(plane_faces[8:10, 1:])
    updated_mesh = pm.Mesh(plane_vertices, plane_faces)
    return plane_vertices, plane_faces, updated_mesh.vertex_normal_matrix()


def case_plane_with_non_manifold_vertex_match() -> (
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.floating]]
):
    mesh_set = pm.MeshSet()
    mesh_set.apply_filter(
        "create_grid", numvertx=5, numverty=4, absscalex=0.5, absscaley=1.0
    )
    mesh = mesh_set.current_mesh()
    plane_vertices = mesh.vertex_matrix()
    plane_faces = mesh.face_matrix()
    n = plane_vertices.shape[0]
    plane_vertices = np.append(plane_vertices, np.array([[0, 0, 1], [1, 0, 1]]), axis=0)
    plane_faces = np.append(plane_faces, np.array([[6, n, n + 1]]), axis=0)
    updated_mesh = pm.Mesh(plane_vertices, plane_faces)
    mesh_set = pm.MeshSet()
    mesh_set.add_mesh(updated_mesh)
    return plane_vertices, plane_faces, updated_mesh.vertex_normal_matrix()


def case_sphere_match() -> (
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.floating]]
):
    mesh_set = pm.MeshSet()
    mesh_set.apply_filter("create_sphere", radius=10.0, subdiv=5)
    mesh = mesh_set.current_mesh()
    return mesh.vertex_matrix(), mesh.face_matrix(), mesh.vertex_normal_matrix()


@parametrize_with_cases(
    "vertices, triangles, vertex_normals", cases=".", glob="*_match"
)
def test_gaussian_mean_curvature_match(
    vertices: npt.NDArray[np.floating],
    triangles: npt.NDArray[np.integer],
    vertex_normals: npt.NDArray[np.floating],
) -> None:
    gaussian_curvature, mean_curvature = gaussian_mean_curvatures(
        vertices, triangles, vertex_normals
    )

    (
        meshlab_gaussian_curvature,
        meshlab_mean_curvature,
    ) = meshlab_gaussian_mean_curvatures(vertices, triangles, vertex_normals)

    assert np.allclose(
        gaussian_curvature, meshlab_gaussian_curvature, atol=1e-5, rtol=1e-3
    )
    assert np.allclose(mean_curvature, meshlab_mean_curvature, atol=1e-5, rtol=1e-3)


def case_half_torus_sign() -> (
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.floating]]
):
    mesh_set = pm.MeshSet()
    mesh_set.apply_filter("create_torus", hsubdiv=48, vsubdiv=24)
    mesh = mesh_set.current_mesh()
    torus_vertices = mesh.vertex_matrix()
    torus_faces = mesh.face_matrix()
    torus_vertex_normals = mesh.vertex_normal_matrix()
    vertices_mask = torus_vertices[:, 2] >= 0.0
    half_torus_vertex_indices = np.flatnonzero(vertices_mask)
    faces_mask = np.all(np.isin(torus_faces, half_torus_vertex_indices), axis=-1)
    half_torus_vertices = torus_vertices[vertices_mask]
    _, half_torus_faces = np.unique(torus_faces[faces_mask], return_inverse=True)
    half_torus_faces = np.reshape(half_torus_faces, (-1, 3))
    half_torus_vertex_normals = torus_vertex_normals[vertices_mask]
    return half_torus_vertices, half_torus_faces, half_torus_vertex_normals


def case_plane_with_inconsistent_orientation_border_sign() -> (
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.floating]]
):
    mesh_set = pm.MeshSet()
    mesh_set.apply_filter(
        "create_grid", numvertx=6, numverty=5, absscalex=0.5, absscaley=1.0
    )
    mesh = mesh_set.current_mesh()
    plane_vertices = mesh.vertex_matrix()
    plane_faces = mesh.face_matrix()
    plane_faces[:2, 1:] = np.fliplr(plane_faces[:2, 1:])
    updated_mesh = pm.Mesh(plane_vertices, plane_faces)
    return plane_vertices, plane_faces, updated_mesh.vertex_normal_matrix()


@parametrize_with_cases("vertices, triangles, vertex_normals", cases=".", glob="*_sign")
def test_gaussian_mean_curvature_noisy_sign(
    vertices: npt.NDArray[np.floating],
    triangles: npt.NDArray[np.integer],
    vertex_normals: npt.NDArray[np.floating],
) -> None:
    gaussian_curvature, mean_curvature = gaussian_mean_curvatures(
        vertices, triangles, vertex_normals
    )

    (
        meshlab_gaussian_curvature,
        meshlab_mean_curvature,
    ) = meshlab_gaussian_mean_curvatures(vertices, triangles, vertex_normals)

    assert np.allclose(
        np.abs(gaussian_curvature),
        np.abs(meshlab_gaussian_curvature),
        atol=1e-5,
        rtol=1e-3,
    )
    assert (
        np.mean(
            np.isclose(
                gaussian_curvature, meshlab_gaussian_curvature, atol=1e-5, rtol=1e-3
            )
        )
        > 0.90
    )
    assert np.allclose(
        np.abs(mean_curvature), np.abs(meshlab_mean_curvature), atol=1e-5, rtol=1e-3
    )
    assert (
        np.mean(
            np.isclose(mean_curvature, meshlab_mean_curvature, atol=1e-5, rtol=1e-3)
        )
        > 0.0
    )


def case_plane_with_isolated_edge_exclude() -> (
    Tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.integer],
        npt.NDArray[np.floating],
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
    ]
):
    mesh_set = pm.MeshSet()
    mesh_set.apply_filter(
        "create_grid", numvertx=5, numverty=4, absscalex=0.5, absscaley=1.0
    )
    mesh = mesh_set.current_mesh()
    plane_vertices = mesh.vertex_matrix()
    plane_faces = mesh.face_matrix()
    plane_vertex_normals = mesh.vertex_normal_matrix()
    plane_vertices = np.append(plane_vertices, np.full((1, 3), -10), axis=0)
    spurious_triangle = np.array([0, 0, plane_vertices.shape[0] - 1])
    plane_faces = np.append(plane_faces, spurious_triangle[np.newaxis], axis=0)
    plane_vertex_normals = np.append(plane_vertex_normals, np.ones((1, 3)), axis=0)
    return (
        plane_vertices,
        plane_faces,
        plane_vertex_normals,
        np.array(0),
        np.array(plane_vertices.shape[0] - 1),
    )


def case_plane_with_edge_exclude() -> (
    Tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.integer],
        npt.NDArray[np.floating],
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
    ]
):
    mesh_set = pm.MeshSet()
    mesh_set.apply_filter(
        "create_grid", numvertx=5, numverty=4, absscalex=0.5, absscaley=1.0
    )
    mesh = mesh_set.current_mesh()
    plane_vertices = mesh.vertex_matrix()
    plane_faces = mesh.face_matrix()
    plane_vertex_normals = mesh.vertex_normal_matrix()
    spurious_triangle = np.array([0, 0, plane_vertices.shape[0] - 1])
    plane_faces = np.append(plane_faces, spurious_triangle[np.newaxis], axis=0)
    return (
        plane_vertices,
        plane_faces,
        plane_vertex_normals,
        np.unique(spurious_triangle),
        np.empty(0, dtype=np.int64),
    )


@parametrize_with_cases(
    "vertices, triangles, vertex_normals, exclude, zeros", cases=".", glob="*_exclude"
)
def test_gaussian_mean_curvature_exclude(
    vertices: npt.NDArray[np.floating],
    triangles: npt.NDArray[np.integer],
    vertex_normals: npt.NDArray[np.floating],
    exclude: npt.NDArray[np.integer],
    zeros: npt.NDArray[np.integer],
) -> None:
    gaussian_curvature, mean_curvature = gaussian_mean_curvatures(
        vertices, triangles, vertex_normals
    )

    (
        meshlab_gaussian_curvature,
        meshlab_mean_curvature,
    ) = meshlab_gaussian_mean_curvatures(vertices, triangles, vertex_normals)
    meshlab_gaussian_curvature[exclude] = gaussian_curvature[exclude]
    meshlab_mean_curvature[exclude] = mean_curvature[exclude]
    meshlab_gaussian_curvature[zeros] = 0.0
    meshlab_mean_curvature[zeros] = 0.0

    assert np.allclose(mean_curvature, meshlab_mean_curvature, atol=1e-5, rtol=1e-4)
    assert np.allclose(
        gaussian_curvature, meshlab_gaussian_curvature, atol=1e-5, rtol=1e-4
    )
