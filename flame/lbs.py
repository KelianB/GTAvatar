# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

""" Adapted by Kelian Baert """

import torch
import torch.nn.functional as F

def flame_fn(
    template_verts, # (V,3) or (B,V,3)
    shape=None,
    expression=None,
    full_pose=None,
    eyelids=None,
    translation=None,
    lbs_weights=None,
    shapedirs_expr=None,
    shapedirs_id=None,
    posedirs=None,
    l_eyelid_dirs=None,
    r_eyelid_dirs=None,
    J_regressor=None,
    joint_centers=None,
    rot_parents=None,
    zero_centered_at_root_node=False, # otherwise, zero centered at the face
):
    """Perform whole forward skinning of FLAME: identity and expression blendshapes, pose correctives and LBS."""
    # full_pose: global (3), neck (3), jaw (3), eyes (6)

    assert not (shape is not None and shapedirs_id is None)    
    assert not (expression is not None and shapedirs_expr is None)    
    assert not (eyelids is not None and (l_eyelid_dirs is None or r_eyelid_dirs is None))
    assert posedirs is not None
    assert rot_parents is not None
    assert (template_verts.ndim == 2 or template_verts.ndim == 3) and template_verts.shape[-1] == 3
    assert (J_regressor is None or joint_centers is None) and not (J_regressor is not None and joint_centers is not None), "flame_fn: J_regressor and joint_centers are mutually exclusive."    

    batch_size = full_pose.shape[0]
    device = full_pose.device
    n_verts = template_verts.shape[-2]
    n_joints = lbs_weights.shape[1]
    assert J_regressor is None or (J_regressor.shape == (n_joints, n_verts)), f"flame_fn: wrong J_regressor shape: {J_regressor.shape} (expected {(n_joints, n_verts)})"

    transform_dict = {}

    if template_verts.ndim == 2:
        template_verts = template_verts.unsqueeze(0).expand(batch_size, template_verts.shape[0], 3)
    
    # Get the joints (5, n)
    J = None if joint_centers is None else joint_centers.repeat(batch_size, 1, 1)

    v_canonical = template_verts
    # Identity shapes
    if shape is not None:
        v_canonical = v_canonical + blend_shapes(shape, shapedirs_id)
    
    verts = v_canonical
    # Expression blendshapes
    if expression is not None:
        verts = verts + blend_shapes(expression, shapedirs_expr)
    # Pose blendshapes
    pose_feature, A = lbs_pose_only(full_pose, v_canonical, J_regressor, rot_parents, J=J)
    verts = verts + blend_shapes(pose_feature, posedirs)
    # Eyelids
    if eyelids is not None:
        assert l_eyelid_dirs is not None and r_eyelid_dirs is not None
        verts = verts + \
            r_eyelid_dirs.expand(batch_size, -1, -1) * eyelids[:, 1:2, None] +\
            l_eyelid_dirs.expand(batch_size, -1, -1) * eyelids[:, 0:1, None]
    # Centering
    if zero_centered_at_root_node:
        vertices = vertices - J[:, [0]]
        J = J - J[:, [0]]

    # Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    T = torch.matmul(W, A.view(batch_size, n_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, verts.shape[1], 1], dtype=verts.dtype, device=device)
    v_homo = torch.cat([verts, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    if translation is not None:
        verts = verts + translation.unsqueeze(1)
    transform_dict.update({"transform_matrix": T})   

    return verts, transform_dict

def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    """Calculates landmarks by barycentric interpolation

    Parameters
    ----------
    vertices: torch.tensor BxVx3, dtype = torch.float32
        The tensor of input vertices
    faces: torch.tensor Fx3, dtype = torch.long
        The faces of the mesh
    lmk_faces_idx: torch.tensor L, dtype = torch.long
        The tensor with the indices of the faces used to calculate the
        landmarks.
    lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
        The tensor of barycentric coordinates that are used to interpolate
        the landmarks

    Returns
    -------
    landmarks: torch.tensor BxLx3, dtype = torch.float32
        The coordinates of the landmarks for each mesh in the batch
    """
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(batch_size, -1, 3)
    lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)

    landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])
    return landmarks

def lbs_pose_only(
    pose,
    v_shaped,
    J_regressor,
    parents,
    pose2rot=True,
    J=None,
    dtype=torch.float32,
):
    """Calculate the pose features and transformations for linear blend skinning."""

    # This is taken from the lbs function below
    batch_size = pose.shape[0]
    device = pose.device
    # Get the joints
    # NxJx3 array
    if J is None:
        J = vertices2joints(J_regressor, v_shaped)
    # Compute pose features and rotation matrices
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.reshape(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view(batch_size, -1)
    else:
        pose_feature = (pose[:, 1:].reshape(batch_size, -1, 3, 3) - ident).view(batch_size, -1)
        rot_mats = pose.view(batch_size, -1, 3, 3)
    # Compute transformations
    _, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    return pose_feature, A


def vertices2joints(J_regressor, vertices):
    """Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    """

    return torch.einsum("bik,ji->bjk", [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    """Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum("bl,mkl->bmk", [betas, shape_disps])
    return blend_shape

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    """Calculates the rotation matrices for a batch of rotation vectors
    Parameters
    ----------
    rot_vecs: torch.tensor Nx3
        array of N axis-angle vectors
    Returns
    -------
    R: torch.tensor Nx3x3
        The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(
        (batch_size, 3, 3)
    )

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    """Creates a batch of transformation matrices
    Args:
        - R: Bx3x3 array of a batch of rotation matrices
        - t: Bx3x1 array of a batch of translation vectors
    Returns:
        - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone().contiguous()
    rel_joints[:, 1:] = rel_joints[:, 1:] - joints[:, parents[1:]]

    transforms_mat = transform_mat(rot_mats.view(-1, 3, 3), rel_joints.view(-1, 3, 1))
    transforms_mat = transforms_mat.view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms
