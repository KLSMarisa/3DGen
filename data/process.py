import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# root = '/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/glbs'
# root = '/mnt/nfs/caixiao/datasets/test/hf-objaverse-v1/glbs'
# cls_list = os.listdir(root)
# with open('/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/downloaded.txt', 'w') as f:
#     for cls in tqdm(cls_list):
#         cls_path = os.path.join(root, cls)
#         path_list = os.listdir(cls_path)
#
#         for path in path_list:
#             obj = os.path.join(cls_path, path)
#             if obj.endswith('.tmp'):
#                 pass
#             else:
#                 f.write(obj + '\n')

# with open('/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/test.txt', 'w') as f:
#     f.write('test' + '\n')
# counter = 0
# while True:
#     try:
#         obj_path = '/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/glbs/000-090/d5c5529e69a44dc69e534dd2c5e0e97e.glb'
#         command = (

#                     f"/home/caixiao/projects/3d_lib/objaverse/blender-3.2.2-linux-x64/blender -b -P /home/caixiao/projects/3DGen/data/blender.py -- "
#                     f" --object_path {obj_path} "
#                 )
#         os.system(command)
#         print(counter, obj_path,'test')
#         counter +=1

#     except Exception as e:
#         print(e)
#         idx = (idx + num_workers) % self.size
#         # signal.setitimer(signal.ITIMER_REAL, 0)
#         continue

#     if counter >=3:
#         break
Device = Union[str, torch.device]
_BatchFloatType = Union[float, Sequence[float], torch.Tensor]


def convert_to_tensors_and_broadcast(
        *args,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
):
    """
    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.

    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.

    Output:
        args: A list of tensors of shape (N, K_i)
    """
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    return args_Nd


def look_at_rotation(
        camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device: Device = "cpu"
) -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(
        camera_position, at, up, device=device
    )
    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)


def look_at_view_transform(
        dist: _BatchFloatType = 1.0,
        elev: _BatchFloatType = 0.0,
        azim: _BatchFloatType = 0.0,
        degrees: bool = True,
        eye: Optional[Union[Sequence, torch.Tensor]] = None,
        at=((0, 0, 0),),  # (1, 3)
        up=((0, 1, 0),),  # (1, 3)
        device: Device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns a rotation and translation matrix
    to apply the 'Look At' transformation from world -> view coordinates [0].

    Args:
        dist: distance of the camera from the object
        elev: angle in degrees or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        azim: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azim is the angle between the projected vector and a
            reference vector at (0, 0, 1) on the reference plane (the horizontal plane).
        dist, elev and azim can be of shape (1), (N).
        degrees: boolean flag to indicate if the elevation and azimuth
            angles are specified in degrees or radians.
        eye: the position of the camera(s) in world coordinates. If eye is not
            None, it will override the camera position derived from dist, elev, azim.
        up: the direction of the x axis in the world coordinate system.
        at: the position of the object(s) in world coordinates.
        eye, up and at can be of shape (1, 3) or (N, 3).

    Returns:
        2-element tuple containing

        - **R**: the rotation to apply to the points to align with the camera.
        - **T**: the translation to apply to the points to align with the camera.

    References:
    [0] https://www.scratchapixel.com
    """

    if eye is not None:
        broadcasted_args = convert_to_tensors_and_broadcast(eye, at, up, device=device)
        eye, at, up = broadcasted_args
        C = eye
    else:
        broadcasted_args = convert_to_tensors_and_broadcast(
            dist, elev, azim, at, up, device=device
        )
        dist, elev, azim, at, up = broadcasted_args
        C = (
                camera_position_from_spherical_angles(
                    dist, elev, azim, degrees=degrees, device=device
                )
                + at
        )

    R = look_at_rotation(C, at, up, device=device)
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    return R, T


def format_tensor(
        input,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
) -> torch.Tensor:
    """
    Helper function for converting a scalar value to a tensor.

    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: Device (as str or torch.device) on which the tensor should be placed.

    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    device_ = make_device(device)
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device_)

    if input.dim() == 0:
        input = input.view(1)

    if input.device == device_:
        return input

    input = input.to(device=device)
    return input


def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.

    Args:
        device: Device (as str or torch.device)

    Returns:
        A matching torch.device object
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def camera_position_from_spherical_angles(
        distance: float,
        elevation: float,
        azimuth: float,
        degrees: bool = True,
        device: Device = "cpu",
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.

    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.

    The vectors are broadcast against each other so they all have shape (N, 1).

    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    broadcasted_args = convert_to_tensors_and_broadcast(
        distance, elevation, azimuth, device=device
    )
    dist, elev, azim = broadcasted_args
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)  # add batch dim.
    return camera_position.view(-1, 3)



class Transform3d:
    """
    A Transform3d object encapsulates a batch of N 3D transformations, and knows
    how to transform points and normal vectors. Suppose that t is a Transform3d;
    then we can do the following:

    .. code-block:: python

        N = len(t)
        points = torch.randn(N, P, 3)
        normals = torch.randn(N, P, 3)
        points_transformed = t.transform_points(points)    # => (N, P, 3)
        normals_transformed = t.transform_normals(normals)  # => (N, P, 3)


    BROADCASTING
    Transform3d objects supports broadcasting. Suppose that t1 and tN are
    Transform3d objects with len(t1) == 1 and len(tN) == N respectively. Then we
    can broadcast transforms like this:

    .. code-block:: python

        t1.transform_points(torch.randn(P, 3))     # => (P, 3)
        t1.transform_points(torch.randn(1, P, 3))  # => (1, P, 3)
        t1.transform_points(torch.randn(M, P, 3))  # => (M, P, 3)
        tN.transform_points(torch.randn(P, 3))     # => (N, P, 3)
        tN.transform_points(torch.randn(1, P, 3))  # => (N, P, 3)


    COMBINING TRANSFORMS
    Transform3d objects can be combined in two ways: composing and stacking.
    Composing is function composition. Given Transform3d objects t1, t2, t3,
    the following all compute the same thing:

    .. code-block:: python

        y1 = t3.transform_points(t2.transform_points(t1.transform_points(x)))
        y2 = t1.compose(t2).compose(t3).transform_points(x)
        y3 = t1.compose(t2, t3).transform_points(x)


    Composing transforms should broadcast.

    .. code-block:: python

        if len(t1) == 1 and len(t2) == N, then len(t1.compose(t2)) == N.

    We can also stack a sequence of Transform3d objects, which represents
    composition along the batch dimension; then the following should compute the
    same thing.

    .. code-block:: python

        N, M = len(tN), len(tM)
        xN = torch.randn(N, P, 3)
        xM = torch.randn(M, P, 3)
        y1 = torch.cat([tN.transform_points(xN), tM.transform_points(xM)], dim=0)
        y2 = tN.stack(tM).transform_points(torch.cat([xN, xM], dim=0))

    BUILDING TRANSFORMS
    We provide convenience methods for easily building Transform3d objects
    as compositions of basic transforms.

    .. code-block:: python

        # Scale by 0.5, then translate by (1, 2, 3)
        t1 = Transform3d().scale(0.5).translate(1, 2, 3)

        # Scale each axis by a different amount, then translate, then scale
        t2 = Transform3d().scale(1, 3, 3).translate(2, 3, 1).scale(2.0)

        t3 = t1.compose(t2)
        tN = t1.stack(t3, t3)


    BACKPROP THROUGH TRANSFORMS
    When building transforms, we can also parameterize them by Torch tensors;
    in this case we can backprop through the construction and application of
    Transform objects, so they could be learned via gradient descent or
    predicted by a neural network.

    .. code-block:: python

        s1_params = torch.randn(N, requires_grad=True)
        t_params = torch.randn(N, 3, requires_grad=True)
        s2_params = torch.randn(N, 3, requires_grad=True)

        t = Transform3d().scale(s1_params).translate(t_params).scale(s2_params)
        x = torch.randn(N, 3)
        y = t.transform_points(x)
        loss = compute_loss(y)
        loss.backward()

        with torch.no_grad():
            s1_params -= lr * s1_params.grad
            t_params -= lr * t_params.grad
            s2_params -= lr * s2_params.grad

    CONVENTIONS
    We adopt a right-hand coordinate system, meaning that rotation about an axis
    with a positive angle results in a counter clockwise rotation.

    This class assumes that transformations are applied on inputs which
    are row vectors. The internal representation of the Nx4x4 transformation
    matrix is of the form:

    .. code-block:: python

        M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]

    To apply the transformation to points, which are row vectors, the latter are
    converted to homogeneous (4D) coordinates and right-multiplied by the M matrix:

    .. code-block:: python

        points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
        [transformed_points, 1] ∝ [points, 1] @ M

    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
        matrix: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.
            device: The device for storing the implemented transformation.
                If `matrix != None`, uses the device of input `matrix`.
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `device` and `dtype`.
        """

        if matrix is None:
            self._matrix = torch.eye(4, dtype=dtype, device=device).view(1, 4, 4)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError('"matrix" has to be a 2- or a 3-dimensional tensor.')
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError(
                    '"matrix" has to be a tensor of shape (minibatch, 4, 4) or (4, 4).'
                )
            # set dtype and device from matrix
            dtype = matrix.dtype
            device = matrix.device
            self._matrix = matrix.view(-1, 4, 4)

        self._transforms = []  # store transforms to compose
        self._lu = None
        self.device = make_device(device)
        self.dtype = dtype

    def __len__(self) -> int:
        return self.get_matrix().shape[0]

    def __getitem__(
        self, index: Union[int, List[int], slice, torch.BoolTensor, torch.LongTensor]
    ) -> "Transform3d":
        """
        Args:
            index: Specifying the index of the transform to retrieve.
                Can be an int, slice, list of ints, boolean, long tensor.
                Supports negative indices.

        Returns:
            Transform3d object with selected transforms. The tensors are not cloned.
        """
        if isinstance(index, int):
            index = [index]
        return self.__class__(matrix=self.get_matrix()[index])

    def compose(self, *others: "Transform3d") -> "Transform3d":
        """
        Return a new Transform3d representing the composition of self with the
        given other transforms, which will be stored as an internal list.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d with the stored transforms
        """
        out = Transform3d(dtype=self.dtype, device=self.device)
        out._matrix = self._matrix.clone()
        for other in others:
            if not isinstance(other, Transform3d):
                msg = "Only possible to compose Transform3d objects; got %s"
                raise ValueError(msg % type(other))
        out._transforms = self._transforms + list(others)
        return out

    def get_matrix(self) -> torch.Tensor:
        """
        Returns a 4×4 matrix corresponding to each transform in the batch.

        If the transform was composed from others, the matrix for the composite
        transform will be returned.
        For example, if self.transforms contains transforms t1, t2, and t3, and
        given a set of points x, the following should be true:

        .. code-block:: python

            y1 = t1.compose(t2, t3).transform(x)
            y2 = t3.transform(t2.transform(t1.transform(x)))
            y1.get_matrix() == y2.get_matrix()

        Where necessary, those transforms are broadcast against each other.

        Returns:
            A (N, 4, 4) batch of transformation matrices representing
                the stored transforms. See the class documentation for the conventions.
        """
        composed_matrix = self._matrix.clone()
        if len(self._transforms) > 0:
            for other in self._transforms:
                other_matrix = other.get_matrix()
                composed_matrix = _broadcast_bmm(composed_matrix, other_matrix)
        return composed_matrix

    def get_se3_log(self, eps: float = 1e-4, cos_bound: float = 1e-4) -> torch.Tensor:
        """
        Returns a 6D SE(3) log vector corresponding to each transform in the batch.

        In the SE(3) logarithmic representation SE(3) matrices are
        represented as 6-dimensional vectors `[log_translation | log_rotation]`,
        i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

        The conversion from the 4x4 SE(3) matrix `transform` to the
        6D representation `log_transform = [log_translation | log_rotation]`
        is done as follows::

            log_transform = log(transform.get_matrix())
            log_translation = log_transform[3, :3]
            log_rotation = inv_hat(log_transform[:3, :3])

        where `log` is the matrix logarithm
        and `inv_hat` is the inverse of the Hat operator [2].

        See the docstring for `se3.se3_log_map` and [1], Sec 9.4.2. for more
        detailed description.

        Args:
            eps: A threshold for clipping the squared norm of the rotation logarithm
                to avoid division by zero in the singular case.
            cos_bound: Clamps the cosine of the rotation angle to
                [-1 + cos_bound, 3 - cos_bound] to avoid non-finite outputs.
                The non-finite outputs can be caused by passing small rotation angles
                to the `acos` function in `so3_rotation_angle` of `so3_log_map`.

        Returns:
            A (N, 6) tensor, rows of which represent the individual transforms
            stored in the object as SE(3) logarithms.

        Raises:
            ValueError if the stored transform is not Euclidean (e.g. R is not a rotation
                matrix or the last column has non-zeros in the first three places).

        [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
        [2] https://en.wikipedia.org/wiki/Hat_operator
        """
        return se3_log_map(self.get_matrix(), eps, cos_bound)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse of self._matrix.
        """
        return torch.inverse(self._matrix)

    def inverse(self, invert_composed: bool = False) -> "Transform3d":
        """
        Returns a new Transform3d object that represents an inverse of the
        current transformation.

        Args:
            invert_composed:
                - True: First compose the list of stored transformations
                  and then apply inverse to the result. This is
                  potentially slower for classes of transformations
                  with inverses that can be computed efficiently
                  (e.g. rotations and translations).
                - False: Invert the individual stored transformations
                  independently without composing them.

        Returns:
            A new Transform3d object containing the inverse of the original
            transformation.
        """

        tinv = Transform3d(dtype=self.dtype, device=self.device)

        if invert_composed:
            # first compose then invert
            tinv._matrix = torch.inverse(self.get_matrix())
        else:
            # self._get_matrix_inverse() implements efficient inverse
            # of self._matrix
            i_matrix = self._get_matrix_inverse()

            # 2 cases:
            if len(self._transforms) > 0:
                # a) Either we have a non-empty list of transforms:
                # Here we take self._matrix and append its inverse at the
                # end of the reverted _transforms list. After composing
                # the transformations with get_matrix(), this correctly
                # right-multiplies by the inverse of self._matrix
                # at the end of the composition.
                tinv._transforms = [t.inverse() for t in reversed(self._transforms)]
                last = Transform3d(dtype=self.dtype, device=self.device)
                last._matrix = i_matrix
                tinv._transforms.append(last)
            else:
                # b) Or there are no stored transformations
                # we just set inverted matrix
                tinv._matrix = i_matrix

        return tinv

    def stack(self, *others: "Transform3d") -> "Transform3d":
        """
        Return a new batched Transform3d representing the batch elements from
        self and all the given other transforms all batched together.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d.
        """
        transforms = [self] + list(others)
        matrix = torch.cat([t.get_matrix() for t in transforms], dim=0)
        out = Transform3d(dtype=self.dtype, device=self.device)
        out._matrix = matrix
        return out

    def transform_points(self, points, eps: Optional[float] = None) -> torch.Tensor:
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: Tensor of shape (P, 3) or (N, P, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, P, 3) or (P, 3) depending
            on the dimensions of the transform
        """
        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))

        N, P, _3 = points_batch.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_batch = torch.cat([points_batch, ones], dim=2)

        composed_matrix = self.get_matrix()
        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
            denom = denom_sign * torch.clamp(denom.abs(), eps)
        points_out = points_out[..., :3] / denom

        # When transform is (1, 4, 4) and points is (P, 3) return
        # points_out of shape (P, 3)
        if points_out.shape[0] == 1 and points.dim() == 2:
            points_out = points_out.reshape(points.shape)

        return points_out

    def transform_normals(self, normals) -> torch.Tensor:
        """
        Use this transform to transform a set of normal vectors.

        Args:
            normals: Tensor of shape (P, 3) or (N, P, 3)

        Returns:
            normals_out: Tensor of shape (P, 3) or (N, P, 3) depending
            on the dimensions of the transform
        """
        if normals.dim() not in [2, 3]:
            msg = "Expected normals to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % (normals.shape,))
        composed_matrix = self.get_matrix()

        # TODO: inverse is bad! Solve a linear system instead
        mat = composed_matrix[:, :3, :3]
        normals_out = _broadcast_bmm(normals, mat.transpose(1, 2).inverse())

        # This doesn't pass unit tests. TODO investigate further
        # if self._lu is None:
        #     self._lu = self._matrix[:, :3, :3].transpose(1, 2).lu()
        # normals_out = normals.lu_solve(*self._lu)

        # When transform is (1, 4, 4) and normals is (P, 3) return
        # normals_out of shape (P, 3)
        if normals_out.shape[0] == 1 and normals.dim() == 2:
            normals_out = normals_out.reshape(normals.shape)

        return normals_out

    def translate(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            Translate(*args, device=self.device, dtype=self.dtype, **kwargs)
        )

    def scale(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            Scale(*args, device=self.device, dtype=self.dtype, **kwargs)
        )

    def rotate(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            Rotate(*args, device=self.device, dtype=self.dtype, **kwargs)
        )

    def rotate_axis_angle(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            RotateAxisAngle(*args, device=self.device, dtype=self.dtype, **kwargs)
        )

    def clone(self) -> "Transform3d":
        """
        Deep copy of Transforms object. All internal tensors are cloned
        individually.

        Returns:
            new Transforms object.
        """
        other = Transform3d(dtype=self.dtype, device=self.device)
        if self._lu is not None:
            other._lu = [elem.clone() for elem in self._lu]
        other._matrix = self._matrix.clone()
        other._transforms = [t.clone() for t in self._transforms]
        return other

    def to(
        self,
        device: Device,
        copy: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Transform3d":
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
          device: Device (as str or torch.device) for the new tensor.
          copy: Boolean indicator whether or not to clone self. Default False.
          dtype: If not None, casts the internal tensor variables
              to a given torch.dtype.

        Returns:
          Transform3d object.
        """
        device_ = make_device(device)
        dtype_ = self.dtype if dtype is None else dtype
        skip_to = self.device == device_ and self.dtype == dtype_

        if not copy and skip_to:
            return self

        other = self.clone()

        if skip_to:
            return other

        other.device = device_
        other.dtype = dtype_
        other._matrix = other._matrix.to(device=device_, dtype=dtype_)
        other._transforms = [
            t.to(device_, copy=copy, dtype=dtype_) for t in other._transforms
        ]
        return other

    def cpu(self) -> "Transform3d":
        return self.to("cpu")

    def cuda(self) -> "Transform3d":
        return self.to("cuda")


def compute_projection_matrix(
         znear, zfar, fov, aspect_ratio, degrees: bool, device: Device = "cpu"
    ) -> torch.Tensor:
        """
        Compute the calibration matrix K of shape (N, 4, 4)

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            fov: field of view angle of the camera.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            degrees: bool, set to True if fov is specified in degrees.

        Returns:
            torch.FloatTensor of the calibration matrix with shape (N, 4, 4)
        """
        K = torch.zeros((1, 4, 4), device=device, dtype=torch.float32)
        ones = torch.ones((1), dtype=torch.float32, device=device)
        if degrees:
            fov = (np.pi / 180) * fov

        if not torch.is_tensor(fov):
            fov = torch.tensor(fov, device=device)
        tanHalfFov = torch.tan((fov / 2))
        max_y = tanHalfFov * znear
        min_y = -max_y
        max_x = max_y * aspect_ratio
        min_x = -max_x

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space positive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0

        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        K[:, 0, 0] = 2.0 * znear / (max_x - min_x)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        K[:, 1, 1] = 2.0 * znear / (max_y - min_y)
        K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)
        K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)
        K[:, 3, 2] = z_sign * ones

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane.
        K[:, 2, 2] = z_sign * zfar / (zfar - znear)
        K[:, 2, 3] = -(zfar * znear) / (zfar - znear)


        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=device
        )

        return K

# 定义角度到弧度的转换函数
def deg_to_rad(deg):
    return deg * np.pi / 180



# 定义旋转矩阵
def rotation_matrix(azimuth, elevation, roll):
    # 将角度转换为弧度
    alpha = deg_to_rad(roll)
    beta = deg_to_rad(elevation)
    gamma = deg_to_rad(azimuth)

    # 计算旋转矩阵
    R_z  = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                     [np.sin(alpha),  np.cos(alpha), 0],
                     [0,              0,             1]])

    R_X = np.array([[1,            0,               0],
                    [0, np.cos(beta),  -np.sin(beta) ],
                    [0, np.sin(beta),   np.cos(beta) ]])

    R_y = np.array([[np.cos(gamma), 0, np.sin(gamma)],
                     [0,             1,             0],
                     [-np.sin(gamma),0, np.cos(gamma)]])

    # 将三个旋转矩阵相乘，得到相机矩阵R
    R = np.dot(R_z, np.dot(R_X, R_y))
    # R[:, [1, 2]] = R[:, [2, 1]]
    # # 三行-1
    # R[0, :] = R[0, :] * -1
    # R[2, :] = R[2, :] * -1
    # # 交换二三行
    # R[[0, 2], :] = R[[2, 0], :]
    # rotation_matrix = np.array([[1,  0,  0],
    #                         [0, -1,  0],
    #                         [0,  0, -1]])

    # # 将旋转矩阵应用到原始矩阵
    # R = np.dot(rotation_matrix, R)
    return R
# positon = camera_position_from_spherical_angles(elevation=0, azimuth=337.5, distance=2.246)
# # at=(-0.03969460725784302, 0.3856583535671234, -0.01171161979436866)
# print(positon )
# R, T = look_at_view_transform(elev=90, azim=0, dist=2.246, at=(-0.03969460725784302, 0.3856583535671234, -0.01171161979436866))
# # R, T = look_at_view_transform(eye=(-0.03969460725784302, 0.3856583535671234, 2.235137081365935), at=(-0.03969460725784302, 0.3856583535671234, -0.01171161979436866))
# print(R[0])
# print(T[0])
# RT = np.load('/mnt/nfs/caixiao/data/pv_views_64/00b360de1846428eb5c23464824c5fc8/pose/pose_48.npy')
# print(RT)
# [[ 1.         -0.         -0.         -0.03969461]
#  [ 0.          1.         -0.          0.38565835]
#  [ 0.          0.          1.          2.23513708]]
# [[ 1.         -0.         -0.          0.03969461]
#  [ 0.          1.         -0.         -0.38565835]
#  [ 0.          0.          1.         -2.23513708]]
# tensor([[-1.0000,  0.0000,  0.0000],
#         [ 0.0000,  0.9239, -0.3827],
#         [-0.0000, -0.3827, -0.9239]])
# tensor([-0.0397, -0.3608,  2.3828])
# tensor([[ 1.0000e+00,  0.0000e+00,  0.0000e+00],
#         [ 0.0000e+00,  1.0000e+00,  9.2883e-08],
#         [ 0.0000e+00, -9.2883e-08,  1.0000e+00]])
# tensor([ 0.0397, -0.3857,  2.2577])