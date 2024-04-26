# python3.7
"""Contains the functions to sample points in 3D space."""

import numpy as np

import torch
import torch.nn.functional as F

__all__ = [
    'PointSampler'
]

_POINT_SAMPLING_STRATEGIES = [
    'uniform', 'normal', 'ray_dependent', 'point_dependent'
]

_POINT_PERTURBING_STRATEGIES = [
    'no', 'middle_uniform', 'uniform', 'self_uniform'
]

_TENSOR_SAMPLING_STRATEGIES = [
    'fix', 'uniform', 'normal', 'hybrid', 'truncated_normal'
]


class PointSampler(torch.nn.Module):
    """Defines the class to help sample points.

    This class implements the `forward()` function for point sampling, which
    includes the following steps:

    1. Sample rays in the camera coordinate system.
    2. Sample points on each ray.
    3. Perturb points on each ray.
    4. Sample camera extrinsics.
    5. Transform points to the world coordinate system.
    """

    def __init__(self,
                 num_points=16,
                 fov=30,
                 image_boundary_value=1.0,
                 cam_look_at_dir=-1,
                 pixel_center=False,
                 y_descending=True,
                 # Point sampling (i.e., radial distance w.r.t. camera) related.
                 sampling_strategy='uniform',
                 focal=None,
                 dis_min=None,
                 dis_max=None,
                 dis_mean=None,
                 dis_stddev=None,
                 per_ray_ref=None,
                 per_point_ref=None,
                 perturbation_strategy='middle_uniform',
                 # Camera sampling related.
                 radius_strategy='fix',
                 radius_fix=None,
                 radius_min=None,
                 radius_max=None,
                 radius_mean=None,
                 radius_stddev=None,
                 polar_strategy='uniform',
                 polar_fix=None,
                 polar_min=None,
                 polar_max=None,
                 polar_mean=None,
                 polar_stddev=None,
                 azimuthal_strategy='uniform',
                 azimuthal_fix=None,
                 azimuthal_min=None,
                 azimuthal_max=None,
                 azimuthal_mean=None,
                 azimuthal_stddev=None,
                 use_spherical_uniform_position=False,
                 pitch_strategy='fix',
                 pitch_fix=0,
                 pitch_min=None,
                 pitch_max=None,
                 pitch_mean=None,
                 pitch_stddev=None,
                 yaw_strategy='fix',
                 yaw_fix=0,
                 yaw_min=None,
                 yaw_max=None,
                 yaw_mean=None,
                 yaw_stddev=None,
                 roll_strategy='fix',
                 roll_fix=0,
                 roll_min=None,
                 roll_max=None,
                 roll_mean=None,
                 roll_stddev=None):
        """Initializes hyper-parameters for point sampling.

        Detailed description of each argument can be found in functions
        `get_ray_per_pixel()`, `sample_points_per_ray()`,
        `perturb_points_per_ray()`, `sample_camera_extrinsics()`.
        """
        super().__init__()
        self.num_points = num_points
        self.fov = fov
        self.image_boundary_value = image_boundary_value
        self.cam_look_at_dir = cam_look_at_dir
        self.pixel_center = pixel_center
        self.y_descending = y_descending

        self.sampling_strategy = sampling_strategy
        self.dis_min = dis_min
        self.dis_max = dis_max
        self.dis_mean = dis_mean
        self.dis_stddev = dis_stddev
        self.per_ray_ref = per_ray_ref
        self.per_point_ref = per_point_ref
        self.perturbation_strategy = perturbation_strategy

        self.radius_strategy = radius_strategy
        self.radius_fix = radius_fix
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.radius_mean = radius_mean
        self.radius_stddev = radius_stddev
        self.polar_strategy = polar_strategy
        self.polar_fix = polar_fix
        self.polar_min = polar_min
        self.polar_max = polar_max
        self.polar_mean = polar_mean
        self.polar_stddev = polar_stddev
        self.azimuthal_strategy = azimuthal_strategy
        self.azimuthal_fix = azimuthal_fix
        self.azimuthal_min = azimuthal_min
        self.azimuthal_max = azimuthal_max
        self.azimuthal_mean = azimuthal_mean
        self.azimuthal_stddev = azimuthal_stddev
        self.use_spherical_uniform_position = use_spherical_uniform_position
        self.pitch_strategy = pitch_strategy
        self.pitch_fix = pitch_fix
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.pitch_mean = pitch_mean
        self.pitch_stddev = pitch_stddev
        self.yaw_strategy = yaw_strategy
        self.yaw_fix = yaw_fix
        self.yaw_min = yaw_min
        self.yaw_max = yaw_max
        self.yaw_mean = yaw_mean
        self.yaw_stddev = yaw_stddev
        self.roll_strategy = roll_strategy
        self.roll_fix = roll_fix
        self.roll_min = roll_min
        self.roll_max = roll_max
        self.roll_mean = roll_mean
        self.roll_stddev = roll_stddev
        self.focal = focal

    def forward(self,
                batch_size,
                image_size,
                focal=None,
                cam2world_matrix=None,
                **kwargs):
        """Samples points.

        `K` denotes the number of points on each ray.

        Args:
            batch_size: Batch size of images. Denoted as `N`.
            image_size: Size of the image. One element indicates square image,
                while two elements stand for height and width respectively.
                Denoted as `H` and `W`.
            **kwargs: Additional keyword arguments to override the variables
                initialized in `__init__()`.

        Returns:
            A dictionary, containing
                - `camera_radius`: camera radius w.r.t. the world coordinate
                    system, with shape [N].
                - `camera_polar`: camera polar w.r.t. the world coordinate
                    system, with shape [N].
                - `camera_azimuthal`: camera azimuthal w.r.t. the world
                    coordinate system, with shape [N].
                - `camera_pitch`: camera pitch w.r.t. the camera coordinate
                    system, with shape [N].
                - `camera_yaw`: camera yaw w.r.t. the camera coordinate system,
                    with shape [N].
                - `camera_roll`: camera roll w.r.t. the camera coordinate
                    system, with shape [N].
                - `camera_pos`: camera position, i.e., the (x, y, z) coordinate
                    in the world coordinate system, with shape [N, 3].
                - `cam2world_matrix`: transformation matrix to transform the
                    camera coordinate system to the world coordinate system,
                    with shape [N, 4, 4].
                - `rays_camera`: ray directions in the camera coordinate system,
                    with shape [N, H, W, 3].
                - `rays_world`: ray directions in the world coordinate system,
                    with shape [N, H, W, 3].
                - `radii_raw`: raw per-point radial distance w.r.t. the camera
                    position, with shape [N, H, W, K].
                - `radii`: per-point radial distance after perturbation w.r.t.
                    the camera position, with shape [N, H, W, K].
                - `points_camera`: per-point coordinate in the camera coordinate
                    system, with shape [N, H, W, K, 3].
                - `points_world`: per-point coordinate in the world coordinate
                    system, with shape [N, H, W, K, 3].
        """
        num_points = kwargs.get('num_points', self.num_points)
        fov = kwargs.get('fov', self.fov)
        image_boundary_value = kwargs.get(
            'image_boundary_value', self.image_boundary_value)
        cam_look_at_dir = kwargs.get('cam_look_at_dir', self.cam_look_at_dir)
        pixel_center = kwargs.get('pixel_center', self.pixel_center)
        y_descending = kwargs.get('y_descending', self.y_descending)
        sampling_strategy = kwargs.get(
            'sampling_strategy', self.sampling_strategy)
        dis_min = kwargs.get('dis_min', self.dis_min)
        dis_max = kwargs.get('dis_max', self.dis_max)
        dis_mean = kwargs.get('dis_mean', self.dis_mean)
        dis_stddev = kwargs.get('dis_stddev', self.dis_stddev)
        per_ray_ref = kwargs.get('per_ray_ref', self.per_ray_ref)
        per_point_ref = kwargs.get('per_point_ref', self.per_point_ref)
        perturbation_strategy = kwargs.get(
            'perturbation_strategy', self.perturbation_strategy)
        radius_strategy = kwargs.get('radius_strategy', self.radius_strategy)
        radius_fix = kwargs.get('radius_fix', self.radius_fix)
        radius_min = kwargs.get('radius_min', self.radius_min)
        radius_max = kwargs.get('radius_max', self.radius_max)
        radius_mean = kwargs.get('radius_mean', self.radius_mean)
        radius_stddev = kwargs.get('radius_stddev', self.radius_stddev)
        polar_strategy = kwargs.get('polar_strategy', self.polar_strategy)
        polar_fix = kwargs.get('polar_fix', self.polar_fix)
        polar_min = kwargs.get('polar_min', self.polar_min)
        polar_max = kwargs.get('polar_max', self.polar_max)
        polar_mean = kwargs.get('polar_mean', self.polar_mean)
        polar_stddev = kwargs.get('polar_stddev', self.polar_stddev)
        azimuthal_strategy = kwargs.get(
            'azimuthal_strategy', self.azimuthal_strategy)
        azimuthal_fix = kwargs.get('azimuthal_fix', self.azimuthal_fix)
        azimuthal_min = kwargs.get('azimuthal_min', self.azimuthal_min)
        azimuthal_max = kwargs.get('azimuthal_max', self.azimuthal_max)
        azimuthal_mean = kwargs.get('azimuthal_mean', self.azimuthal_mean)
        azimuthal_stddev = kwargs.get('azimuthal_stddev', self.azimuthal_stddev)
        use_spherical_uniform_position = kwargs.get(
            'use_spherical_uniform_position',
            self.use_spherical_uniform_position)
        pitch_strategy = kwargs.get('pitch_strategy', self.pitch_strategy)
        pitch_fix = kwargs.get('pitch_fix', self.pitch_fix)
        pitch_min = kwargs.get('pitch_min', self.pitch_min)
        pitch_max = kwargs.get('pitch_max', self.pitch_max)
        pitch_mean = kwargs.get('pitch_mean', self.pitch_mean)
        pitch_stddev = kwargs.get('pitch_stddev', self.pitch_stddev)
        yaw_strategy = kwargs.get('yaw_strategy', self.yaw_strategy)
        yaw_fix = kwargs.get('yaw_fix', self.yaw_fix)
        yaw_min = kwargs.get('yaw_min', self.yaw_min)
        yaw_max = kwargs.get('yaw_max', self.yaw_max)
        yaw_mean = kwargs.get('yaw_mean', self.yaw_mean)
        yaw_stddev = kwargs.get('yaw_stddev', self.yaw_stddev)
        roll_strategy = kwargs.get('roll_strategy', self.roll_strategy)
        roll_fix = kwargs.get('roll_fix', self.roll_fix)
        roll_min = kwargs.get('roll_min', self.roll_min)
        roll_max = kwargs.get('roll_max', self.roll_max)
        roll_mean = kwargs.get('roll_mean', self.roll_mean)
        roll_stddev = kwargs.get('roll_stddev', self.roll_stddev)

        rays_camera = get_ray_per_pixel(batch_size=batch_size,
                                        image_size=image_size,
                                        fov=fov,
                                        boundary=image_boundary_value,
                                        focal=focal,
                                        cam_look_at_dir=cam_look_at_dir,
                                        pixel_center=pixel_center,
                                        y_descending=y_descending)

        radii_raw = sample_points_per_ray(batch_size=batch_size,
                                          image_size=image_size,
                                          num_points=num_points,
                                          strategy=sampling_strategy,
                                          dis_min=dis_min,
                                          dis_max=dis_max,
                                          dis_mean=dis_mean,
                                          dis_stddev=dis_stddev,
                                          per_ray_ref=per_ray_ref,
                                          per_point_ref=per_point_ref)
        radii = perturb_points_per_ray(radii=radii_raw,
                                       strategy=perturbation_strategy)

        camera_info = {}
        if cam2world_matrix is not None:
            camera_info.update(dict(
                cam2world_matrix=cam2world_matrix,
                radius=None,
                polar=None,
                azimuthal=None,
                pitch=None,
                yaw=None,
                roll=None,
                camera_pos=None,
            ))
        else:
            camera_info = sample_camera_extrinsics(
                batch_size=batch_size,
                radius_strategy=radius_strategy,
                radius_fix=radius_fix,
                radius_min=radius_min,
                radius_max=radius_max,
                radius_mean=radius_mean,
                radius_stddev=radius_stddev,
                polar_strategy=polar_strategy,
                polar_fix=polar_fix,
                polar_min=polar_min,
                polar_max=polar_max,
                polar_mean=polar_mean,
                polar_stddev=polar_stddev,
                azimuthal_strategy=azimuthal_strategy,
                azimuthal_fix=azimuthal_fix,
                azimuthal_min=azimuthal_min,
                azimuthal_max=azimuthal_max,
                azimuthal_mean=azimuthal_mean,
                azimuthal_stddev=azimuthal_stddev,
                use_spherical_uniform_position=use_spherical_uniform_position,
                pitch_strategy=pitch_strategy,
                pitch_fix=pitch_fix,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
                pitch_mean=pitch_mean,
                pitch_stddev=pitch_stddev,
                yaw_strategy=yaw_strategy,
                yaw_fix=yaw_fix,
                yaw_min=yaw_min,
                yaw_max=yaw_max,
                yaw_mean=yaw_mean,
                yaw_stddev=yaw_stddev,
                roll_strategy=roll_strategy,
                roll_fix=roll_fix,
                roll_min=roll_min,
                roll_max=roll_max,
                roll_mean=roll_mean,
                roll_stddev=roll_stddev)

        points = get_point_coord(
            rays_camera=rays_camera,
            radii=radii,
            cam2world_matrix=camera_info['cam2world_matrix'])

        return {
            'camera_radius': camera_info['radius'],  # [N]
            'camera_polar': camera_info['polar'],  # [N]
            'camera_azimuthal': camera_info['azimuthal'],  # [N]
            'camera_pitch': camera_info['pitch'],  # [N]
            'camera_yaw': camera_info['yaw'],  # [N]
            'camera_roll': camera_info['roll'],  # [N]
            'camera_pos':camera_info['camera_pos'],  # [N, 3]
            'cam2world_matrix': camera_info['cam2world_matrix'],  # [N, 4, 4]
            'rays_camera': rays_camera,  # [N, H, W, 3]
            'rays_world': points['rays_world'],  # [N, H, W, 3]
            'ray_origins_world': points['ray_origins_world'], # [N, H, W, 3]
            'radii_raw': radii_raw,  # [N, H, W, K]
            'radii': radii,  # [N, H, W, K]
            'points_camera': points['points_camera'],  # [N, H, W, K, 3]
            'points_world': points['points_world']  # [N, H, W, K, 3]
        }


def get_ray_per_pixel(batch_size,
                      image_size,
                      fov,
                      boundary=1.0,
                      focal=None,
                      cam_look_at_dir=-1,
                      pixel_center=False,
                      y_descending=True):
    """Gets ray direction for each image pixel under camera coordinate system.

    Each ray direction is represent by a vector, [x, y, z], under the following
    coordinate system:

    - The origin is set at the camera position.
    - The X axis is set as the horizontal direction of the image plane, with
      larger value on the right.
    - The Y axis is set as the vertical direction of the image plane, with
      larger value on the top.
    - The Z axis is set as the direction perpendicular to the image plane,
      from the image center pointing to the camera. In other words, the z
      coordinate of the image plane is negative.
    - The above coordinate system is a right-hand one.

    Taking a 5x5 image (with boundary 1.0) as an instance, the per-pixel (x, y)
    coordinates should look like:

    (-1.0,  1.0) (-0.5,  1.0) (0.0,  1.0) (0.5,  1.0) (1.0,  1.0)
    (-1.0,  0.5) (-0.5,  0.5) (0.0,  0.5) (0.5,  0.5) (1.0,  0.5)
    (-1.0,  0.0) (-0.5,  0.0) (0.0,  0.0) (0.5,  0.0) (1.0,  0.0)
    (-1.0, -0.5) (-0.5, -0.5) (0.0, -0.5) (0.5, -0.5) (1.0, -0.5)
    (-1.0, -1.0) (-0.5, -1.0) (0.0, -1.0) (0.5, -1.0) (1.0, -1.0)

    NOTE:
        The X-axis focal and Y-axis focal are assumed to be the same according
        to the pinhole camera model.

    Args:
        batch_size: Batch size of images, each of which has the same ray
            directions. Denoted as `N`.
        image_size: Size of the image. One element indicates square image, while
            two elements stand for height and width respectively. Denoted as `H`
            and `W`.
        fov: Field of view (along X axis) of the camera, in unit of degree.
        boundary: The maximum value of the X coordinate. Defaults to `1.0`.
        focal (optional): Focal Length of camera. If given, it will cover the
            focal calculated by `fov`. Note that the focal is a normalized one
            which is divided by size of the image.
        cam_look_at_dir: Direction of camera looks at. Defaults to `-1`, which
            means camera looks at `-z` direction.
        pixel_center: Whether rays originate from the pixel center or not. For
            example, assume a pixel is at (H, W). If `pixel_center` is set
            `True`, then the ray originate from (H+0.5, W+0.5), otherwise it
            originate from (H, W).
        y_descending: Whether the Y axis is in descending order from top to
            bottom. If set `True`, the coordinates are the same as the above
            example. If set `False`, the coordinate system is consistent with
            2D image plane coordinate system, where Y axis is in ascending
            order. Defaults to `True`.
    Returns:
        A tensor, with shape [N, H, W, 3], representing the per-pixel ray
            direction. Each direction is normalized to a unit vector.
    """
    # Check inputs.
    assert isinstance(batch_size, int) and batch_size > 0
    N = batch_size
    assert isinstance(image_size, (int, list, tuple))
    if isinstance(image_size, int):
        H = image_size
        W = image_size
    else:
        H, W = image_size
    assert isinstance(H, int) and H > 0
    assert isinstance(W, int) and W > 0
    assert 0 < fov < 180
    assert boundary > 0

    # Get running device.
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # Get (x, y) grid by boundary.
    max_x = boundary
    max_y = boundary / W * H
    if pixel_center:
        y, x = torch.meshgrid(
            torch.linspace(max_y - 0.5 / H, -max_y + 0.5 / H, H,
                           device=device),
            torch.linspace(-max_x + 0.5 / W, max_x - 0.5 / W, W,
                           device=device))
    else:
        y, x = torch.meshgrid(torch.linspace(max_y, -max_y, H, device=device),
                          torch.linspace(-max_x, max_x, W, device=device))
    # Get z coordinate of the image plane by focal (i.e., FOV).
    if not y_descending:
        y = -y
    if focal is None:
        focal = boundary / np.tan((2 * np.pi * fov / 360) / 2)
    z = np.sign(cam_look_at_dir) * focal * torch.ones_like(x)  # [H, W]
    # Normalize directions to unit vectors.
    rays = F.normalize(torch.stack([x, y, z], dim=-1), dim=-1)  # [H, W, 3]

    return rays.unsqueeze(0).repeat(N, 1, 1, 1)  # [N, H, W, 3]


def sample_points_per_ray(batch_size,
                          image_size,
                          num_points,
                          strategy='uniform',
                          dis_min=None,
                          dis_max=None,
                          dis_mean=None,
                          dis_stddev=None,
                          per_ray_ref=None,
                          per_point_ref=None):
    """Samples per-point radial distance on each ray.

    This function is independent of ray directions, hence, each point is
    represent by a number, indicating its radial distance to the origin (i.e.,
    the camera).

    The following sampling strategies are supported:

    - `uniform`:
        For each ray, the points uniformly locate in range `[dis_min, dis_max]`.

    - `normal`:
        For each ray, the points are sampled subject to
        `Gaussian(dis_mean, dis_stddev^2)`.

    - `ray_dependent`:
        Each ray follows a separate strategy, controlled by `per_ray_ref`.

    - `point_dependent`:
        Each point follows a separate strategy, controlled by `per_point_ref`.

    Args:
        batch_size: Batch size of images, for which points are sampled
            independently. Denoted as `N`.
        image_size: Size of the image. One element indicates square image, while
            two elements stand for height and width respectively. Denoted as `H`
            and `W`.
        num_points: Number of points sampled on each ray. Denoted as `K`.
        strategy: Strategy for point sampling. Defaults to `uniform`.
        dis_min: Minimum radial distance (with camera as the origin) for each
            point. Defaults to `None`.
        dis_max: Maximum radial distance (with camera as the origin) for each
            point. Defaults to `None`.
        dis_mean: Mean radial distance (with camera as the origin) for each
            point. Defaults to `None`.
        dis_stddev: Standard deviation of the radial distance (with camera as
            the origin) for each point. Defaults to `None`.
        per_ray_ref: Reference for each ray, which will guide the sampling
            process. Shape [N, H, W, c] is expected, where `c` is the dimension
            of a single reference. Defaults to `None`.
        per_point_ref: Reference for each point, which will guide the sampling
            process. Shape [N, H, W, K, c] is expected, where `c` is the
            dimension of a single reference. Defaults to `None`.

    Returns:
        A tensor, with shape [N, H, W, K], representing the per-point radial
            distance on each ray. All numbers should be positive, and the
            distances on each ray should follow a non-descending order.

    Raises:
        ValueError: If the sampling strategy is not supported.
        NotImplementedError: If the sampling strategy is not implemented.
    """
    # Check inputs.
    assert isinstance(batch_size, int) and batch_size > 0
    N = batch_size
    assert isinstance(image_size, (int, list, tuple))
    if isinstance(image_size, int):
        H = image_size
        W = image_size
    else:
        H, W = image_size
    assert isinstance(H, int) and H > 0
    assert isinstance(W, int) and W > 0
    assert isinstance(num_points, int) and num_points > 0
    K = num_points
    strategy = strategy.lower()
    if strategy not in _POINT_SAMPLING_STRATEGIES:
        raise ValueError(f'Invalid point sampling strategy: `{strategy}`!\n'
                         f'Strategies allowed: {_POINT_SAMPLING_STRATEGIES}.')

    # Get running device.
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # Sample points according to strategy.
    if strategy == 'uniform':
        assert dis_max >= dis_min > 0
        radii = torch.linspace(dis_min, dis_max, K, device=device)  # [K]
        return radii.reshape(1, 1, 1, K).repeat(N, H, W, 1)  # [N, H, W, K]

    if strategy == 'normal':
        # TODO: Should we support the normal sampling strategy?
        assert dis_mean > 0 and dis_stddev >= 0

    if strategy == 'ray_dependent':
        # TODO: Strategy dependent on depth?
        assert per_ray_ref.ndim == 4
        assert per_ray_ref.shape[:3] == (N, H, W)

    if strategy == 'point_dependent':
        # TODO: This is hierarchical sampling?
        assert per_point_ref.ndim == 5
        assert per_point_ref.shape[:4] == (N, H, W, K)

    raise NotImplementedError(f'Not implemented point sampling strategy: '
                              f'`{strategy}`!')


def perturb_points_per_ray(radii, strategy='middle_uniform'):
    # Stratified sampling approach described in original NeRF paper.
    """Perturbs point radii within their local range on each ray.

    `N`, `H`, `W`, `K` denote batch size, image height, image width, number of
    points per ray, respectively.

    The following perturbing strategies are supported:

    - `no`:
        Disable point perturbation.

    - `middle_uniform`:
        For each point, it is perturbed between two midpoints. One locates
        within the point itself and its previous one on the same ray, while the
        other locates within the point itself and its next one on the same ray.

    - `uniform`:
        For each point, it is perturbed between itself and its next one.
        For example, there are `n+1` points on the ray: [x_0, x_1, ..., x_n].
        Then the perturbed points are [x_0', x_1', ..., x_n'] with distribution
        xi' ~ U(x_i, x_i+1), where x_n+1 = x_n + (x_n - x_n-1).

    - `self_uniform`:
        For each point, it is perturbed around itself.For example, there are
        `n+1` points on the ray: [x_0, x_1, ..., x_n]. Then the perturbed points
        are [x_0', x_1', ..., x_n'] with distribution
        xi' ~ U(x_i - 0.5, x_i+1 - 0.5).

    Args:
        radii: A collection of point radii, with shape [N, H, W, K].
        strategy: Strategy to perturb each point. Defaults to `middle_uniform`.

    Returns:
        A tensor, with shape [N, H, W, K], representing the per-point radial
            distance on each ray. All numbers should be positive, and the
            distances on each ray should follow a non-descending order.

    Raises:
        ValueError: If the input point radii are with invalid shape, or the
            perturbing strategy is not supported.
        NotImplementedError: If the perturbing strategy is not implemented.
    """
    # Check inputs.
    if radii.ndim != 4:
        raise ValueError(f'The input point radii should be with shape '
                         f'[batch_size, height, width, num_points], '
                         f'but `{radii.shape}` is received!')
    strategy = strategy.lower()
    if strategy not in _POINT_PERTURBING_STRATEGIES:
        raise ValueError(f'Invalid point perturbing strategy: `{strategy}`!\n'
                         f'Strategies allowed: {_POINT_PERTURBING_STRATEGIES}.')

    if strategy == 'no':
        return radii

    if strategy == 'middle_uniform':
        # Get midpoints.
        midpoint = (radii[..., 1:] + radii[..., :-1]) / 2  # [N, H, W, K-1]
        # Get intervals.
        left = torch.cat([radii[..., :1], midpoint], dim=-1)  # [N, H, W, K]
        right = torch.cat([midpoint, radii[..., -1:]], dim=-1)  # [N, H, W, K]
        # Uniformly sample within each interval.
        t = torch.rand_like(radii)  # [N, H, W, K]
        return left + (right - left) * t  # [N, H, W, K]
    elif strategy == 'uniform':
        delta = radii[..., 1:2] - radii[..., 0:1]   # [N, H, W, 1]
        t = torch.rand_like(radii)   # [N, H, W, K]
        return radii + t * delta     # [N, H, W, K]
    elif strategy == 'self_uniform':
        delta = radii[..., 1:2] - radii[..., 0:1]   # [N, H, W, 1]
        t = torch.rand_like(radii) - 0.5   # [N, H, W, K]
        return radii + t * delta     # [N, H, W, K]

    raise NotImplementedError(f'Not implemented point perturbing strategy: '
                              f'`{strategy}`!')


def sample_tensor(size,
                  strategy='uniform',
                  entry_fix=None,
                  entry_min=None,
                  entry_max=None,
                  entry_mean=None,
                  entry_stddev=None):
    """Samples a tensor according to specified strategy.

    The following sampling strategies are supported:

    - `fix`:
        Each entry is fixed as `entry_fix`.

    - `uniform`:
        Each entry is uniformly sampled from range `[entry_min, entry_max]`.

    - `normal`:
        Each entry is sampled subject to `Gaussian(entry_mean, entry_stddev^2)`.

    - `hybrid`:
        Each entry is 50% sampled with `uniform` and 50% sampled with `normal`.

    - `truncated_normal`:
        Each entry is sampled subject to a truncated normal distribution, with
        `entry_min` and `entry_max` as the cut-off values.


    Args:
        size: Size of the sampled tensor. This field is expected to be an
            integer, a list, or a tuple.
        strategy: Strategy to sample points. Defaults to `uniform`.
        entry_min: Minimum value of each entry. Defaults to `None`.
        entry_max: Maximum value of each entry. Defaults to `None`.
        entry_mean: Mean value of each entry. Defaults to `None`.
        entry_stddev: Standard deviation of each entry. Defaults to `None`.

    Returns:
        A tensor, with expected size.

    Raises:
        ValueError: If the sampling strategy is not supported.
        NotImplementedError: If the sampling strategy is not implemented.
    """
    # Check inputs.
    if isinstance(size, int):
        size = (size,)
    elif isinstance(size, list):
        size = tuple(size)
    assert isinstance(size, tuple)
    strategy = strategy.lower()
    if strategy not in _TENSOR_SAMPLING_STRATEGIES:
        raise ValueError(f'Invalid tensor sampling strategy: `{strategy}`!\n'
                         f'Strategies allowed: {_TENSOR_SAMPLING_STRATEGIES}.')

    # Get running device.
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    if strategy == 'fix':
        assert entry_fix is not None
        return torch.ones(size, device=device) * entry_fix

    if strategy == 'uniform':
        assert entry_max >= entry_min
        t = torch.rand(size, device=device)
        return entry_min + (entry_max - entry_min) * t

    if strategy == 'normal':
        assert entry_mean is not None and entry_stddev >= 0
        return torch.randn(size, device=device) * entry_stddev + entry_mean

    if strategy == 'hybrid':
        assert entry_max >= entry_min
        assert entry_mean is not None and entry_stddev >= 0
        if np.random.random() < 0.5:
            t = torch.rand(size, device=device)
            return entry_min + (entry_max - entry_min) * t
        return torch.randn(size, device=device) * entry_stddev + entry_mean

    if strategy == 'truncated_normal':
        # TODO: Truncated normal distribution differs from cut-off.
        assert entry_max >= entry_min
        assert entry_mean is not None and entry_stddev >= 0
        tensor = torch.randn(size, device=device) * entry_stddev + entry_mean
        tensor = torch.clamp(tensor, entry_min, entry_max)
        return tensor

    raise NotImplementedError(f'Not implemented tensor sampling strategy: '
                              f'`{strategy}`!')


def sample_camera_extrinsics(batch_size,
                             radius_strategy='fix',
                             radius_fix=None,
                             radius_min=None,
                             radius_max=None,
                             radius_mean=None,
                             radius_stddev=None,
                             polar_strategy='uniform',
                             polar_fix=None,
                             polar_min=None,
                             polar_max=None,
                             polar_mean=None,
                             polar_stddev=None,
                             azimuthal_strategy='uniform',
                             azimuthal_fix=None,
                             azimuthal_min=None,
                             azimuthal_max=None,
                             azimuthal_mean=None,
                             azimuthal_stddev=None,
                             use_spherical_uniform_position=False,
                             pitch_strategy='fix',
                             pitch_fix=0,
                             pitch_min=None,
                             pitch_max=None,
                             pitch_mean=None,
                             pitch_stddev=None,
                             yaw_strategy='fix',
                             yaw_fix=0,
                             yaw_min=None,
                             yaw_max=None,
                             yaw_mean=None,
                             yaw_stddev=None,
                             roll_strategy='fix',
                             roll_fix=0,
                             roll_min=None,
                             roll_max=None,
                             roll_mean=None,
                             roll_stddev=None):
    """Samples camera extrinsics.

    This function supports sampling camera extrinsics from 6 dimensions (here,
    all angles are in unit of radian):

    - Camera position:
        - radius: Distance from the camera position to the origin of the world
            coordinate system.
        - polar: The polar angle with respect to the origin of the world
            coordinate system.
        - azimuthal: The azimuthal angle with respect to the origin of the world
            coordinate system.
    - Camera orientation:
        - pitch: Pitch angle (X axis) regarding the camera coordinate system.
        - yaw: Yaw angle (Y axis) regarding the camera coordinate system.
        - roll: Roll angle (Z axis) regarding the camera coordinate system.

    and then convert the camera extrinsics to camera position and coordinate
    transformation matrix.

    More details about sampling as well as arguments can be found in function
    `sample_tensor()`.

    NOTE:
        Without camera orientation (i.e., `pitch = 0, yaw = 0, roll = 0`), this
        function assumes the camera pointing to the origin of the world
        coordinate system. Furthermore, camera orientation controls the rotation
        within the camera coordinate system, which is independent from the
        transformation across coordinate systems. As a result, the camera does
        not necessarily point to the origin of the world coordinate system
        anymore.

    Args:
        batch_size: Batch size of the sampled camera. Denoted as `N`.
        use_spherical_uniform_position: Whether to sample the camera position
            subject to a spherical uniform distribution. Defaults to False.

    Returns:
        A dictionary, containing
            - `camera_radius`: camera radius w.r.t. the world coordinate system,
                with shape [N].
            - `camera_polar`: camera polar w.r.t. the world coordinate system,
                with shape [N].
            - `camera_azimuthal`: camera azimuthal w.r.t. the world coordinate
                system, with shape [N].
            - `camera_pitch`: camera pitch w.r.t. the camera coordinate system,
                with shape [N].
            - `camera_yaw`: camera yaw w.r.t. the camera coordinate system,
                with shape [N].
            - `camera_roll`: camera roll w.r.t. the camera coordinate system,
                with shape [N].
            - `camera_pos`: camera position, i.e., the (x, y, z) coordinate
                in the world coordinate system, with shape [N, 3].
            - `cam2world_matrix`: transformation matrix to transform the camera
                coordinate system to the world coordinate system, with shape
                [N, 4, 4].
    """
    # Sample camera position.
    radius = sample_tensor(size=batch_size,
                           strategy=radius_strategy,
                           entry_fix=radius_fix,
                           entry_min=radius_min,
                           entry_max=radius_max,
                           entry_mean=radius_mean,
                           entry_stddev=radius_stddev)
    if use_spherical_uniform_position:
        # TODO: Check the local spherical uniform distribution?
        polar = sample_tensor(size=batch_size,
                              strategy='uniform',
                              entry_fix=polar_fix,
                              entry_min=polar_min,
                              entry_max=polar_max,
                              entry_mean=polar_mean,
                              entry_stddev=polar_stddev)
        azimuthal_cos_val = sample_tensor(size=batch_size,
                                          strategy='uniform',
                                          entry_min=azimuthal_min / np.pi,
                                          entry_max=azimuthal_max / np.pi)
        azimuthal = torch.arccos(1 - 2 * azimuthal_cos_val)
    else:
        polar = sample_tensor(size=batch_size,
                              strategy=polar_strategy,
                              entry_fix=polar_fix,
                              entry_min=polar_min,
                              entry_max=polar_max,
                              entry_mean=polar_mean,
                              entry_stddev=polar_stddev)
        azimuthal = sample_tensor(size=batch_size,
                                  strategy=azimuthal_strategy,
                                  entry_fix=azimuthal_fix,
                                  entry_min=azimuthal_min,
                                  entry_max=azimuthal_max,
                                  entry_mean=azimuthal_mean,
                                  entry_stddev=azimuthal_stddev)

    # Sample camera orientation.
    pitch = sample_tensor(size=batch_size,
                          strategy=pitch_strategy,
                          entry_fix=pitch_fix,
                          entry_min=pitch_min,
                          entry_max=pitch_max,
                          entry_mean=pitch_mean,
                          entry_stddev=pitch_stddev)
    yaw = sample_tensor(size=batch_size,
                        strategy=yaw_strategy,
                        entry_fix=yaw_fix,
                        entry_min=yaw_min,
                        entry_max=yaw_max,
                        entry_mean=yaw_mean,
                        entry_stddev=yaw_stddev)
    roll = sample_tensor(size=batch_size,
                         strategy=roll_strategy,
                         entry_fix=roll_fix,
                         entry_min=roll_min,
                         entry_max=roll_max,
                         entry_mean=roll_mean,
                         entry_stddev=roll_stddev)

    # Get running device.
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # Get camera position.
    N = batch_size
    camera_pos = torch.zeros((N, 3), device=device)
    camera_pos[:, 0] = radius * torch.sin(polar) * torch.cos(azimuthal)
    camera_pos[:, 1] = radius * torch.cos(polar)
    camera_pos[:, 2] = radius * torch.sin(polar) * torch.sin(azimuthal)

    # Get transformation matrix with the following steps.
    #   1. Use pitch, yaw, and roll to get the rotation matrix within the camera
    #      coordinate system.
    #   2. Get the forward axis, which points from the camper position to the
    #      origin of the world coordinate system.
    #   3. Get a "pseudo" up axis, which is [0, 1, 0].
    #   4. Get the left axis by crossing the "pseudo" up axis with the forward
    #      axis.
    #   5. Get the "actual" up axis by crossing the forward axis with the left
    #      axis.
    #   6. Get the camera-to-world rotation matrix with the aforementioned
    #      forward axis, left axis, and "actual" up axis.
    #   7. Get the camera-to-world transformation matrix.
    pitch_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    pitch_matrix[:, 1, 1] = torch.cos(pitch)
    pitch_matrix[:, 2, 2] = torch.cos(pitch)
    pitch_matrix[:, 1, 2] = -torch.sin(pitch)
    pitch_matrix[:, 2, 1] = torch.sin(pitch)  # [N, 4, 4]
    yaw_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    yaw_matrix[:, 0, 0] = torch.cos(yaw)
    yaw_matrix[:, 2, 2] = torch.cos(yaw)
    yaw_matrix[:, 2, 0] = -torch.sin(yaw)
    yaw_matrix[:, 0, 2] = torch.sin(yaw)  # [N, 4, 4]
    roll_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    roll_matrix[:, 0, 0] = torch.cos(roll)
    roll_matrix[:, 1, 1] = torch.cos(roll)
    roll_matrix[:, 0, 1] = -torch.sin(roll)
    roll_matrix[:, 1, 0] = torch.sin(roll)  # [N, 4, 4]

    forward_axis = F.normalize(camera_pos * -1, dim=-1)  # [N, 3]
    pseudo_up_axis = torch.as_tensor([0.0, 1.0, 0.0], device=device)  # [3]
    pseudo_up_axis = pseudo_up_axis.reshape(1, 3).repeat(N, 1)  # [N, 3]
    left_axis = torch.cross(pseudo_up_axis, forward_axis, dim=-1)  # [N, 3]
    left_axis = F.normalize(left_axis, dim=-1)  # [N, 3]
    up_axis = torch.cross(forward_axis, left_axis, dim=-1)  # [N, 3]
    up_axis = F.normalize(up_axis, dim=-1)  # [N, 3]

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    rotation_matrix[:, :3, 0] = -left_axis
    rotation_matrix[:, :3, 1] = up_axis
    rotation_matrix[:, :3, 2] = -forward_axis  # [N, 4, 4]

    translation_matrix = torch.eye(4, device=device)
    translation_matrix = translation_matrix.unsqueeze(0).repeat(N, 1, 1)
    translation_matrix[:, :3, 3] = camera_pos  # [N, 4, 4]

    cam2world_matrix = (translation_matrix @ rotation_matrix @
                        roll_matrix @ yaw_matrix @ pitch_matrix)  # [N, 4, 4]

    return {
        'radius': radius,
        'polar': polar,
        'azimuthal': azimuthal,
        'pitch': pitch,
        'yaw': yaw,
        'roll': roll,
        'camera_pos':camera_pos,
        'cam2world_matrix': cam2world_matrix
    }


def get_point_coord(rays_camera, radii, cam2world_matrix):
    """Gets pre-point coordinate in the world coordinate system.

    `N`, `H`, `W`, `K` denote batch size, image height, image width, number of
    points per ray, respectively.

    Args:
        rays_camera: Per-pixel ray direction, with shape [N, H, W, 3], in the
            camera coordinate system.
        radii: Per-point radial distance on each ray, with shape [N, H, W, K].
        cam2world_matrix: Transformation matrix that transforms the camera
            coordinate system to the world coordinate system, with shape
            [N, 4, 4].

    Returns:
        A dictionary, containing
            - `rays_world`: ray directions in the world coordinate system,
                with shape [N, H, W, 3].
            - `points_camera`: per-point coordinate in the camera coordinate
                system, with shape [N, H, W, K, 3].
            - `points_world`: per-point coordinate in the world coordinate
                system, with shape [N, H, W, K, 3].

    Raises:
        ValueError: If any input has invalid shape.
    """
    # Check inputs.
    if rays_camera.ndim != 4 or rays_camera.shape[3] != 3:
        raise ValueError(f'The input rays should be with shape '
                         f'[batch_size, height, width, 3], '
                         f'but `{rays_camera.shape}` is received!')
    N, H, W, _ = rays_camera.shape
    if radii.ndim != 4 or radii.shape[:3] != (N, H, W):
        raise ValueError(f'The input radii should be with shape '
                         f'[batch_size, height, width, num_points],  where '
                         f'batch_size, height, width align with those of rays, '
                         f'but `{radii.shape}` is received!')
    K = radii.shape[3]
    if cam2world_matrix.shape != (N, 4, 4):
        raise ValueError(f'The input cam2world_matrix should be with shape '
                         f'[batch_size, 4, 4], where batch_size align with '
                         f'that of rays and radii '
                         f'but `{cam2world_matrix.shape}` is received!')

    # Get running device.
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # Transform rays.
    rays_world = (cam2world_matrix[:, :3, :3] @
                  rays_camera.reshape(N, -1, 3).permute(0, 2, 1))
    rays_world = rays_world.permute(0, 2, 1).reshape(N, H, W, 3)

    # Transform ray origins.
    ray_origins_homo = torch.zeros((N, H * W, 4), device=device)
    ray_origins_homo[..., 3] = 1
    ray_origins_world = torch.bmm(cam2world_matrix,
                                  ray_origins_homo.permute(0, 2, 1)).permute(
                                      0, 2, 1)[..., :3]
    ray_origins_world = ray_origins_world.reshape(N, H, W, 3)

    # Transform points.
    points_camera = (rays_camera.unsqueeze(3) *
                     radii.unsqueeze(4))  # [N, H, W, K, 3]
    points_camera_homo = torch.cat(
        [points_camera, torch.ones((N, H, W, K, 1), device=device)],
        dim=-1)  # [N, H, W, K, 4]
    points_world_homo = (cam2world_matrix @
                         points_camera_homo.reshape(N, -1, 4).permute(0, 2, 1))
    points_world = points_world_homo.permute(0, 2, 1)[:, :, :3]
    points_world = points_world.reshape(N, H, W, K, 3)

    return {
        'rays_world': rays_world,
        'ray_origins_world': ray_origins_world,
        'points_camera': points_camera,
        'points_world': points_world,
    }
