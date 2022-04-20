from cmath import cos
from math import gamma
from typing import Tuple
import numpy as np
import torch
import constants
#from pipeline import run

def gaussian_apply(var: int, sigma: int, matrix: torch.Tensor) -> torch.Tensor:
    """ apply gaussian filter on the matrix
    """
    return 

def gaussian(var: int, sigma: int, size: int) -> torch.Tensor:
    """ return gaussian filter of size <size> """
    return 

def ray(height: int, width: int, focal_length: int, camera2world) -> Tuple[torch.Tensor, torch.Tensor]:
    """ return <d> and <o> as specified in paper """

    # Origin of rays passing through center of each pixel is the same. In camera
    # coordinates the origin is (0, 0, 0). To transform this origin to world 
    # coordinates we will multiply it by the camera-to-world matrix (3X4)
    camera2world = camera2world.detach().numpy()
    camera2world = camera2world.reshape(4,4)
    origin = np.zeros((height, width, 3))
    #print(camera2world.shape)
    org = camera2world[:, -1].reshape(4,1)
    #print(camera2world.shape)
    for pixel_y in range(height):
        for pixel_x in range(width):
            origin[pixel_y, pixel_x] = org[:-1].reshape(3,)
    direction = np.zeros((height, width, 3))

    # Iterate over each pixel and compute direction of the ray passing through 
    # the center of the pixel in world coordinates
    for pixel_y in range(height):
        for pixel_x in range(width):
            x_dir = (2 * (pixel_x + 0.5)/width -1) 
            # multiply by aspect ratio and scale
            x_dir = x_dir * (width/height) * (1/(2*focal_length))
            y_dir = (1 - 2*(pixel_y + 0.5)/height) * (1/(2*focal_length))
            d = np.array([x_dir, y_dir, -1])
            dir = np.sum(d[np.newaxis, :] * camera2world[:3,:3],-1) # getting 5/9 factor in direction
            direction[pixel_y][pixel_x] = dir/np.linalg.norm(dir)
    return torch.from_numpy(origin), torch.from_numpy(direction)


def tester_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    k = dirs[..., np.newaxis, :] * c2w[:3, :3]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def positional_encoding(p: int, L: int) -> torch.Tensor:
    """
    needs to be implemented each coordinate of x
    p - float
    L - dimension for returned vector
    gamma - a vector with 2L dimension
    """
    p = p.detach().numpy()
    twos_power = np.array([i for i in range(L)])
    index = np.array([j for i in range(L) for j in (i, L-1+i)])
    temp = np.dot(p[..., None],((2^twos_power)*np.pi).reshape(-1,1).T)
    sin = np.sin(temp)
    cos = np.cos(temp)
    gamma = np.concatenate((sin,cos), axis=-1)[...,index]
    #print(gamma.shape, 'k')
    return torch.tensor(gamma)

def convert_to_ndc(o, d):
    """ converts origin and direction to normalized device coordinates"""
    
    return o, d


def batchify(x, batch_size: int =  1024 * 8) -> torch.Tensor:
    """ returns batches of <x> with size at most <batch_size> 
    dim=-1 is always batch_size
    """
    return [x[i : i + batch_size] for i in range(0, x.shape[0], batch_size)]

def sample(input, weight, num_samples: int = 64):
    # normalise weights 
    weights = weight  + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )  # (batchsize, len(bins))

    # Take uniform samples

    u = torch.linspace(
        0.0, 1.0, steps=num_samples, dtype=weights.dtype, device=weights.device
    )
    u = u.expand(list(cdf.shape[:-1]) + [num_samples])

    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()
    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(input.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def run_nerf_for_model(model, x, direction):
    #print(x.shape, direction.shape)
    gamma_x = positional_encoding(x, constants.EMBEDD_X)
    gamma_x=gamma_x.reshape((*(gamma_x.shape[:2]),-1))
    batch_size = gamma_x.shape[0]
    num_samples = gamma_x.shape[1]
    gamma_d = positional_encoding(direction, constants.EMBEDD_D).reshape(batch_size,1,3,8).expand(batch_size,num_samples,3,8).reshape(batch_size, num_samples, 24)
    temp0 = gamma_x.size(dim=-1)
    temp1 = gamma_d.size(dim=-1)
    #print(gamma_x.shape, gamma_d.shape)
    batch_x, batch_d = [], []
    batches = batchify(torch.cat((gamma_d, gamma_x),dim=-1))
    for i in batches:
        #print(i.shape)
        T0, T1 = torch.split(i, [temp0, temp1], dim=-1)
        batch_x.append(T0)
        batch_d.append(T1)
    preds = [model(batch_x[i].double(), batch_d[i].double()) for i in range(len(batches))]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(list(x.shape[:-1]) + [radiance_field.shape[-1]])
    return radiance_field

def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
):
    # TESTED
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :3])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values
    depth_map = depth_map.sum(dim=-1)
    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, weights
def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod

def render(radiance_field, depth, dir):
    """Returns rbg map and weights for rays"""
    far_pos = torch.tensor([constants.LARGE_NUM], dtype=dir.dtype, device=dir.device).expand(depth[..., :1].shape)
    dists = torch.cat((depth[..., 1:] - depth[..., :-1],far_pos),dim=-1) * dir[..., None, :].norm(p=2, dim=-1)
    alpha = 1.0-torch.exp(-torch.nn.functional.relu(radiance_field[..., 3]) * dists)
    temp = torch.roll(torch.cumprod(1.0 - alpha + 1e-10, -1),1,-1)
    temp[...,0] = 1
    weights = alpha * temp 
    rgb = torch.sigmoid(radiance_field[..., :3])
    rgb= (weights[..., None] * rgb).sum(dim=-2)
    print(rgb.shape,'hjhjhjhjhj')
    return rgb, weights # (rgb, weights)


def predict_coarse_then_fine(coarse_model, fine_model, origin, direction):
    t = torch.linspace(0., 1., constants.COARSE_NUM, dtype = origin.dtype, device= origin.device)
    batch_size = origin.size(dim=0)
    batch_t = t.expand([batch_size, constants.COARSE_NUM]) #batch_size x COARSE_NUM
    #TODO : implement stratified sampling on <batch_t>
    x = origin[..., None, :] + direction[..., None, :] * batch_t[..., :, None] #batch_size x COARSE_NUM x 3
    coarse_radiance = run_nerf_for_model(coarse_model, x, direction)
    print(coarse_radiance.shape)
    coarse, weights = volume_render_radiance_field(coarse_radiance, batch_t, direction)
    print(coarse.shape)
    t_samples = sample(0.5*(batch_t[...,1:]+batch_t[...,:-1]), weights[...,1:-1],).detach() #input size is batch_size x (COARSE_NUM -1)
    batch_t, _ = torch.sort(torch.cat((t_samples, batch_t), -1), dim=-1)
    x = origin[..., None, :] + direction[..., None, :] * batch_t[..., :, None]
    fine_radiance = run_nerf_for_model(fine_model, x, direction)
    fine, _ = volume_render_radiance_field(fine_radiance, batch_t, direction)
    return coarse, fine



# def predict_coarse_then_fine(coarse_model, fine_model, origin_batches, direction_batches):
#     t = torch.linspace(0., 1., constants.COARSE_NUM, dtype = origin_batches.dtype, device= origin_batches.device)
#     batch_size = origin_batches.size(dim=0)
#     num_batches = origin_batches.size(dim=2)
#     batch_t = t.expand([batch_size, constants.COARSE_NUM, num_batches])
#     #TODO : implement stratified sampling on <batch_t>
#     x = origin_batches[..., None, :, :num_batches] + direction_batches[..., None, :, :num_batches] * batch_t[..., :, None, num_batches] #batch_size x COARSE_NUM x 3 x num_batches
#     gamma_x = positional_encoding(x, constants.EMBEDD_X)
#     gamma_d = positional_encoding(direction_batches, constants.EMBEDD_D) #TODO : make sure it returns a tensor whose last dimension is <num_batches>
    

    
#     return


def one_iteration(coarse_model, fine_model, o, d):
    origin, direction = convert_to_ndc(o, d)
    
    origin = torch.reshape(origin, (-1, 3)) #Nx3
    num_samples = origin.size(dim=0)
    direction = torch.reshape(direction, (-1, 3)) #Nx3
    batches = batchify(torch.cat((origin, direction), 1)) #batch_size x (3 + 3) x N/batch_size
    #print(batches[0].shape)
    pred = []
    for i in batches:

        t0, t1 = torch.split(i, [3,3], 1)
        pred.append(predict_coarse_then_fine(coarse_model, fine_model, t0, t1))
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    return tuple(synthesized_images)

if __name__ == "__main__":
    height = 5
    width = 7
    c2w = np.arange(3,19).reshape((4, 4))
    #origin, direction = tester_rays(height, width, 4, c2w)
    origin, direction = ray(height, width, 4, c2w)
    print(origin)
    print("///////")
    print(direction)
