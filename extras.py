from cmath import cos
from math import gamma
from typing import Tuple
import numpy as np
import torch
import constants
from pipeline import run

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

    origin = np.zeros((height, width, 3))
    org = camera2world[:, -1]
    for pixel_y in range(height):
        for pixel_x in range(width):
            origin[pixel_y][pixel_x] = org
    

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
    return origin, direction


def tester_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    print(dirs[..., np.newaxis, :].shape)
    k = dirs[..., np.newaxis, :] * c2w[:3, :3]
    print(k.shape)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    print(rays_d.shape)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def render_rays():
    
    return 

def positional_encoding(p: int, L: int) -> torch.Tensor:
    """
    needs to be implemented each coordinate of x
    p - float
    L - dimension for returned vector
    gamma - a vector with 2L dimension
    """
    twos_power = np.array([i for i in range(L)])
    temp = (2^twos_power)*np.pi*p
    sin = np.sin(temp)
    cos = np.cos(temp)
    gamma = np.hstack((sin,cos)).reshape(-1, 1)
    return torch.tensor(gamma)

def convert_to_ndc(o, d):
    """ converts origin and direction to normalized device coordinates"""
    return o, d


def batchify(x, batch_size: int) -> torch.Tensor:
    """ returns batches of <x> with size at most <batch_size> 
    dim=-1 is always batch_size
    """
    return x

def sample(input, weight):
    return 

def run_nerf_for_model(model, x, direction):
    gamma_x = positional_encoding(x, constants.EMBEDD_X)
    gamma_d = positional_encoding(direction, constants.EMBEDD_D)
    temp0 = gamma_x.size(dim=-1)
    temp1 = gamma_d.size(dim=-1)
    batches = batchify(torch.cat((gamma_d, gamma_x),dim=-1))
    batch_d, batch_x = torch.split(batches, [temp0, temp1], dim=-2)
    preds = [model(batch_x[..., i], batch_d[..., i]) for i in range(batch_d.size(dim=-1))]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(origin.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field

def render(radiance_field, depth, dir):
    """Returns rbg map"""
    rgb_color = torch.sigmoid(radiance_field[..., :3])
    sigma = torch.nn.functional.relu(radiance_field[..., 3])
    return

def predict_coarse_then_fine(coarse_model, fine_model, origin, direction):
    t = torch.linspace(0., 1., constants.COARSE_NUM, dtype = origin.dtype, device= origin.device)
    batch_size = origin.size(dim=0)
    batch_t = t.expand([batch_size, constants.COARSE_NUM])
    #TODO : implement stratified sampling on <batch_t>
    x = origin[..., None, :] + direction[..., None, :] * batch_t[..., :, None] #batch_size x COARSE_NUM x 3
    coarse_radiance = run_nerf_for_model(coarse_model, x, direction)
    coarse, weights = render(coarse_radiance, batch_t, direction)
    t_samples = sample(0.5*(batch_t[...,1:]+batch_t[...,:-1]), weights[...,1:-1]).detach()
    batch_t, _ = torch.sort(torch.cat((t_samples, batch_t), -1), dim=-1)
    x = origin[..., None, :] + direction[..., None, :] * batch_t[..., :, None]
    fine_radiance = run_nerf_for_model(fine_model, x, direction)
    fine, _ = render(fine_radiance, batch_t, direction)
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
    origin_batches, direction_batches = torch.split(batches, [3,3], 1)
    predict_coarse_then_fine(coarse_model, fine_model, origin_batches, direction_batches)

    return 

if __name__ == "__main__":
    height = 5
    width = 7
    c2w = np.arange(3,15).reshape((3, 4))
    #origin, direction = tester_rays(height, width, 4, c2w)
    origin, direction = ray(height, width, 4, c2w)
    #print(origin)
    print("///////")
    print(direction)
