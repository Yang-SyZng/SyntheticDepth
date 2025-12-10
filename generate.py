import os
import torch
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import warpped_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import numpy as np


def generate_depth(dataset, opt, pipe):
    scene = Scene(dataset)
    scene.training_setup(opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()

    with torch.no_grad():
        for camera in tqdm(viewpoint_stack):
            xyz, features, opacity, scales, rotations, cov3D_precomp, \
                active_sh_degree, max_sh_degree, masks = scene.get_gaussian_parameters(camera.world_view_transform, pipe.compute_cov3D_python, random = scene.max_level)
            render_pkg = render(camera, xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, pipe, background, cov3D_precomp = cov3D_precomp)
            accum_alpha = 1.0 - render_pkg["final_T"]
            depth = torch.where(accum_alpha == 0.0, torch.zeros_like(render_pkg["depth"]), render_pkg["depth"] / accum_alpha)
            depth = warpped_depth(depth)
            depth = Image.fromarray(torch.stack([depth[0] * 255.0, accum_alpha[0] * 255.0], dim=2).detach().cpu().numpy().astype(np.uint8), mode="LA")
            depth.save(os.path.join(args.source_path, 'depths', camera.image_name + ".png"))


# Usage:
# python generate.py -s path/to/data --data_device cpu
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6021)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    
    print("[ Generating Depth ] Optimizing With Parameters: " + str(vars(args)))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    generate_depth(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("[ Generating Depth ] complete.")