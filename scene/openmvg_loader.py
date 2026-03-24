#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from natsort import natsorted
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.scene_utils import CameraInfo, SceneInfo
from utils.general_utils import PILtoTorch

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}

'''
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # positions = np.vstack([vertices['x'], -vertices['z'], vertices['y']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
'''
def fetchPly(path):
    # filter out the green points (camera trajectory)
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    xs = vertices['x']
    ys = vertices['y']
    zs = vertices['z']
    rs = vertices['red']
    gs = vertices['green']
    bs = vertices['blue']
    mask = ~((rs == 0) & (gs == 255) & (bs == 0))
    positions = np.vstack([xs[mask], ys[mask], zs[mask]]).T
    colors = np.vstack([rs[mask], gs[mask], bs[mask]]).T / 255.0
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCamerasFromOpenMVG(args, path, extrinsicsfile, cam_dict, white_background):
    cam_infos = []

    with open(os.path.join(path, extrinsicsfile)) as json_file:
        contents = json.load(json_file)
        fovx = 3.13768641 # panoramic render do not need this

        frames = contents["extrinsics"]
        time_duration = args.time_duration
        frame_num = len(frames)

        # TODO change to right order
        for idx, frame in enumerate(frames):
            cam_key = frame["key"]
            cam_name = os.path.join(path, 'images', cam_dict[cam_key])
            
            R = np.array(frame["value"]["rotation"]).T # .T: w2c->c2w
            T = -np.array(frame["value"]["rotation"]) @ np.array(frame["value"]["center"]) # w2c

            #image_path = os.path.join(path, cam_name)
            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            image = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            #image = Image.fromarray((arr * 255.0).astype(np.uint8), "RGB")
            #image = PILtoTorch(image,None)
            if args.load_dynamic_mask:
                mask_path = os.path.join(path, 'dynamic_mask', cam_dict[cam_key].replace('jpg','png'))
                mask = Image.open(mask_path).convert('L')
                dynamic_mask = np.array(mask)
                dynamic_mask = (dynamic_mask>0).astype(np.uint8)
            else:
                dynamic_mask = None
            
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[0])
            FovY = fovy 
            FovX = fovx

            timestamp =  time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (frame_num - 1)
            cam_infos.append(
                CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                    dynamic_mask=dynamic_mask, fx=-1,fy=-1,cx=-1,cy=-1,
                    image_path=image_path, image_name=image_name, 
                    width=image.shape[1], height=image.shape[0],
                    timestamp = timestamp
                )
            )
            
    return cam_infos


def readOpenMVGInfo(args):
    path, eval = args.source_path, args.eval
    white_background = args.white_background


    print("Reading Transforms from OpenMVG")

    my_views = os.path.join(path, "data_views.json")
    camfile_dict = {}
    with open(my_views) as views:
        json_views = json.load(views)
        camview_list = json_views["views"]
        for camview in camview_list:
            camfile_dict[camview["key"]] = camview["value"]["ptr_wrapper"]["data"]["filename"]

    cam_infos_unsorted = readCamerasFromOpenMVG(args, path, "data_extrinsics.json", camfile_dict, white_background)
    cam_infos = natsorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    if os.path.exists(os.path.join(path, 'train.txt')) and os.path.exists(os.path.join(path, 'test.txt')):
        train_file = os.path.join(path, 'train.txt')
        test_file = os.path.join(path, 'test.txt')
        with open(train_file, 'r') as f:
            train_name_list = f.read().splitlines() 
        with open(test_file, 'r') as f:
            test_name_list = f.read().splitlines() 
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_name_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_name_list]
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold == 0]
        
    print(f"# of Train: {len(train_cam_infos)}, \t# of Test: {len(test_cam_infos)}")

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if os.path.exists(os.path.join(path, "pcd.ply")):
        print("Points without camera position (Green points) are initialized")
        ply_path = os.path.join(path, "pcd.ply")
    else:
        ply_path = os.path.join(path, "colorized.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError('No initial pcd file found!')
        # # Since this data set has no colmap data, we start with random points
        # num_pts = 100_000
        # print(f"Generating random point cloud ({num_pts})...")
        
        # # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        # shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_duration=args.time_duration,
                           time_interval=(args.time_duration[1] - args.time_duration[0]) / (len(cam_infos)))
    return scene_info


