# imports
import asyncio
import copy
import os
import time

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

import Inpainting.MiDaS.MiDaS_utils as MiDaS_utils
from Inpainting.MiDaS.run import run_depth
from Inpainting.bilateral_filtering import sparse_bilateral_filtering
from Inpainting.mesh import write_ply, read_ply, output_3d_photo
from Inpainting.utils import get_MiDaS_samples, read_MiDaS_depth


def tnow():
    return time.strftime("%H:%M:%S", time.localtime())


class proc_input:
    def __init__(self,
                 config,
                 device,
                 sample,
                 img,
                 mfi,
                 rgb_model,
                 depth_edge_model,
                 depth_feat_model,
                 ):
        self.config = config
        self.device = device
        self.sample = sample
        self.img = img
        self.mfi = mfi
        self.rgb_model = rgb_model
        self.depth_edge_model = depth_edge_model
        self.depth_feat_model = depth_feat_model

    def final(self):
        try:
            process(
                self.config,
                self.device,
                self.sample,
                self.img,
                self.mfi,
                self.rgb_model,
                self.depth_edge_model,
                self.depth_feat_model,
            )
        except Exception as e:
            if self.config is not None:
                print(f'Error during process for {self.config["specific"]}: {e}')
            else:
                print(f'Error during process (hint: config nil): {e}')
                exit(1)


async def magic(config, device, model, rgb_model, depth_edge_model, depth_feat_model):
    sample_list = get_MiDaS_samples(
        config["src_folder"], config["depth_folder"], config, config["specific"]
    )  # dict of important stuffs
    print(f'processing {config["specific"]} at {tnow()}')

    task_queue = asyncio.Queue()
    second_task_queue = asyncio.Queue()

    async def worker(queue):
        while True:
            if queue.empty():
                break
            sample = await queue.get()
            if sample == None:
                break
            do_depth(sample, config)
            queue.task_done()
            print("+1 done")

    def do_depth(smp, conf):
        try:
            print(f'Running depth extraction for {config["specific"]} at {tnow()}')
            run_depth(
                device,
                [smp["ref_img_fi"]],
                conf["src_folder"],
                conf["depth_folder"],  # compute depth
                model,
                MiDaS_utils,
                target_w=1280,
            )
        except Exception as e:
            print(f'Error during do_depth: {e}')

    # iterate over each image.
    for idx in tqdm(range(len(sample_list))):
        try:
            sample = sample_list[idx]  # select image
            # print("Current Source ==> ", sample["src_pair_name"])
            mesh_fi = os.path.join(config["mesh_folder"], sample["src_pair_name"] + ".ply")
            image = imageio.imread(sample["ref_img_fi"])
            if config["require_midas"] is True:
                # if we need midas then queue up the depth extraction as a task
                task_queue.put_nowait(sample)
        except Exception as e:
            print(f'Error during sample listing: {e}')
            continue

        # either way, queue up the process as a task that will wait for all pending depth extractions to finish
        second_task_queue.put_nowait(
            proc_input(config, device, sample, image, mesh_fi, rgb_model, depth_edge_model, depth_feat_model)
        )

    # run all depth extractions
    await asyncio.gather(*[worker(task_queue) for _ in range(5)])

    # wait for all depth extractions to finish
    await task_queue.join()

    # return the second task queue so that we can start it later
    return second_task_queue


def process(
        config,
        device,
        sample,
        image,
        mesh_fi,
        rgb_model,
        depth_edge_model,
        depth_feat_model,
        normal_canvas=None,
        all_canvas=None):
    depth = None

    if "npy" in config["depth_format"]:
        config["output_h"], config["output_w"] = np.load(sample["depth_fi"]).shape[:2]
    else:
        config["output_h"], config["output_w"] = imageio.imread(
            sample["depth_fi"]
        ).shape[:2]

    frac = config["longer_side_len"] / max(config["output_h"], config["output_w"])
    config["output_h"], config["output_w"] = int(config["output_h"] * frac), int(
        config["output_w"] * frac
    )
    config["original_h"], config["original_w"] = (
        config["output_h"],
        config["output_w"],
    )
    if image.ndim == 2:
        image = image[..., None].repeat(3, -1)
    if (
            np.sum(np.abs(image[..., 0] - image[..., 1])) == 0
            and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0
    ):
        config["gray_image"] = True
    else:
        config["gray_image"] = False

    image = cv2.resize(
        image,
        (config["output_w"], config["output_h"]),
        interpolation=cv2.INTER_AREA,
    )

    depth = read_MiDaS_depth(
        sample["depth_fi"], 3.0, config["output_h"], config["output_w"]
    )  # read normalized depth computed

    mean_loc_depth = depth[depth.shape[0] // 2, depth.shape[1] // 2]

    # starty = time.time()

    if not (config["load_ply"] is False or os.path.exists(mesh_fi) == False):
        mesh(  # load ply file
            config,
            device,
            sample,
            image,
            depth,
            mesh_fi,
            mean_loc_depth,
            normal_canvas,
            all_canvas,
        )

    vis_photos, vis_depths = sparse_bilateral_filtering(
        depth.copy(),
        image.copy(),
        config,
        num_iter=config["sparse_iter"],
        spdb=False,
    )  # do bilateral filtering
    depth = vis_depths[-1]
    model = None
    torch.cuda.empty_cache()

    ## MODEL INITS

    # print(f'Running 3D_Photo and loading edge model for {config["specific"]} at {tnow()}')

    print(f'doing depth ply and other heavy lifting for {config["specific"]} at {tnow()}')

    # do some mesh work
    # starty = time.time()
    rt_info = write_ply(
        image,
        depth,
        sample["int_mtx"],
        mesh_fi,
        config,
        rgb_model,
        depth_edge_model,
        depth_edge_model,
        depth_feat_model,
    )

    if rt_info is False:
        return

    # rgb_model = None
    # color_feat_model = None
    # depth_edge_model = None
    # depth_feat_model = None
    # torch.cuda.empty_cache()
    # print(f'Total Time taken for {config["specific"]}: {time.strftime("%M:%S", time.time() - starty)}')


def mesh(config, rt_info, mesh_fi, sample, image, depth, normal_canvas, all_canvas, mean_loc_depth):
    print(f'Loading mesh for {config["specific"]} at {tnow()}')

    if config["save_ply"] is True or config["load_ply"] is True:
        verts, colors, faces, Height, Width, hFov, vFov = read_ply(
            mesh_fi
        )  # read from whatever mesh thing has done
    else:
        verts, colors, faces, Height, Width, hFov, vFov = rt_info

    print(f'Making video for {config["specific"]} at {tnow()}')
    videos_poses, video_basename = (
        copy.deepcopy(sample["tgts_poses"]),
        sample["tgt_name"],
    )
    top = (
            config.get("original_h") // 2 - sample["int_mtx"][1, 2] * config["output_h"]
    )
    left = (
            config.get("original_w") // 2 - sample["int_mtx"][0, 2] * config["output_w"]
    )
    down, right = top + config["output_h"], left + config["output_w"]
    border = [int(xx) for xx in [top, down, left, right]]
    normal_canvas, all_canvas = output_3d_photo(
        verts.copy(),
        colors.copy(),
        faces.copy(),
        copy.deepcopy(Height),
        copy.deepcopy(Width),
        copy.deepcopy(hFov),
        copy.deepcopy(vFov),
        copy.deepcopy(sample["tgt_pose"]),
        sample["video_postfix"],
        copy.deepcopy(sample["ref_pose"]),
        copy.deepcopy(config["video_folder"]),
        image.copy(),
        copy.deepcopy(sample["int_mtx"]),
        config,
        image,
        videos_poses,
        video_basename,
        config.get("original_h"),
        config.get("original_w"),
        border=border,
        depth=depth,
        normal_canvas=normal_canvas,
        all_canvas=all_canvas,
        mean_loc_depth=mean_loc_depth,
    )


async def inpaint(config, device, model, rgb_model, depth_edge_model, depth_feat_model):
    try:
        # llm: append the returned task queue from magic() to the big_tasks queue to be processed later
        return await magic(config, device, model, rgb_model, depth_edge_model, depth_feat_model)
    except Exception as e:
        print(f'FUBAR\'d: {e}')
        exit(1)
