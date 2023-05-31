import asyncio
import os
import sys

import torch
import yaml

from Inpainting import main
from Inpainting.MiDaS.monodepth_net import MonoDepthNet
from Inpainting.networks import Inpaint_Edge_Net, Inpaint_Depth_Net, Inpaint_Color_Net

# -----------------------

global deviceStr
global model
global device
global depth_edge_model_ckpt
global depth_edge_model
global depth_feat_model_ckpt
global depth_feat_model
global rgb_model_ckpt
global rgb_model


def setup_new_arg_file(file_name):
    if not os.path.exists("Inpainting/tmp"):
        os.mkdir("Inpainting/tmp")
    # copy Inpainting/argument.yml to Inpainting/{file_name}.yml and then use the new file as argtarget value
    # this is to make sure that each image is inpainted with its own argument.yml file
    argtarget = f'Inpainting/tmp/{file_name}.yml'
    if not os.path.exists(argtarget):
        print(f"copying argument.yml to {argtarget}...")
        with open("Inpainting/argument.yml") as f:
            with open(argtarget, "w") as f1:
                for line in f:
                    f1.write(line)
    else:
        print(f"{argtarget} already exists. using it...")
    return argtarget


def create_dirs(config):
    # create some directories
    try:
        os.makedirs(config["mesh_folder"], exist_ok=True)
        os.makedirs(config["video_folder"], exist_ok=True)
        os.makedirs(config["depth_folder"], exist_ok=True)
    except:
        pass


def config_ops(argtarget, file_name):
    # y tho???

    print(f"reading {argtarget} for arguments...")
    with open(argtarget) as f:
        list_doc = yaml.safe_load(f)
        f.close()

    list_doc["src_folder"] = sys.argv[1]
    list_doc["depth_folder"] = "Output"
    list_doc["require_midas"] = True

    list_doc["specific"] = file_name.split(".")[0]

    with open(argtarget, "w") as f:
        yaml.dump(list_doc, f)

    # command line arguments
    with open(argtarget, "r") as f:
        config = yaml.safe_load(f)
        f.close()

    # replacing that ridiculous sed call...
    config["offscreen_rendering"] = False

    # uhh... but don't we know that it's false? (well now we DEFINITELY do...)
    # if config["offscreen_rendering"] is True:
    #    vispy.use(app="egl")

    return config


def get_device_and_model(gpu_ids, model_path, nnet):
    global depth_edge_model_ckpt
    global depth_edge_model
    global depth_feat_model_ckpt
    global depth_feat_model
    global rgb_model_ckpt
    global rgb_model
    global deviceStr
    global device
    global model

    depth_edge_model_ckpt = "Inpainting/checkpoints/edge-model.pth"
    depth_feat_model_ckpt = "Inpainting/checkpoints/depth-model.pth"
    rgb_model_ckpt = "Inpainting/checkpoints/color-model.pth"

    # set gpu ids
    deviceStr = "cuda:0"

    print(f'using device {deviceStr}')

    # select device
    device = torch.device(deviceStr)
    # print("device: %s" % device)

    print('Loading model from {}...'.format(model_path))

    # load network
    model = nnet(model_path)
    model.to(device)
    model.eval()
    depth_edge_model = Inpaint_Edge_Net(
        init_weights=True
    )  # init edge inpainting model
    depth_edge_weight = torch.load(
        depth_edge_model_ckpt, map_location="cuda:" + str(gpu_ids[0])
    )
    depth_edge_model.load_state_dict(depth_edge_weight)
    depth_edge_model = depth_edge_model.to(deviceStr)
    depth_edge_model.eval()  # in eval mode

    print(f'Loading depth model from {depth_feat_model_ckpt}')
    depth_feat_model = Inpaint_Depth_Net()  # init depth inpainting model
    depth_feat_weight = torch.load(
        depth_feat_model_ckpt, map_location=torch.device(deviceStr)
    )
    depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
    depth_feat_model = depth_feat_model.to(deviceStr)
    depth_feat_model.eval()
    depth_feat_model = depth_feat_model.to(deviceStr)

    print(f'Loading rgb model from {rgb_model_ckpt}')  # init color inpainting model
    rgb_model = Inpaint_Color_Net()
    rgb_feat_weight = torch.load(
        rgb_model_ckpt, map_location=torch.device(deviceStr)
    )
    rgb_model.load_state_dict(rgb_feat_weight)
    rgb_model.eval()
    rgb_model = rgb_model.to(deviceStr)
    # graph = None


async def run():
    get_device_and_model([0], "Inpainting/MiDaS/model.pt", MonoDepthNet)
    sem = asyncio.Semaphore(5)
    # create task queue
    tasks = asyncio.Queue()
    # collect all the tasks from concurrent_inpaint into a big task queue to be executed at the end
    for file_name in os.listdir(sys.argv[1]):
        async with sem:
            print(f'inpainting started for {file_name}')
            argtarget = setup_new_arg_file(file_name)
            config = config_ops(argtarget, file_name)
            create_dirs(config)
            pi = await main.inpaint(config, deviceStr, model, rgb_model, depth_edge_model, depth_feat_model)
            tasks.put_nowait(pi)
    # complete all the tasks in the task queue
    while not tasks.empty():
        proc_input = await tasks.get()
        proc_input.final()
        tasks.task_done()


if __name__ == '__main__':
    asyncio.run(run())
