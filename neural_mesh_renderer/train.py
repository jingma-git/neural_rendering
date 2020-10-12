import torch
import os
import argparse
from datasets import ShapeNet
import models
from tensorboardX import SummaryWriter
from datetime import datetime
import numpy as np
import neural_renderer as nr
import cv2

# python train.py -eid chair_03001627 -cls 03001627 -dd data/shapenet/mesh_reconstruction -ls 0.001 -li 1000 -ni 25000

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LR_REDUCE_POINT = 0.95

elevation = 30.
distance = 2.732

NUM_ITERATIONS = 1000000
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
LAMBDA_SMOOTHNESS = 0

LOG_INTERVAL = 10
RANDOM_SEED = 0
MODEL_DIRECTORY = './data/models'
DATASET_DIRECTORY = './data/shapenet/mesh_reconstruction/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# for visualization
distances = np.ones(24, 'float32') * distance
elevations = np.ones(24, 'float32') * elevation
azimuths = torch.linspace(0, 360, 24).to(device)
distances = torch.tensor(distances).to(device)
elevations = torch.tensor(elevations).to(device)
viewpoints = nr.get_points_from_angles(distances, elevations, azimuths)

def parse_args():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', '--experiment_id', type=str)
    parser.add_argument('-d', '--model_directory', type=str, default=MODEL_DIRECTORY)
    parser.add_argument('-dd', '--dataset_directory', type=str, default=DATASET_DIRECTORY)
    parser.add_argument('-cls', '--class_ids', type=str, default=CLASS_IDS_ALL)
    parser.add_argument('-bs', '--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('-ls', '--lambda_smoothness', type=float, default=LAMBDA_SMOOTHNESS)
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('-lrp', '--lr_reduce_point', type=float, default=LR_REDUCE_POINT)
    parser.add_argument('-ni', '--num_iterations', type=int, default=NUM_ITERATIONS)
    parser.add_argument('-li', '--log_interval', type=int, default=LOG_INTERVAL)
    parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    return args


def eval_impl(model, dataset_val, step=None, output_directory=None):
    if step == None:
        step = "_eval"
    if output_directory == None:
        output_directory = "eval"

    model.eval()
    with torch.no_grad():
        for class_id in dataset_val.class_ids:
            eval_cnt = 0
            for images, voxles in dataset_val.get_all_batches_for_evaluation(100, class_id):
                images_out, _, _ = model.reconstruct_and_render(images[:24], viewpoints)
                images_out = images_out.detach().cpu().numpy()
                out_dir = os.path.join(output_directory, "step{}".format(step), str(eval_cnt))
                os.makedirs(out_dir, exist_ok=True)
                for i in range(24):
                    img = images_out[i].transpose((1, 2, 0))
                    out_path = os.path.join(out_dir, "img{}.jpg".format(i))
                    cv2.imwrite(out_path, img * 255)
                eval_cnt += 1

def gen_shape():
    args = parse_args()
    output_directory = os.path.join(args.model_directory, args.experiment_id)
    checkpoint_path = os.path.join(output_directory, 'checkpoint.pth')

    model = models.Model(lambda_smoothness=args.lambda_smoothness)
    if os.path.exists(checkpoint_path):
        cp = torch.load(checkpoint_path)
        model.load_state_dict(cp)
        print("load model successfully!")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        images_out, _, _ = model.gen_shape(viewpoints)
        images_out = images_out.detach().cpu().numpy()
        out_dir = os.path.join(output_directory, "random_shape")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(24):
            img = images_out[i].transpose((1, 2, 0))
            out_path = os.path.join(out_dir, "img{}.jpg".format(i))
            cv2.imwrite(out_path, img * 255)

def eval():
    args = parse_args()
    output_directory = os.path.join(args.model_directory, args.experiment_id)
    checkpoint_path = os.path.join(output_directory, 'checkpoint.pth')

    dataset_val = ShapeNet(args.dataset_directory, args.class_ids.split(','), 'val')
    model = models.Model(lambda_smoothness=args.lambda_smoothness)
    if os.path.exists(checkpoint_path):
        cp = torch.load(checkpoint_path)
        model.load_state_dict(cp)
        print("load model successfully!")
    model.to(device)

    eval_impl(model, dataset_val, output_directory=output_directory)

def train():
    args = parse_args()
    output_directory = os.path.join(args.model_directory, args.experiment_id)
    checkpoint_path = os.path.join(output_directory, 'checkpoint.pth')
    logger = SummaryWriter(os.path.join(output_directory, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")))

    dataset_train = ShapeNet(args.dataset_directory, args.class_ids.split(','), 'train')
    dataset_val = ShapeNet(args.dataset_directory, args.class_ids.split(','), 'val')
    model = models.Model(lambda_smoothness=args.lambda_smoothness)
    if os.path.exists(checkpoint_path):
        cp = torch.load(checkpoint_path)
        model.load_state_dict(cp)
        print("load model successfully!")
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for step in range(args.num_iterations):
        images_a, images_b, viewpoints_a, viewpoints_b = dataset_train.get_random_batch(args.batch_size)
        images_a = images_a.to(device)
        images_b = images_b.to(device)
        viewpoints_a = viewpoints_a.to(device)
        viewpoints_b = viewpoints_b.to(device)

        model.train()
        optim.zero_grad()
        loss, loss_sil, loss_smooth = model(images_a, images_b, viewpoints_a, viewpoints_b)
        loss.backward()
        optim.step()

        print("[step {}] loss:{:.8f}, loss_sil:{:.8f}, loss_smooth:{:.8f}".format(
                step,
                loss.detach().cpu().item(),
                loss_sil.detach().cpu().item(),
                loss_smooth.detach().cpu().item()
            ))
        if step % args.log_interval == 0:
            eval_impl(model, dataset_val, step, output_directory)
            # checkpoint_path = os.path.join(output_directory, f'checkpoint{step:06}.pth')
            state_dict = model.state_dict()
            torch.save(state_dict, checkpoint_path)


if __name__ == "__main__":
    # train()
    # eval()
    gen_shape()
