import os
import tqdm
import numpy as np
import cv2
import neural_renderer
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ShapeNet(object):
    def __init__(self, directory=None, class_ids=None, set_name=None):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732

        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            img_path = os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))
            images.append(np.load(img_path)['arr_0'])
            # for view_id in range(24):
            #     img = images[-1][0, view_id, :3, :, :].transpose((1, 2, 0))
            #     mask = images[-1][0, view_id, -1, :, :]
            #     cv2.imshow("img", img)
            #     cv2.imshow("mask", mask)
            #     cv2.waitKey(-1)
            voxel_path = os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))
            voxels.append(np.load(voxel_path)['arr_0'])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
        images = np.concatenate(images, axis=0).reshape((-1, 4, 64, 64)) # object x # view
        images = np.ascontiguousarray(images)
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        del images
        del voxels


    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = np.zeros(batch_size, 'float32')
        viewpoint_ids_b = np.zeros(batch_size, 'float32')
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])
            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        images_a = self.images[data_ids_a].astype('float32') / 255.
        images_b = self.images[data_ids_b].astype('float32') / 255.
        images_a = torch.tensor(images_a).to(device)
        images_b = torch.tensor(images_b).to(device)

        distances = np.ones(batch_size, 'float32') * self.distance
        elevations = np.ones(batch_size, 'float32') * self.elevation
        distances = torch.tensor(distances).to(device)
        elevations = torch.tensor(elevations).to(device)
        viewpoint_ids_a = torch.tensor(viewpoint_ids_a).to(device)
        viewpoint_ids_b = torch.tensor(viewpoint_ids_b).to(device)
        viewpoints_a = neural_renderer.get_points_from_angles(distances, elevations, -viewpoint_ids_a * 15.0)
        viewpoints_b = neural_renderer.get_points_from_angles(distances, elevations, -viewpoint_ids_b * 15.0)

        return images_a, images_b, viewpoints_a, viewpoints_b

    def statistics(self):
        print("dataset: ", self.set_name)
        for k, v in self.num_data.items():
            print(k, v)

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids
        for i in range(int((data_ids.size - 1) / batch_size) + 1):
            images = self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.
            voxels = self.voxels[(data_ids[i * batch_size:(i + 1) * batch_size] / 24).astype(np.int)]
            images = torch.tensor(images).to(device)
            voxels = torch.tensor(voxels).to(device)
            yield images, voxels