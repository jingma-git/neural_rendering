import torch

# bias = torch.randn((64, 642, 3))
# x = torch.randn((642, 3))[None, :, :].expand_as(bias)
# y = torch.sum(bias, dim=-1)
# print(y.shape)
# torch.repeat()

# import numpy as np
# a=np.array([[1, 2, 3], [4, 5, 6]])
# np.savez_compressed("a.npz", a)
#
# a_ = np.load("a.npz")
# print(a_['arr_0'])


# from datasets import ShapeNet
# dataset_directory = "./data/shapenet/mesh_reconstruction"
# class_ids = ['03001627']
# dataset_train = ShapeNet(dataset_directory, class_ids, 'train')
# dataset_val = ShapeNet(dataset_directory, class_ids, 'val')
#
# dataset_train.statistics()
# dataset_val.statistics()


import numpy as np
x = torch.linspace(0, 360, 24)
print(x)