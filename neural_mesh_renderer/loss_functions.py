import torch
import numpy as np

def smoothness_loss_parameters(faces):
    faces = faces.detach().cpu().numpy()
    vertices = np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0)
    vertices = list(set([tuple(v) for v in np.sort(vertices)]))

    v0s = np.array([v[0] for v in vertices], dtype=np.int32)
    v1s = np.array([v[1] for v in vertices], dtype=np.int32)
    v2s = []
    v3s = []
    for v0, v1 in zip(v0s, v1s):
        count = 0
        for face in faces:
            if v0 in face and v1 in face:
                v = np.copy(face)
                v = v[v!=v0]
                v = v[v!=v1]
                if count == 0:
                    v2s.append(int(v[0]))
                    count += 1
                else:
                    v3s.append(int(v[0]))
    v2s = np.array(v2s, np.int32)
    v3s = np.array(v3s, np.int32)
    return v0s, v1s, v2s, v3s

def iou(data1, data2):
    batch_size = data1.shape[0]
    axes = tuple(range(data1.dim())[1:])
    intersection = torch.sum(data1*data2, dim=axes)
    union = torch.sum(data1 + data2 - data1 * data2, dim=axes)
    return torch.sum(intersection/union) / batch_size

def iou_loss(data1, data2):
    return 1 - iou(data1, data2)


def smoothness_loss(vertices, parameters, eps=1e-6):
    # make v0s, v1s, v2s, v3s
    v0s, v1s, v2s, v3s = parameters
    batch_size = vertices.shape[0]

    # compute angle of two adjacent triangles
    # notations are illustrated in https://github.com/hiroharu-kato/mesh_reconstruction/blob/master/data/misc/smooth_loss.png
    # This code computes the angle using vectors c1->b1 and c2->b2

    # get a1 and b2
    # * triangle A: [v0, v1, v2]
    # * triangle B: [v0, v1, v3]
    # v0s.shape == [batch size, num of triangle pairs, 3(xyz)].
    v0s = vertices[:, v0s, :]
    v1s = vertices[:, v1s, :]
    v2s = vertices[:, v2s, :]
    v3s = vertices[:, v3s, :]
    a1 = v1s - v0s
    b1 = v2s - v0s

    # compute dot and cos of a1 and b1
    a1l2 = torch.sum(a1**2, dim=-1)
    b1l2 = torch.sum(b1**2, dim=-1)
    a1l1 = torch.sqrt(a1l2 + eps)
    b1l1 = torch.sqrt(b1l2 + eps)
    ab1 = torch.sum(a1 * b1, dim=-1)
    cos1 = ab1 / (a1l1 * b1l1 + eps)
    sin1 = torch.sqrt(1 - cos1**2 + eps)

    # c1 = (a1/|a1|) * (|b1|*cos) = a1 * dot(a1, b1) / |a1|^2
    c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None].expand_as(a1)

    # vector of c1->b1, and its length
    cb1 = b1 - c1
    cb1l1 = b1l1 * sin1

    # same computation for triangle B
    a2 = v1s - v0s
    b2 = v3s - v0s
    a2l2 = torch.sum(a2**2, dim=-1)
    b2l2 = torch.sum(b2**2, dim=-1)
    a2l1 = torch.sqrt(a2l2 + eps)
    b2l1 = torch.sqrt(b2l2 + eps)
    ab2 = torch.sum(a2 * b2, dim=-1)
    cos2 = ab2 / (a2l1 * b2l1 + eps)
    sin2 = torch.sqrt(1 - cos2**2 + eps)
    c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None].expand_as(a2)
    cb2 = b2 - c2
    cb2l1 = b2l1 * sin2

    # cos of c1b1 and c2b2
    cos = torch.sum(cb1 * cb2, dim=-1) / (cb1l1 * cb2l1 + eps)

    loss = torch.sum((cos + 1)**2) / batch_size
    return loss