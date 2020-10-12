import torch.nn as nn
import torch
import torch.nn.functional as F
import neural_renderer as nr
import loss_functions

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim=64, dim2=1024):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        # ToDO: add batch_norm
        self.conv1 = nn.Conv2d(dim_in, dim, kernel_size=5, stride=2, padding=2) # 64
        self.conv2 = nn.Conv2d(dim, dim * 2, kernel_size=5, stride=2, padding=2) # 128
        self.conv3 = nn.Conv2d(dim * 2, dim * 2 * 2, kernel_size=5, stride=2, padding=2)  # 256 x 64 x 64
        self.linear1 = nn.Linear((dim*2*2)*8*8, dim2)
        self.linear2 = nn.Linear(dim2, dim2)
        self.linear3 = nn.Linear(dim2, dim_out)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0,
            centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()
        self.vertices_base, self.faces = nr.load_obj(filename_obj)
        self.vertices_base = self.vertices_base.to(device)
        self.faces = self.faces.to(device)

        self.num_vertices = self.vertices_base.shape[0]
        self.num_faces = self.faces.shape[0] # ToDO: add centroid_lr and bias_lr
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim * 2]
        self.linear1 = nn.Linear(dim_in, dim_hidden[0])
        self.linear2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.linear_centroids = nn.Linear(dim_hidden[1], 3)
        self.linear_bias = nn.Linear(dim_hidden[1], self.num_vertices * 3)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        centroids = self.linear_centroids(h) * self.centroid_scale
        bias = self.linear_bias(h) * self.bias_scale
        bias = bias.reshape((-1, self.num_vertices,  3))

        base = self.vertices_base * self.obj_scale
        base = base[None, :, :].expand_as(bias)

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1-base))

        centroids = centroids[:, None, :].expand_as(bias)
        centroids = torch.tanh(centroids)
        scale_pos = (1-centroids)
        scale_neg = centroids + 1

        vertices = torch.sigmoid(base + bias)
        vertices = vertices * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices += centroids
        vertices *= 0.5
        faces = self.faces[None, :, :].repeat(x.shape[0], 1, 1)
        return vertices, faces


class Model(nn.Module):
    def __init__(self, filename_obj='./data/obj/sphere_642.obj', lambda_smoothness=0.):
        super(Model, self).__init__()
        self.lambda_smoothness = lambda_smoothness
        self.vertices_predicted_a = None
        self.vertices_predicted_b = None

        self.encoder = Encoder()
        self.decoder = Decoder(filename_obj)

        self.smoothness_loss_parameters = loss_functions.smoothness_loss_parameters(self.decoder.faces)
        self.renderer = nr.Renderer(camera_mode='look_at')
        self.renderer.image_size = 64
        self.renderer.viewing_angle = 15.
        self.renderer.anti_aliasing = True


    def predict(self, images_a, images_b, viewpoints_a, viewpoints_b):
        batch_size = images_a.shape[0]
        viewpoints = torch.cat((viewpoints_a, viewpoints_a, viewpoints_b, viewpoints_b), dim=0)
        images = torch.cat([images_a, images_b], dim=0)
        self.renderer.eye = viewpoints

        features = self.encoder(images)
        vertices, faces = self.decoder(features)  # [a, b]

        vertices_c = torch.cat((vertices, vertices), dim=0)  # [a, b, a, b]
        faces_c = torch.cat((faces, faces), dim=0)
        silhouettes = self.renderer.render_silhouettes(vertices_c, faces_c)  # [a/a, b/a, a/b, b/b]
        silhouettes_a_a = silhouettes[0 * batch_size:1 * batch_size]
        silhouettes_b_a = silhouettes[1 * batch_size:2 * batch_size]
        silhouettes_a_b = silhouettes[2 * batch_size:3 * batch_size]
        silhouettes_b_b = silhouettes[3 * batch_size:4 * batch_size]
        vertices_a = vertices[:batch_size]
        vertices_b = vertices[batch_size:]
        return silhouettes_a_a, silhouettes_b_a, silhouettes_a_b, silhouettes_b_b, vertices_a, vertices_b


    def forward(self, images_a, images_b, viewpoints_a, viewpoints_b):
        silhouettes_a_a, silhouettes_b_a, silhouettes_a_b, silhouettes_b_b, vertices_a, vertices_b \
            = self.predict(images_a, images_b, viewpoints_a, viewpoints_b)

        # compute loss
        loss_silhouettes = (
                               loss_functions.iou_loss(images_a[:, 3, :, :], silhouettes_a_a) +
                               loss_functions.iou_loss(images_a[:, 3, :, :], silhouettes_b_a) +
                               loss_functions.iou_loss(images_b[:, 3, :, :], silhouettes_a_b) +
                               loss_functions.iou_loss(images_b[:, 3, :, :], silhouettes_b_b)) / 4

        if self.lambda_smoothness != 0:
            loss_smoothness = (
                                  loss_functions.smoothness_loss(vertices_a, self.smoothness_loss_parameters) +
                                  loss_functions.smoothness_loss(vertices_b, self.smoothness_loss_parameters)) / 2
        else:
            loss_smoothness = 0
        loss = loss_silhouettes + self.lambda_smoothness * loss_smoothness

        return loss, loss_silhouettes, loss_smoothness

    def reconstruct(self, images):
        features = self.encoder(images)
        vertices, faces = self.decoder(features)
        return vertices, faces

    def gen_shape(self, viewpoints):
        self.renderer.eye = viewpoints
        features = torch.randn((viewpoints.shape[0], 512)).to(device)
        vertices, faces = self.decoder(features)
        texture_size = 2
        textures = torch.ones(24, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(
            device)
        images_out = self.renderer.render(vertices, faces, textures)
        return images_out


    def reconstruct_and_render(self, images_in, viewpoints):
        self.renderer.eye = viewpoints
        vertices, faces = self.reconstruct(images_in)
        texture_size = 2
        textures = torch.ones(24, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(device)
        images_out = self.renderer.render(vertices, faces, textures)
        return images_out


