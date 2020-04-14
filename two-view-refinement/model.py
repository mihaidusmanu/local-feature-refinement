import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16


class PANet(nn.Module):
    def __init__(self, model_path='two-view-refinement/checkpoint.pth'):
        super(PANet, self).__init__()

        vgg = list(vgg16(pretrained=False).features.children())
        vgg_layers_block1 = vgg[: 4]
        vgg_layers_block2 = vgg[5 : 9]
        self.backbone = nn.Sequential(
            *vgg_layers_block1,
            nn.MaxPool2d(3, stride=2, padding=1),
            *vgg_layers_block2
        )

        input_image_size = 33
        feature_map_size = input_image_size // 2 + 1

        self.refine_net = nn.Sequential(
            nn.Conv2d(feature_map_size * feature_map_size, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.predict_net = nn.Sequential(
            nn.Linear((feature_map_size - 16) * (feature_map_size - 16) * 64, 2)
        )

        self.load_state_dict(torch.load(model_path)['model'])

        self.eval()

    def normalize_batch(self, images):
        device = images.device
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float).view(1, 3, 1, 1).to(device)
        return (images.float() / 255. - mean) / std

    def forward(self, reference_batch, target_batch):
        b = reference_batch.size(0)

        batch = torch.cat([reference_batch, target_batch], dim=0)

        features = F.normalize(self.backbone(batch), dim=1)
        _, c, h, w = features.size()

        reference_features = features[: b]
        target_features = features[b :]

        correlation = (
            reference_features.view(-1, c, h * w).permute([0, 2, 1]) @
            target_features.view(-1, c, h * w)
        ).view(-1, h, w, h, w)

        # 1 -> 2
        corr12 = F.normalize(F.relu(correlation.view(-1, h, w, h * w).permute([0, 3, 1, 2])), dim=1).contiguous()

        # Refinement.
        refined_correspondences = self.refine_net(corr12)

        # Prediction.
        displacements = self.predict_net(refined_correspondences.view(b, -1))

        return displacements

    def forward_sym(self, reference_batch, target_batch):
        b = reference_batch.size(0)

        batch = torch.cat([reference_batch, target_batch], dim=0)

        features = F.normalize(self.backbone(batch), dim=1)
        _, c, h, w = features.size()

        reference_features = features[: b]
        target_features = features[b :]

        correlation = (
            reference_features.view(-1, c, h * w).permute([0, 2, 1]) @
            target_features.view(-1, c, h * w)
        )

        # 1 -> 2
        corr12 = F.normalize(F.relu(correlation.view(-1, h, w, h * w).permute([0, 3, 1, 2])), dim=1).contiguous()

        # 2 -> 1
        corr21 = F.normalize(F.relu(correlation.view(-1, h * w, h, w)), dim=1).contiguous()

        # Refinement.
        refined_correspondences = self.refine_net(torch.cat([corr12, corr21], dim=0))

        # Prediction.
        displacements = self.predict_net(refined_correspondences.view(2 * b, -1))

        return displacements[: b], displacements[b :]
