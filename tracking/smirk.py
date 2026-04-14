from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .cropping import CachedImageCropper


class SMIRKWrapper(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.model = SmirkModel().to(device)
        # Cache cropped images to avoid recomputing landmarks and crops during training
        # Use separate caches since train and test frames share ids
        self.cropper_train = CachedImageCropper()
        self.cropper_test = CachedImageCropper()

    def get_tune_parameters(self):
        return [
            *self.model.expression_encoder.encoder.parameters(),
            *self.model.expression_encoder.expression_layers.parameters(),
        ]

    def capture(self):
        return self.state_dict()

    def restore(self, state):
        self.load_state_dict(state)

    def forward(self, views: Dict, is_train: bool):
        cropper = self.cropper_train if is_train else self.cropper_test
        crops = cropper(views["idx"], views["img"], views["landmarks_mediapipe"])

        if crops is None:
            return

        # outputs = self.model(image.permute(0,3,1,2))
        # Batching yields weird results so we do it one by one
        outputs = {}
        for img in crops:
            v = self.model(img.permute(2,0,1).unsqueeze(0))
            for k in v:
                outputs[k] = torch.cat((outputs[k], v[k])) if k in outputs else v[k]

        # from dataset.dataset_util import save_img
        # save_img(f"/bulk/tests/warped_image_{views['idx' ][0].item()}.png", image[0])

        pose = views["flame_pose"]
        pose = torch.cat((
            pose[:, :6],
            outputs["jaw_params"], # (3)
            pose[:, 9:15],
            outputs["eyelid_params"], # (2)
            pose[:, 17:],
        ), dim=1)
        expr = outputs["expression_params"]

        return pose, expr

####################################################################################################
# Smirk Model
####################################################################################################

def create_backbone(backbone_name, pretrained=True,checkpoint_path=None):
    backbone = timm.create_model(backbone_name,
                        pretrained=pretrained,checkpoint_path=checkpoint_path,
                        features_only=True)
    feature_dim = backbone.feature_info[-1]['num_chs']
    return backbone, feature_dim


class ExpressionEncoder(nn.Module):
    def __init__(self, n_exp=50, pretrained=True) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100', pretrained=pretrained)
        self.expression_layers = nn.Sequential(
            nn.Linear(feature_dim, n_exp+2+3) # num expressions + jaw + eyelid
        )

        self.n_exp = n_exp
        self.init_weights()

    def init_weights(self):
        self.expression_layers[-1].weight.data *= 0.1
        self.expression_layers[-1].bias.data *= 0.1

    def forward(self, img):
        features = self.encoder(img)[-1]
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        parameters = self.expression_layers(features).reshape(img.size(0), -1)

        outputs = {}
        outputs['expression_params'] = parameters[...,:self.n_exp]
        outputs['eyelid_params'] = torch.clamp(parameters[...,self.n_exp:self.n_exp+2], 0, 1)
        outputs['jaw_params'] = torch.cat([F.relu(parameters[...,self.n_exp+2].unsqueeze(-1)),
                                           torch.clamp(parameters[...,self.n_exp+3:self.n_exp+5], -.2, .2)], dim=-1)
        outputs["image_feature"]=features
        return outputs

class SmirkModel(nn.Module):
    def __init__(self, exp_dim=50):
        super(SmirkModel, self).__init__()
        self.model_path = "./assets/SMIRK_em1.pt"

        # Since we load SMIRK's own weights, we don't need to load pretrained weights
        self.expression_encoder = ExpressionEncoder(n_exp=exp_dim, pretrained=False)
        self.exp_dim = exp_dim
        self.load_initial_state() 

    def forward(self, img):
        return self.expression_encoder(img)
        
    def load_initial_state(self):
        checkpoint = torch.load(self.model_path)
        checkpoint_expression = {k.replace('smirk_encoder.expression_encoder.', ''): v for k, v in checkpoint.items() \
                                         if 'smirk_encoder.expression_encoder' in k}
        checkpoint_expression_encoder = {k.replace('encoder.', ''): v for k, v in checkpoint_expression.items() \
                                         if 'encoder' in k}
        checkpoint_expression_mlp = {k.replace('expression_layers.', ''): v for k, v in checkpoint_expression.items() \
                                         if 'expression_layers' in k}
        self.expression_encoder.encoder.load_state_dict(checkpoint_expression_encoder)
        assert self.exp_dim == 50, f"Wrong exp_dim for using SMIRK (got {self.exp_dim}, expected 50)"
        self.expression_encoder.expression_layers.load_state_dict(checkpoint_expression_mlp)
