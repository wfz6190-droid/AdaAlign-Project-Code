from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models
from .senet import cse_resnet50, cse_resnet50_hashing
from modeling.vision_transformer import vit_small, vit_base
from collections import OrderedDict
import torch.nn.functional as F



class SemanticMap(nn.Module):
    """Map the feature to clip space"""
    def __init__(self, in_feature=2048, out_feature=1024):
        super(SemanticMap, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.map =  nn.Sequential(
            nn.Linear(self.in_feature, self.out_feature, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(self.out_feature, self.out_feature, bias=True),
        )
    def forward(self, inputs):
        out = self.map(inputs)  # 2048 --> 1024
        return out

class SemanticMapConv(nn.Module):
    """Map the feature to clip space"""
    def __init__(self, in_feature=2048, out_feature=1024):
        super(SemanticMapConv, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.map =  nn.Sequential(
            nn.Conv2d(self.in_feature, self.out_feature, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.in_feature, self.out_feature, kernel_size=1, stride=1, padding=0, bias=True),
        )
    def forward(self, inputs):
        out = self.map(inputs)  # 2048 --> 1024
        N = out.size(0)
        return self.avg_pool(out).view(N, -1)


class ResnetModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False):
        super(ResnetModel, self).__init__()
        
        self.num_classes = num_classes
        self.modelName = arch
        
        original_model = models.__dict__[arch](pretrained=pretrained)
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])
        self.linear = nn.Linear(in_features=2048, out_features=num_classes)
        
        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False
                    
    def forward(self, x):
        out = self.features(x)
        out = self.last_block(out)
        out = out.view(out.size()[0],-1)
        out = self.linear(out)
        return out
      
    
class CSEResnetModel_KD(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False):
        super(CSEResnetModel_KD, self).__init__()
        self.num_classes = num_classes
        self.modelName = arch
        if pretrained:
            self.original_model = cse_resnet50()  # no ems
        else:
            self.original_model = cse_resnet50(pretrained=None)

        self.feature_size = 2048
        self.semantic_map = SemanticMap(self.feature_size)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.feature_size, self.num_classes)
        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False
                    
      
    def forward(self, x, y):
        out_feature = self.original_model.features(x,y)
        out_kd = self.original_model.logits(out_feature)
        out_feature = self.avgpool(out_feature)  # output [N, C, 1, 1]
        out_feature = out_feature.view(out_feature.size()[0],-1)  # [N, C]
        clip_sapce_feture = self.semantic_map(out_feature)  # [N, 512]

        return clip_sapce_feture, out_kd

    
class SherryCSEResnet(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, clip_feature,  pretrained=True, add_adapter=False):
        super(SherryCSEResnet, self).__init__()
        
        self.hashing_dim = hashing_dim
        self.num_classes = num_classes

        if pretrained:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, add_adapter=add_adapter)
        else:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, pretrained=None, add_adapter=add_adapter)
        
        self.feature_size = 2048
        self.clip_feature = clip_feature
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.semantic_map = SemanticMap(self.feature_size, self.clip_feature)  # 2048 --> 1024
        # self.linear = nn.Linear(in_features=self.clip_feature, out_features=num_classes)
                    
    def forward(self, x, y):
        # features --> hashing --> EMSLayer and logits(last linear)
        out_feature = self.original_model.features(x,y)  # dim: [N, C, 7, 7]
        out_hashing = self.original_model.hashing(out_feature)  # [N, C, 7, 7] --> [N, hashing_dim]
        out_feature = self.avgpool(out_feature).view(out_feature.size(0), -1)  # # [N, C, 7, 7] --> [N, C]
        clip_space_feature = self.semantic_map(out_feature)
        
        out_kd = self.original_model.logits(out_hashing)  # hashing_dim --> 1000
        # out = self.linear(clip_space_feature)  # [N, clip_feature] -- > [N, n_cls]
        return clip_space_feature, out_kd
    
    def features(self, x, y):
        out_feature = self.original_model.features(x,y)  # dim: [N, C, 7, 7]
        out_hashing = self.original_model.hashing(out_feature)  # [N, C, 7, 7] --> [N, hashing_dim]
        out_feature = self.avgpool(out_feature).view(out_feature.size(0), -1)  # # [N, C, 7, 7] --> [N, C]
        return out_feature, out_hashing
    
    def get_clip_features(self, x, y):
        # features --> hashing --> EMSLayer and logits(last linear)
        out_feature = self.original_model.features(x,y)  # dim: [N, C, 7, 7]
        out_hashing = self.original_model.hashing(out_feature)  # [N, C, 7, 7] --> [N, hashing_dim]
        out_feature = self.avgpool(out_feature).view(out_feature.size(0), -1)  # # [N, C, 7, 7] --> [N, C]
        clip_space_feature = self.semantic_map(out_feature)
        return clip_space_feature, out_hashing
        
class SherryDINO(nn.Module):
    """based on facebookresearch/dino
    Paper:https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html
    Code:https://github.com/facebookresearch/dino
    """
    def __init__(self, args, hashing_dim=64, is_teacher=False, add_adapter=False, prompt_learning=False):
        super(SherryDINO, self).__init__()
        arch = args.arch[4:]
        # model_pretrained_path = f'checkpoints/dino_vitbase16_pretrain.pth'
        # linear_pretrained_path = f'checkpoints/dino_vitbase16_linearweights.pth'
        model_pretrained_path = f'checkpoints/dino_deitsmall8_pretrain.pth'
        linear_pretrained_path = f'checkpoints/dino_deitsmall8_linearweights.pth'
        self.feature_size = 384  # for small model
        # self.feature_size = 768  # for base model
        self.is_teacher = is_teacher
        self.clip_feature = args.clip_feature
        # self.original_model = vit_base(patch_size=16, num_classes=0, add_adapter=add_adapter, prompt_learning=prompt_learning)
        self.original_model = vit_small(patch_size=8, num_classes=0, add_adapter=add_adapter, prompt_learning=prompt_learning)
        self.expansion = 4  # for small model
        # self.expansion = 2  # for base model
        self.emb_dim = self.original_model.embed_dim * self.expansion
        self.original_classifier = nn.Linear(self.emb_dim, 1000)
        # load pre-trained weighted
        self.original_model.load_state_dict(torch.load(model_pretrained_path), strict=False)
        new_state_dict = OrderedDict()
        for key, value in torch.load(linear_pretrained_path)['state_dict'].items():
            new_state_dict[key[14:]] = value  # filter the 'module.linear.'
        self.original_classifier.load_state_dict(new_state_dict)
        self.second_last_linear = nn.Linear(self.feature_size, hashing_dim)
        self.semantic_map = SemanticMap(hashing_dim, self.clip_feature)
        # self.linear = nn.Linear(in_features=self.clip_feature, out_features=args.num_classes)
        if self.is_teacher:
            for pp in self.original_model.parameters():
                pp.requires_grad_(False)
                        
    def forward(self, x, tag=None):
        intermediate_output = self.original_model.get_intermediate_layers(x, self.expansion)  # from last blocks
        hidden_state_nblock = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        logits_kd = self.original_classifier(hidden_state_nblock)
        hash_out = self.second_last_linear(intermediate_output[-1][:, 0])
        clip_space_feature = self.semantic_map(hash_out)
        # out_logits = self.linear(clip_space_feature)
        if not self.is_teacher:
            return clip_space_feature, logits_kd
        else:
            return logits_kd

    def features(self, x, tag=None):
        hidden_states = self.original_model(x)
        feature_hashing = self.second_last_linear(hidden_states)
        return hidden_states, feature_hashing

