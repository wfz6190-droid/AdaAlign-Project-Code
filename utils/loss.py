import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def compute_logits(visaual_features, text_features, norm=True, comman_modality=True):
    """normalization the input features to get the cosine similarity logits"""
    if norm is True:
        visaual_features =  F.normalize(visaual_features, dim=-1)
        # visaual_features =  F.normalize(text_features, dim=-1)
    # text_features:(n_class * 2, 1024)
    if not comman_modality:
        num_sample_per_modality = visaual_features.shape[0] // 2
        logits_sk = torch.mm(visaual_features[:num_sample_per_modality], text_features[0].T)
        logits_im = torch.mm(visaual_features[num_sample_per_modality:], text_features[1].T)
        logits =  torch.cat((logits_sk, logits_im), dim=0)
    else:
        logits = torch.mm(visaual_features, text_features.T)
    return logits

def compute_info_nce_probability(scores_matrix, positives_keys):
    """
    querys:
    centers:
    positives_keys:
    """
    mask = torch.zeros(scores_matrix.shape)
    positives_keys = positives_keys.unsqueeze(dim=1)
    src = torch.ones(positives_keys.shape).to(mask.dtype)
    mask = mask.scatter_(dim=1, index=positives_keys, src=src)
    probability = scores_matrix[mask.bool()].reshape(positives_keys.shape)
    return probability

class FeaturesL1Loss(nn.Module):
    def __init__(self):
        super(FeaturesL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, stu_features, teacher_features):
        stu_features = F.normalize(stu_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)
        loss = self.criterion(stu_features, teacher_features)
        return loss


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, input_logits, target_logits, mask=None, mask_pos=None):
        """
        :param input_logits: prediction logits
        :param target_logits: target logits
        :mask_pos: refine the target logits(default: wordnet's output)
        :return: loss
        """
        log_likelihood = - F.log_softmax(input_logits, dim=1)
        
        if mask_pos is not None:
            target_logits = target_logits + mask_pos
        
        if mask is None:
            sample_num, class_num = target_logits.shape
            loss = torch.sum(torch.mul(log_likelihood, F.softmax(target_logits, dim=1))) / sample_num
        else:
            sample_num = torch.sum(mask)
            loss = torch.sum(torch.mul(torch.mul(log_likelihood, F.softmax(target_logits, dim=1)), mask)) / sample_num
        return loss


class SoftKLDistillation(nn.Module):
    """(DeiT) Training data-efficient image transformers & distillation through attention: https://arxiv.org/abs/2012.12877
    Soft Distillition Loss 
    """
    def __init__(self, temperature=0.5) -> None:
        super(SoftKLDistillation, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.temperature = temperature
        self.eps = 1e-10
    
    def forward(self, student_logits, teacher_logits, mask=None):
        """
        args:
            student_logits:
            teacher_logits:
        """
        student_logits = F.softmax(student_logits / (self.temperature + self.eps), dim=1)
        teacher_logits = F.softmax(teacher_logits / (self.temperature + self.eps), dim=1)
        sample_num, _ = student_logits.shape
        if mask is not None:
            sample_num = torch.sum(mask)
            # student_logits = student_logits * mask
            loss = torch.sum((torch.mul(teacher_logits.log(), teacher_logits) - torch.mul(student_logits.log(), teacher_logits)) * mask)
        else:
            loss = torch.sum(torch.mul(teacher_logits.log(), teacher_logits) - torch.mul(student_logits.log(), teacher_logits))
        
        return loss / sample_num
    
class SoftCEDistillation(nn.Module):
    """(DeiT) Training data-efficient image transformers & distillation through attention: https://arxiv.org/abs/2012.12877
    Soft Distillition Loss 
    """
    def __init__(self, temperature=0.5) -> None:
        super(SoftCEDistillation, self).__init__()
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits, mask=None):
        """
        args:
            student_logits:
            teacher_logits:
        """
        log_likelihood = - F.log_softmax(student_logits, dim=-1)
        sample_num, class_num = student_logits.shape
        if mask is None:
            loss = torch.sum(torch.mul(log_likelihood, F.softmax(teacher_logits / self.temperature, dim=-1))) / sample_num
        else:
            sample_num = torch.sum(mask)
            loss = torch.sum(torch.mul(torch.mul(log_likelihood, F.softmax(teacher_logits /self.temperature, dim=1)), mask)) / sample_num
        return loss


class HardCrossEntropy(nn.Module):
    """(DeiT) Training data-efficient image transformers & distillation through attention: https://arxiv.org/abs/2012.12877
    Hard Distillition CrossEntropy Loss for distilltion and label smoothing
    args:
        lable_smooth:(float)
    """
    def __init__(self, lable_smooth:float=0.0) :
        super(HardCrossEntropy, self).__init__()
        self.label_smooth = lable_smooth

    def forward(self, input_logits, target, mask=None):
        """
        input_logits: logits from student
        target: logits from teacher
        """
        target = torch.argmax(target, dim=1, keepdim=False)  # 转换为hard label,行向量
        target_logits = F.one_hot(target, num_classes=input_logits.shape[1])  # 转换为one hot
        if self.label_smooth != 0:
            weight = input_logits.new_ones(input_logits.size()) * self.label_smooth / (input_logits.size(-1) - 1.)
            weight.scatter_(-1, target.unsqueeze(-1), (1. - self.label_smooth))
            target_logits = weight
        log_prob = - F.log_softmax(input_logits, dim=1)
        if mask is None:
            loss = (target_logits * log_prob).sum(dim=-1).mean()
        else:
            sample_num = torch.sum(mask)
            loss = ((target_logits * log_prob).sum(dim=-1) * mask).sum() / sample_num
        return loss


class AMCrossEntropy(nn.Module):
    """AngularMargin CrossEntropy
    Additive Margin Softmax for Face Verification:https://ieeexplore.ieee.org/abstract/document/8331118
    args:
        margin: scalar of cos(angular)
    """
    def __init__(self, margin, temperature=0.05) -> None:
        super(AMCrossEntropy, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, input_logits, target):
        """
        args:
            input_logits: logits from normalized features and weights (Angular)
            target: target(one hot indices)
        """
        target_logits = F.one_hot(target, num_classes=input_logits.shape[1])
        input_logits = input_logits - self.margin * target_logits
        loss = self.criterion(input_logits / self.temperature, target)
        return loss


class AAMCrossEntropy(nn.Module):
    """ AdditiveAngularMargin CrossEntropy
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition: https://arxiv.org/abs/1801.07698
     .. math::
        cos(a+b)=cosa x cosb - sina x sinb
    args:
        margin: scalar of angular default 0.2
        1 --> pi/2
    """
    def __init__(self, margin=0.5, temperature=0.05) -> None:
        super().__init__()
        self.margin = margin
        self.cos_m = math.cos(margin * 2 / math.pi)
        self.sin_m = torch.sqrt(1.0 - torch.pow(self.cos_m, 2))
        # self.th = math.cos(math.pi - self.margin)
        # self.mm = math.sin(math.pi - self.margin) * self.margin
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_logits, target):
        """
        args:
            input_logits: logits from normalized features and weights (Angular)
            target: target(one hot indices)
        """
        cosine = input_logits
        target_logits = F.one_hot(target, num_classes=input_logits.shape[1])  # 转换为onehot
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # sin(angular)
        phi = (cosine * self.cos_m  - sine * self.sin_m) * target_logits + (1 - target_logits) * cosine  # cos(theta + m)
        loss = self.criterion(phi / self.temperature,  target)
        return loss

