import torch
import torch.nn.functional as F
from torch import nn, autograd
import collections
import numpy as np

# model utils file
# =======================================
# TODO:
# =======================================

@torch.no_grad()
def get_features(data_loader, model, args, tag=1):
    # switch to evaluate mode
    # if args.ngpu > 1:
    #     model = model.module
    model.eval()
    features_list = []
    targets_list = []
    for i, (input, target) in enumerate(data_loader):
        if i%10==0:
            print(i, end=' ', flush=True)
        tag_input = (torch.ones(input.size()[0],1)*tag).to(args.device)
        input = torch.autograd.Variable(input, requires_grad=False).to(args.device)
        # compute output
        features, _ = model.features(input, tag_input)
        features = F.normalize(features)
        features_list.append(features.reshape(input.size()[0],-1))
        targets_list.append(target)
    features_list_tensor = torch.cat(features_list, dim=0)
    targets_list_tensor = torch.cat(targets_list, dim=0)
    print('Features ready: {}, {}'.format(features_list_tensor.shape, targets_list_tensor.shape))
    return features_list_tensor, _, targets_list_tensor

def freeze_BN(m):
    """freeze the Batch Norm layer"""
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()

@torch.no_grad()
def get_memory_features(train_loader, train_loader_ext, model, args):
    im_features, _, im_gt_labels = get_features(train_loader_ext, model, args=args)  # numpy
    sk_features, _, sk_gt_labels = get_features(train_loader, model, args=args)  # numpy
    sk_features  = torch.from_numpy(sk_features).type(torch.Tensor).to(args.device)
    im_features = torch.from_numpy(im_features).type(torch.Tensor).to(args.device)
    ma_features = torch.cat([sk_features, im_features], dim=0)
    ma_labels = np.concatenate([sk_gt_labels, im_gt_labels], axis=0)
    ma_centers = generate_cluster_features(ma_labels, ma_features)
    return ma_centers

@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers


class CM_Mean(autograd.Function):

    @staticmethod
    # ctx在这里类似self，ctx的属性可以在backward中调用。
    # 自己定义的Function中的forward()方法，所有的Variable参数将会转成tensor！因此这里的input也是tensor．在传入forward前，autograd engine会自动将Variable unpack成Tensor。
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)  # 将Tensor转变为Variable保存到ctx中
        outputs = inputs.mm(ctx.features.t())  # input * features.T

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        # grad_output为反向传播上一级计算得到的梯度值
        inputs, targets = ctx.saved_tensors  # 分别代表前向过程存储的值
        grad_inputs = None
        if ctx.needs_input_grad[0]:  # input是否需要求梯度
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)
        for index, features in batch_centers.items():
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * torch.stack(features).mean(dim=0)
            ctx.features[index] /= ctx.features[index].norm()
        return grad_inputs, None, None, None


def cm_mean(inputs, indexes, features, momentum=0.2):
    # 调用forward,返回 inputs * features.T
    return CM_Mean.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None
    
def cm_hard(inputs, indexes, features, momentum=0.2):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module):
    """
    args:
        num_features:
        num_samples:
        temp: 
        momentum:
    """
    def __init__(self, num_features, num_samples, temp=0.2, momentum=0.2, use_hard=False) -> None:
        super(ClusterMemory, self).__init__()
        self.num_feature = num_features
        self.num_sample = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.centers = None
        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):
        """
        args:
            inputs:
            targets:
        """
        inputs = F.normalize(inputs, dim=1).to(inputs.device)  # 
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm_mean(inputs, targets, self.features, self.momentum)  # 一次前向过程
        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss