import os
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import datetime
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import warnings
from options import Options
from utils import evaluate, dataset, tools
from utils.model import get_features
from modeling.model import SherryCSEResnet, SherryDINO
import json

warnings.filterwarnings("default")


def main():
    args = Options().parse()
    args = tools.merge_args(args)
    tools.random_seed(args.SEED)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    ################### SBIR
    feature_file = os.path.join(args.log_dir, args.file_name, 'sbir_features.pth')
    if os.path.exists(feature_file):
        print('load saved SBIR features')
        features_gallery, features_query, gt_labels_gallery, gt_labels_query = torch.load(feature_file)
        # features_gallery, features_query, gt_labels_gallery, gt_labels_query = res_dict["features_gallery"], res_dict["features_query"], res_dict["gt_labels_gallery"], res_dict["gt_labels_query"]
    else:
        print('prepare SBIR features using saved model')
        features_gallery, features_query, gt_labels_gallery, gt_labels_query = prepare_features(args)

    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    res = _calculate_ret_metric(features_gallery, features_query, gt_labels_gallery, gt_labels_query, device, args)
    for k, v in res.items():
        if k != "top10_retrieved":
            print(f'{k}: {v}')
    out = {}
    fls_sk, fls_im = get_fls(args)
    # fls_sk = fls_sk.tolist()
    # fls_im = fls_im.tolist()
    for k, v in res.items():
        if k != "top10_retrieved": # metric
            out[k] = v
        else:
            top10_retrived_out = {}
            for sketch_idx, top10_retrieved_ori in enumerate(zip(v)):
                sketch_path = fls_sk[sketch_idx]
                top10_retrieved_ori_paths = [fls_im[idx] for idx in top10_retrieved_ori[0]]
                top10_retrived_out[sketch_path] = top10_retrieved_ori_paths
            out["top10_retrived"] = top10_retrived_out

    with open(os.path.join(args.log_dir, args.file_name, "eval_result.json"), "w") as json_file:
        json.dump(out, json_file, indent=4)

    # the fellow function is too slow to run
    # scores = - cdist(features_query, features_gallery, metric='cosine')

def _calculate_ret_metric(features_gallery, features_query, gt_labels_gallery, gt_labels_query, device, args):
    sketch_emb = features_query
    picture_emb = features_gallery
    sketch_cls = gt_labels_query
    picture_cls = gt_labels_gallery

    str_sim = torch.zeros(sketch_emb.size(0), picture_emb.size(0))
    sim_euc = torch.zeros(sketch_emb.size(0), picture_emb.size(0))

    MX = 512
    sketch_emb = sketch_emb.to(device)
    picture_emb = picture_emb.to(device)
    sketch_emb_norm = sketch_emb / sketch_emb.norm(dim=1)[:, None]
    picture_emb_norm = picture_emb / picture_emb.norm(dim=1)[:, None]
    sim_euc_list= []
    for i in range(sketch_cls.size(0)//MX + 1):
        sketch_emb_norm_batch = sketch_emb_norm[i*MX:(i+1)*MX]
        # TODO: OOM here
        # sim_euc_list.append(torch.mm(sketch_emb_norm_batch, picture_emb_norm.T))
        sim_euc_list.append(torch.mm(sketch_emb_norm_batch, picture_emb_norm.T).cpu().numpy())
    del sketch_emb_norm, sketch_emb, picture_emb, picture_emb_norm
    # sim_euc = torch.cat(sim_euc_list, dim=0); del sim_euc_list
    sim_euc = np.concatenate(sim_euc_list, axis=0); del sim_euc_list
    
    str_sim_list = []
    for i in range(sketch_cls.size(0)//MX + 1):
        sketch_cls_batch = sketch_cls[i*MX:(i+1)*MX]
        # str_sim_list.append(((sketch_cls_batch.unsqueeze(1) == picture_cls.unsqueeze(0)) * 1))
        str_sim_list.append(((sketch_cls_batch.unsqueeze(1) == picture_cls.unsqueeze(0)) * 1).cpu().numpy())
    del sketch_cls, picture_cls
    # str_sim = torch.cat(str_sim_list, dim=0); del str_sim_list
    str_sim = np.concatenate(str_sim_list, axis=0); del str_sim_list
    
    map_all, map_200, precision_100, precision_200, topk_photos = evaluate.calculate(np.array(-sim_euc), np.array(str_sim), True if args.dataset != 'quickdraw' else False)
    # map_all, map_200, precision_100, precision_200, topk_photos = evaluate.calculate(np.array(-sim_euc.cpu()), np.array(str_sim.cpu()), True)
    report_dict = {"mAP@all": map_all, "p@100": precision_100, "mAP@200": map_200,"p@200": precision_200,
                   "top10_retrieved": topk_photos}
    
    
    return report_dict

def get_class_dict(file_path):
    """创建key:class_number value:class_name"""
    with open(file_path, 'r') as f:
        file_content = f.readlines()
    d = {str(idx):ff.split('\n')[0] for idx, ff in enumerate(file_content)}
    return d


def get_fls(args):
    if args.dataset == 'sketchy_split1':
        dataset_root = '/data/sydong/datasets/SBIR/Sketchy/'
        fls_sk_file = dataset_root + 'zeroshot1/sketch_tx_000000000000_ready_filelist_zero.txt'
        fls_im_file = dataset_root + '/zeroshot1/all_photo_filelist_zero.txt'

    elif args.dataset == 'sketchy_split2':
        dataset_root = '/data/sydong/datasets/SBIR/Sketchy/'
        fls_sk_file = dataset_root + 'zeroshot2/sketch_tx_000000000000_ready_filelist_zero.txt'
        fls_im_file = dataset_root + 'zeroshot2/all_photo_filelist_zero.txt'

    elif args.dataset == 'tuberlin':
        dataset_root = '/data/sydong/datasets/SBIR/TUBerlin/'
        fls_sk_file = dataset_root + 'zeroshot/png_ready_filelist_zero.txt'
        fls_im_file = dataset_root + 'zeroshot/ImageResized_ready_filelist_zero.txt'

    elif args.dataset == 'quickdraw':
        dataset_root = '/data/sydong/datasets/SBIR/QuickDraw/'
        fls_sk_file = dataset_root + 'zeroshot/sketch_filelist_zero.txt'
        fls_im_file = dataset_root + 'zeroshot/photo_filelist_zero.txt'
    with open(fls_sk_file, 'r') as fh:
        sk_file_content = fh.readlines()
    with open(fls_im_file, 'r') as fh:
        im_file_content = fh.readlines()
    
    fls_sk = np.array([' '.join(ff.strip().split()[:-1]) for ff in sk_file_content])
    fls_im = np.array([' '.join(ff.strip().split()[:-1]) for ff in im_file_content])
    return fls_sk, fls_im


def prepare_features(args):
    print(args.clip_feature)
    if args.arch  == 'cse_resnet50':
        model = SherryCSEResnet(args.arch, args.num_hashing, args.num_classes, args.clip_feature, add_adapter=args.add_adapter)
    elif args.arch  in ['vit_small8', 'vit_base']:
        model = SherryDINO(args, is_teacher=False, add_adapter=args.add_adapter)
    print(str(datetime.datetime.now()) + ' model inited.')
    model = model.to(args.device)

    # resume from a checkpoint
    if args.file_name:
        resume = os.path.join(args.log_dir, args.file_name, 'model_best.pth')
    else:
        resume = os.path.join(args.log_dir, 'model_best.pth')

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        save_dict = checkpoint['state_dict']
        # print(checkpoint['best_map'])
        model_dict = model.state_dict()
        # trash
        trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
        print('trashed vars from resume dict:')
        print(trash_vars)
        resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
        model_dict.update(resume_dict)
        model.load_state_dict(model_dict)
        print("=> loaded checkpoint '{}' (epoch {}), (best map: {})"
              .format(resume, checkpoint['epoch'], checkpoint['best_map']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        return
    # return 
    cudnn.benchmark = True

    # load data
    immean = [0.485, 0.456, 0.406] # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]
    immean_sk = [0.48145466, 0.4578275, 0.40821073]
    imstd_sk = [0.26862954, 0.26130258, 0.27577711]
    transformations_im = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize([224,224]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(immean, imstd)])
    transformations_sk = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize([224,224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(immean_sk, imstd_sk)])

    transformations = {'im':transformations_im, 'sk':transformations_sk}
    zero_loader_ext, zero_loader = dataset.load_dataset_zero(args, transformations=transformations)
    print(str(datetime.datetime.now()) + ' data loaded.')

    features_gallery, features_gallery_hash, gt_labels_gallery = get_features(zero_loader_ext, model, args=args)
    features_query, features_query_hash, gt_labels_query = get_features(zero_loader, model, args=args, tag=0)
    # features_query_bin, features_gallery_bin = evaluate.compressITQ(features_query_hash, features_gallery_hash)
    # scores_bin = - cdist(features_query_bin, features_gallery_bin, metric='hamming')
    # print('hamming distance calculated')

    # 以二进制的方式写入文件
    save_dict = {
        "features_gallery":features_gallery,
        "features_query": features_query,
        "gt_labels_gallery": gt_labels_gallery,
        "gt_labels_query":gt_labels_query
    }
    # torch.save(save_dict, os.path.join(args.log_dir, args.file_name, 'sbir_features.pth'))
    torch.save([features_gallery, features_query, gt_labels_gallery, gt_labels_query], os.path.join(args.log_dir, args.file_name, 'sbir_features.pth'))
    print("saved features finished")
    print("✅ prepare_features() reached the end, ready to return features.")
    return features_gallery, features_query, gt_labels_gallery, gt_labels_query


def prepare_sbsr_features(predicted_features_ext, gt_labels_ext, args):
    query_index = []
    for ll in np.unique(gt_labels_ext):
        query_index.append(np.where(gt_labels_ext==ll)[0][0:100])
        
    query_index = np.concatenate(query_index)
    
    query_index_bool = np.zeros(gt_labels_ext.shape[0]).astype(bool)
    query_index_bool[query_index] = True
    
    predicted_features_query = predicted_features_ext[query_index_bool]
    gt_labels_query = gt_labels_ext[query_index_bool]
    predicted_features_gallery = predicted_features_ext[np.logical_not(query_index_bool)]
    gt_labels_gallery = gt_labels_ext[np.logical_not(query_index_bool)]
    
        
    return predicted_features_gallery, predicted_features_query, gt_labels_gallery, gt_labels_query


if __name__ == '__main__':
    main()