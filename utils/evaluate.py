from scipy.spatial.distance import cdist
import os
import torch.nn.functional as F
from PIL import Image
import numpy as np
import utils.model as mutils

import time
import numpy as np
import torch
import multiprocessing
from joblib import delayed, Parallel
from sklearn.metrics import average_precision_score


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 判断是不是相等
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test_map(im_loader, sk_loader, epoch, model, args):
    """每个epoch后计算map
    不经过ITQ算法
    """
    features_gallery, _, gt_labels_gallery = mutils.get_features(im_loader, model, args=args)
    features_query, _, gt_labels_query = mutils.get_features(sk_loader, model, args=args, tag=0)

    # TODO: split the block
    start = time.time()
    MX = 512
    sim_euc_list = []
    for i in range(gt_labels_query.size(0)//MX + 1):
        sketch_emb_norm_batch = features_query[i*MX:(i+1)*MX]
        sim_euc_list.append(torch.mm(sketch_emb_norm_batch, features_gallery.T))
    # del sketch_emb_norm, sketch_emb, picture_emb, picture_emb_norm
    sim_euc = torch.cat(sim_euc_list, dim=0); del sim_euc_list
    scores = sim_euc.cpu().numpy()
    time_cp = time.time() - start
    print(f"process sim_euc {time_cp:.3f}")

    gt_labels_gallery = gt_labels_gallery.cpu().numpy()
    gt_labels_query = gt_labels_query.cpu().numpy()


    # the follow compute scores is too slow
    # scores = - cdist(features_query.cpu(), features_gallery.cpu())  # Negative distance matrix (distance of each sample)
    
    # the following code is to calculate the mAP, but is too slow
    # TODO:  multi-process
    single_process_start = time.time()
    
    mAP_ = 0.0
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    
    for fi in range(features_query.shape[0]):  # N个 
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=args.top)  # gt_labels_query：(12994, )
        mAP_ls[gt_labels_query[fi]].append(mapi)
    single_process_end = time.time()
    print(f"single-process time: {(single_process_end-single_process_start):.2f}")

    # mAP_ = 0.0
    # mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    # milti_process_start = time.time()
    # num_cores = min(multiprocessing.cpu_count(), 4)

    # def process_query(fi):
    #     mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=args.top)
    #     return gt_labels_query[fi], mapi

    # results = Parallel(n_jobs=num_cores)(delayed(process_query)(fi) for fi in range(features_query.shape[0]))

    # for label, mapi in results:
    #     mAP_ls[label].append(mapi)

    # milti_process_end = time.time()
    # print(f"multi-process time: {(milti_process_end-milti_process_start):.2f}")

    for _, mAPs in enumerate(mAP_ls):
        mAP_ += np.nanmean(mAPs)
    map_valid = mAP_ / (len(mAP_ls))
    print('Epoch: [{}/{}] \t validate map: {:.4f}'.format(epoch + 1, args.epochs, map_valid))
    with open(os.path.join(args.log_dir, args.file_name, 'validation_log.txt'), 'a') as f:
        f.writelines('Epoch: [{}/{}] \t validate map: {:.4f} \n'.format(epoch + 1, args.epochs, map_valid))
    return map_valid



def ITQ(V, iters=150):
    """
    Main function for  ITQ which finds a rotation of the PCA embedded data
    Input:
        V: nxc PCA embedded data, n is the number of images and c is the code length
        n_iter: max number of iterations, 50 is usually enough
    Output:
        B: nxc binary matrix
        R: the ccc rotation matrix found by ITQ
    Publications:
        Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A
        Procrustes Approach to Learning Binary Codes. In CVPR 2011.
    Initialize with a orthogonal randomion in rotatitialize with a orthogonal random rotation
    """
    bit = V.shape[1]
    np.random.seed(0)
    R = np.random.randn(bit, bit)
    U11, _, _ = np.linalg.svd(R)  # SVD
    R = U11[:, :bit]  # rotation matrix
    #  ITQ to find optimal rotation
    for _ in range(iters):
        Z = np.matmul(V, R)
        UX = np.ones(Z.shape) * -1  # element wise product
        UX[Z >= 0] = 1 
        C = np.matmul(UX.T, V)
        UB, _, UA = np.linalg.svd(C)
        R = np.matmul(UA, UB.T)
    B = UX
    B[B < 0] = 0
    return B, R


def compressITQ(Xtrain, Xtest, n_iter=50):
    """
    compressITQ runs ITQ
    Center the data, VERY IMPORTANT
    args:
        Xtrain:
        Xtest:
        n_iter:
    """
    Xtrain = Xtrain - np.mean(Xtrain, axis=0, keepdims=True)
    Xtest = Xtest - np.mean(Xtest, axis=0, keepdims=True)
    # PCA
    C = np.cov(Xtrain, rowvar=False)  # covariance
    l, pc = np.linalg.eigh(C, 'U')  # Returns the eigenvalue eigenvector
    idx = l.argsort()[::-1]  # Returns the position index of the feature value from large to small
    pc = pc[:, idx]  # Get non-zero eigenvectors
    XXtrain = np.matmul(Xtrain, pc)  # PCA
    XXtest = np.matmul(Xtest, pc)  # PCA
    # ITQ
    _, R = ITQ(XXtrain, n_iter)
    Ctrain = np.matmul(XXtrain, R)  # rotation
    Ctest = np.matmul(XXtest, R)  # rotation
    # bool
    # Ctrain = Ctrain > 0
    # Ctest = Ctest > 0

    # bool to 0, 1
    indxs = Ctrain > 0
    Ctrain[indxs] = 1
    Ctrain[~indxs] = 0

    indxs = Ctest> 0
    Ctest[indxs] = 1
    Ctest[~indxs] = 0
    return Ctrain, Ctest



def calculate(distance, class_same, return_all=False, return_topk=False, out_path=None):
    arg_sort_topk = None
    
    if return_all:
        arg_sort_sim = distance.argsort()   # 得到从小到大索引值
        arg_sort_topk = arg_sort_sim[:, :10]
        sort_label = []
        for index in range(0, arg_sort_sim.shape[0]):
            # 将label重新排序，根据距离的远近，距离越近的排在前面
            sort_label.append(class_same[index, arg_sort_sim[index, :]])
        sort_label = np.array(sort_label)
    else: # quick-draw
        MX=2048
        sort_label = np.zeros_like(distance, dtype=np.int8)

        start_idx_ = 0
        for start_idx in range(0, len(distance), MX):
            end_idx = min(start_idx+MX, len(distance))
            distance_sorted_batch = distance[start_idx:end_idx].argsort() # MX, # of pictures
            if return_topk: 
                torch.save(distance_sorted_batch[:,:10], f"{out_path}/topk_idx_{start_idx_}")
                start_idx_ += 1
            class_same_batch = class_same[start_idx:end_idx] #(92291, 54358)
            sort_label[start_idx:end_idx] = np.take_along_axis(class_same_batch, distance_sorted_batch, axis=1)
            print(f"\t{start_idx} / {len(distance)}")

    # 多进程计算
    num_cores = min(multiprocessing.cpu_count(), 4)

    start = time.time()
    if return_all:
        aps_all = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq]) for iq in range(distance.shape[0]))
        aps_200 = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq], 200) for iq in range(distance.shape[0]))
        map_all = np.nanmean(aps_all)
        map_200 = np.nanmean(aps_200)

        precision_100 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 100) for iq in range(sort_label.shape[0]))
        precision_100 = np.nanmean(precision_100)
        precision_200 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 200) for iq in range(sort_label.shape[0]))
        precision_200 = np.nanmean(precision_200)
        
        # print("eval time:", time.time() - start)
    
    else: # quick-draw
        map_all, map_200, precision_100, precision_200 = None, None, None, None

        aps_all = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq]) for iq in range(distance.shape[0]))
        map_all = np.nanmean(aps_all)
        print(f"aps_all done: {time.time() - start}")
        start = time.time()

        precision_200 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 200) for iq in range(sort_label.shape[0]))
        precision_200 = np.nanmean(precision_200)
        print(f"prec_200 done: {time.time() - start}")


    return map_all, map_200, precision_100, precision_200, arg_sort_topk



def voc_eval(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)
    fp = np.logical_not(tp)
    tot = tp.shape[0]
    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        precision = tp / (tp + fp)
    except:
        print("error", tot_pos)
        return np.nan

    ap = voc_ap(rec, precision)

    return ap

def voc_ap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def precision_eval(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)

    if top is not None:
        top = min(top, tot_pos)
    else:
        top = tot_pos

    return np.mean(sort_class_same[:top])

def eval_AP_inner(inst_id, scores, gt_labels, top='all'):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)  # bool True--1, False--0 total positive
    
    sort_idx = np.argsort(-scores)  # return high -- low indices of scores
    tp = pos_flag[sort_idx]  # bool 
    fp = np.logical_not(tp)  # bool
    
    if top != 'all':
        top = int(top)
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)
    
    fp = np.cumsum(fp)  # truth positive cumsum 
    tp = np.cumsum(tp)  # false positive cumsum
    try:
        rec = tp / tot_pos  # recall  truth positive / total positive
        prec = tp / (tp + fp)  # precision
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)  # Calculate the ap of a single query
    return ap

def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)
    
    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)
    
    for ii in range(len(mpre)-2,-1,-1):
        mpre[ii] = max(mpre[ii], mpre[ii+1])
        
    msk = [i!=j for i,j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk]-mrec[0:-1][msk])*mpre[1:][msk])
    return ap


# def eval_precision(inst_id, scores, gt_labels, top='all'):
#     if top != 'all':
#         top = int(top)
#     else:
#         top = 100
#     pos_flag = gt_labels == inst_id
#     tot = scores.shape[0]  # total
#     top = min(top, tot)
    
#     sort_idx = np.argsort(-scores)
#     return np.sum(pos_flag[sort_idx][:top])/top



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_qualitative_results(aps, length_aps, fls_sk, fls_im, sim, n_q=6, n_sim=10, is_best=True, args=None):
    """
    args:
        aps: ap array
        n_q: n qualitative sample
        n_sim: n simialr sample
    """
    save_root_dir = 'retrievals_results'
    dataset_root_dir = 'dataset'
    if args.dataset in ['sketchy_split1', 'sketchy_split2']:
        dataset_root_dir = os.path.join(dataset_root_dir, 'Sketchy')
    elif args.dataset == 'tuberlin':
        dataset_root_dir = os.path.join(dataset_root_dir, 'TUBerlin')
    elif args.dataset == 'quickdraw':
        dataset_root_dir = os.path.join(dataset_root_dir, 'QuickDraw')
    save_image = True
    save_dir = os.path.join(save_root_dir, args.dataset)
    if is_best:
        ind_sk = np.argsort(aps)[:n_q]
    else:
        np.random.seed(1)
        ind_sk = np.random.choice(len(aps), n_q, replace=False)
    file = os.path.join(save_dir, f"Results_{args.file_name}.txt")
    with open(file, "w") as fp:

        for i, isk in enumerate(ind_sk):
            isk = np.sum(length_aps[:isk]) + 1 
            fp.write("{0}, ".format(os.path.join(dataset_root_dir, fls_sk[isk])))
            if save_image:
                sdir_op = os.path.join(save_dir, f"{args.file_name}_" + str(i + 1))
                if not os.path.isdir(sdir_op):
                    os.makedirs(sdir_op)
                sk = Image.open(os.path.join(dataset_root_dir, fls_sk[isk])).convert(mode='RGB')
                sk.save(os.path.join(sdir_op, fls_sk[isk].split('/')[0] + '.png'))
            ind_im = np.argsort(-sim[isk])[:n_sim]  # 
            for j, iim in enumerate(ind_im):
                if j < len(ind_im)-1:
                    fp.write("{0} {1}, ".format(fls_im[iim], sim[isk][iim]))
                else:
                    fp.write("{0} {1}".format(fls_im[iim], sim[isk][iim]))
                if save_image:
                    im = Image.open(os.path.join(dataset_root_dir, fls_im[iim])).convert(mode='RGB')
                    im.save(os.path.join(sdir_op, str(j + 1) + '_' + str(sim[isk][iim]) + '.png'))
            fp.write("\n")