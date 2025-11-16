import os
import time
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import datetime
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from peft import LoraConfig, inject_adapter_in_model, get_peft_model

from options import Options
from modeling.senet import cse_resnet50
from modeling.model import SherryCSEResnet, SherryDINO
from modeling.text_encoder import build_text_encoder
from utils import dataset, tools, evaluate, loss
from utils.text_utils import generate_class_embeddings

lora_config = LoraConfig(
    target_modules=".*attn\.(qkv|proj)",
    init_lora_weights=True,
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    # target_modules="all-linear",
)

def count_parameters(model):
    """统计模型的可训练参数和总参数"""
    tunable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param = sum(p.numel() for p in model.parameters())
    return tunable_param, total_param

def save_checkpoint_with_text_encoder(state_dict, filename, text_encoder=None):
    """保存checkpoint，包括文本编码器（如果存在）"""
    checkpoint = state_dict.copy()
    
    # 保存文本编码器状态
    if text_encoder is not None:
        checkpoint['text_encoder_state_dict'] = text_encoder.state_dict()
        checkpoint['has_text_encoder'] = True
        print(f"✓ 保存文本编码器状态到checkpoint")
    else:
        checkpoint['has_text_encoder'] = False
    
    tools.save_checkpoint(checkpoint, filename=filename)

def load_checkpoint_with_text_encoder(checkpoint_path, model_student, optimizer, text_encoder=None):
    """加载checkpoint，包括文本编码器（如果存在）"""
    checkpoint = torch.load(checkpoint_path)
    
    # 加载学生模型
    model_dict = model_student.state_dict()
    resume_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
    model_dict.update(resume_dict)
    model_student.load_state_dict(model_dict)
    
    # 加载优化器
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 加载文本编码器（如果存在）
    if text_encoder is not None and checkpoint.get('has_text_encoder', False):
        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
            print(f"✓ 已加载文本编码器状态")
        else:
            print(f"⚠️  Checkpoint中未找到文本编码器状态")
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_map', 0)

def main():
    args = Options().parse()
    args = tools.merge_args(args)
    tools.random_seed(args.SEED)
    
    # 构建实验文件名
    args.file_name = f'{args.arch}_{args.zero_version}_ep-{args.epochs}_bs-{args.batch_size}_lr-{args.lr}_KD-{args.kd_lambda}_' + \
                     f'CLIP-{args.arch_CLIP}_temperature-{args.temperature}_prompt-{args.prompt}_lr_scale-1_constant_epoch_schedule_3warmpup' 
    if args.add_adapter:
        args.file_name += '_adapter'
    if args.prompt_learning:
        args.file_name += '_prompt_learning'
    if args.lora:
        args.file_name += '_lora'
    if args.text_adapter:
        args.file_name += '_text_adapter'
    if args.text_lora:
        args.file_name += '_text_lora'
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    y_m_d_prefix = datetime.datetime.now().strftime('%Y-%m-%d')
    nowTime = datetime.datetime.now().strftime('%H-%M') 
    args.file_name += f"_{nowTime}"
    args.file_name = y_m_d_prefix + "/" + args.file_name
    args.device = device

    print("=" * 80)
    print(f"实验配置: {args.file_name}")
    print("=" * 80)

    # 创建TensorBoard writer
    writer = SummaryWriter(os.path.join(args.log_dir, args.file_name, 'logs'))
    
    # 数据预处理配置
    immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]
    immean_sk = [0.48145466, 0.4578275, 0.40821073]  # CLIP normalization
    imstd_sk = [0.26862954, 0.26130258, 0.27577711]
    
    transformations_im = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(immean, imstd)
    ])
    transformations_sk = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(immean_sk, imstd_sk)
    ])
    transformations = {'im': transformations_im, 'sk': transformations_sk}
    
    # 加载数据集
    print(f"\n{datetime.datetime.now()} 加载数据集...")
    sketch_train_loader, photo_train_loader, sk_test_loader, im_test_loader, class_smbs, class_name = \
        dataset.load_dataset(args, transformations)
    print(f"✓ 数据集加载完成: {len(class_name)} 个类别")

    # 初始化视觉模型
    print(f"\n{datetime.datetime.now()} 初始化视觉模型...")
    if args.arch == 'cse_resnet50':
        model_stu = SherryCSEResnet(args.arch, args.num_hashing, args.num_classes, 
                                     args.clip_feature, add_adapter=args.add_adapter)
        model_teacher = cse_resnet50(add_adapter=False)
    elif args.arch in ['vit_small8', 'vit_base']:
        model_stu = SherryDINO(args, is_teacher=False, add_adapter=args.add_adapter, 
                               prompt_learning=args.prompt_learning)
        model_teacher = SherryDINO(args, is_teacher=True, add_adapter=False)
    
    # 仅训练adapter（可选）
    if args.adapter_only:
        for name, param in model_stu.named_parameters():
            if "original_model" in name and "adapter" not in name:
                param.requires_grad = False
        print("✓ 仅训练Adapter模块")

    print(f"✓ 视觉模型初始化完成")

    # ============================================================
    # 文本编码器PEFT模块（核心新增功能）
    # ============================================================
    text_encoder = None
    tokenizer = None
    
    if args.text_adapter or args.text_lora:
        print(f"\n{'=' * 80}")
        print("文本编码器PEFT初始化")
        print(f"{'=' * 80}")
        
        try:
            text_encoder_result = build_text_encoder(args)
            
            if text_encoder_result is not None:
                text_encoder, tokenizer = text_encoder_result
                text_encoder = text_encoder.to(args.device)
                
                # 统计参数
                text_tunable, text_total = count_parameters(text_encoder)
                print(f"✓ 文本编码器加载成功")
                print(f"  - 可训练参数: {text_tunable:,}")
                print(f"  - 总参数: {text_total:,}")
                print(f"  - 可训练比例: {100*text_tunable/text_total:.2f}%")
                
                # 动态生成类别嵌入
                print(f"\n生成类别嵌入...")
                with torch.no_grad():
                    class_smbs_tensor = generate_class_embeddings(
                        class_name, text_encoder, tokenizer, 
                        prompt_type=args.prompt, normalize=True, device=args.device
                    )
                class_smbs = class_smbs_tensor.cpu().numpy()
                print(f"✓ 类别嵌入生成完成: {class_smbs.shape}")
            else:
                print("⚠️  文本编码器构建失败，将使用预计算嵌入")
                
        except Exception as e:
            print(f"❌ 文本编码器加载失败: {e}")
            print(f"⚠️  将使用预计算的类别嵌入继续训练")
            text_encoder = None
            tokenizer = None
    else:
        print(f"\n使用预计算的类别嵌入（未启用文本PEFT）")

    # 统计视觉模型参数
    tunable_param, total_param = count_parameters(model_stu)
    print(f"\n视觉模型参数统计:")
    print(f"  - 可训练参数: {tunable_param:,}")
    print(f"  - 总参数: {total_param:,}")
    print(f"  - 可训练比例: {100*tunable_param/total_param:.2f}%")

    # 保存配置到日志
    config = vars(args)
    with open(os.path.join(args.log_dir, args.file_name, 'logs.txt'), 'a') as f:
        f.write("=" * 80 + "\n")
        f.write("训练配置\n")
        f.write("=" * 80 + "\n")
        for key, value in config.items():
            f.write(f'{key}: {value}\n')
        f.write(f"\n视觉模型 - 可训练参数: {tunable_param:,}, 总参数: {total_param:,}\n")
        if text_encoder is not None:
            text_tunable, text_total = count_parameters(text_encoder)
            f.write(f"文本编码器 - 可训练参数: {text_tunable:,}, 总参数: {text_total:,}\n")
        f.write("=" * 80 + "\n\n")

    # 配置优化器
    base_lr = []
    scale_lr = []
    for name, m in model_stu.named_parameters():
        if 'original_model' in name:
            base_lr.append(m)
        else:
            scale_lr.append(m)

    params_to_optimize = [
        {'params': scale_lr, 'lr': args.lr, 'lr_scale': 1},
        {'params': base_lr, 'lr': args.lr, 'lr_scale': 1}
    ]
    
    # 添加文本编码器参数到优化器
    if text_encoder is not None:
        text_encoder_params = [p for p in text_encoder.parameters() if p.requires_grad]
        if len(text_encoder_params) > 0:
            params_to_optimize.append({
                'params': text_encoder_params, 
                'lr': args.text_encoder_lr,
                'lr_scale': 1
            })
            print(f"\n✓ 已添加 {len(text_encoder_params)} 个文本编码器参数到优化器")
            print(f"  文本编码器学习率: {args.text_encoder_lr}")
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器（3 epoch warmup + exponential decay）
    warm_up_epochs = 3
    lambda_epoch = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
                                  else math.pow(0.001, float(epoch + 1 - warm_up_epochs) / args.epochs)
    lr_scheduler_epoch = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_epoch)

    # 构建模型字典
    model = {'student': model_stu, 'teacher': model_teacher}
    if text_encoder is not None:
        model['text_encoder'] = text_encoder
    
    # 定义损失函数
    criterion_target = nn.CrossEntropyLoss()  # 分类损失
    criterion_kd = loss.SoftCrossEntropy()    # 知识蒸馏损失
    criterion = {'target': criterion_target, 'source': criterion_kd}

    # 移动模型到GPU
    for key, v in model.items():
        model[key] = v.to(args.device)
    
    print(f"\n✓ 所有模型已移动到设备: {args.device}")

    best_map = 0
    best_epoch = 0

    # 从checkpoint恢复（可选）
    if args.resume_file != '':
        print(f"\n从checkpoint恢复训练: {args.resume_file}")
        checkpoint_path = os.path.join(args.log_dir, y_m_d_prefix, args.resume_file)
        if os.path.exists(checkpoint_path):
            args.start_epoch, best_map = load_checkpoint_with_text_encoder(
                checkpoint_path, model_stu, optimizer, text_encoder
            )
            best_epoch = args.start_epoch
            print(f"✓ 从epoch {args.start_epoch} 恢复，best MAP: {best_map:.4f}")
        else:
            print(f"⚠️  Checkpoint文件不存在: {checkpoint_path}")

    # 混合精度训练
    scaler = GradScaler() if args.use_mixed_precision else None
    if args.use_mixed_precision:
        print("\n✓ 启用混合精度训练")
    
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80 + "\n")

    # 训练循环
    early_stop_counter = 0
    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个epoch
        train(sketch_train_loader, photo_train_loader, model, criterion, class_smbs, writer,
              optimizer, epoch, args, args.gradient_accumulation_steps, scaler, 
              tokenizer=tokenizer, class_names=class_name)
        
        # 更新学习率
        lr_scheduler_epoch.step()
        
        # 验证
        map_valid = evaluate.test_map(im_test_loader, sk_test_loader, epoch, model['student'], args)
        
        # 记录日志
        log_message = f'Epoch: [{epoch+1}/{args.epochs}] | Valid MAP: {map_valid:.4f} | Best MAP: {best_map:.4f}'
        print(log_message)
        with open(os.path.join(args.log_dir, args.file_name, 'logs.txt'), 'a') as f:
            f.write(log_message + '\n')
        
        # 保存最佳模型
        if map_valid > best_map:
            best_map = map_valid
            early_stop_counter = 0
            best_epoch = epoch + 1
            
            checkpoint_state = {
                'epoch': best_epoch,
                'arch': args.arch,
                'state_dict': model['student'].state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_map': best_map,
            }
            
            save_checkpoint_with_text_encoder(
                checkpoint_state,
                filename=os.path.join(args.log_dir, args.file_name, 'model_best.pth'),
                text_encoder=text_encoder
            )
            
            print(f"✓ 保存最佳模型 (Epoch {best_epoch}, MAP: {best_map:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop:
                print(f"\n早停触发 (连续 {args.early_stop} 个epoch无提升)")
                print(f"最佳结果: Epoch {best_epoch}, MAP: {best_map:.4f}")
                with open(os.path.join(args.log_dir, args.file_name, 'logs.txt'), 'a') as f:
                    f.write(f"\n早停: Epoch {best_epoch}, Best MAP: {best_map:.4f}\n")
                break
    
    writer.close()
    print("\n" + "=" * 80)
    print(f"训练完成！最佳MAP: {best_map:.4f} (Epoch {best_epoch})")
    print("=" * 80)

        
def train(train_loader, train_loader_ext, model, criterion, class_embs_all, writer,
          optimizer, epoch, args, gradient_accumulation_steps=4, scaler=None, 
          tokenizer=None, class_names=None):
    """训练一个epoch"""
    
    batch_time = evaluate.AverageMeter()
    top5 = evaluate.AverageMeter()
    top1 = evaluate.AverageMeter()
    
    # 设置训练模式
    if args.ngpu > 1:
        for _, v in model.items():
            v = v.module
    
    model['student'].train()
    model['teacher'].eval()
    
    # 处理文本编码器
    use_text_peft = 'text_encoder' in model and tokenizer is not None and class_names is not None
    if use_text_peft:
        model['text_encoder'].train()
        # 在epoch开始时生成类别嵌入，并detach避免梯度累积
        with torch.no_grad():
            class_embs_all = generate_class_embeddings(
                class_names, model['text_encoder'], tokenizer, 
                prompt_type=args.prompt, normalize=True, device=args.device
            )
        print(f"Generated class embeddings at epoch start: {class_embs_all.shape}")
        # 定期刷新的步数（每N个batch更新一次类别嵌入）
        refresh_interval = max(1, len(train_loader) // 10)  # 每个epoch刷新10次
    else:
        # 如果不使用文本PEFT，转换预计算嵌入为tensor
        class_embs_all = torch.from_numpy(class_embs_all).type(torch.Tensor).to(args.device)
        refresh_interval = None
    
    logits_scale = 4.6052  # CLIP logits scale
    total_iters = min(len(train_loader), len(train_loader_ext))
    end = time.time()
    
    pbar = tqdm(zip(train_loader, train_loader_ext), 
                ncols=100, 
                total=min(len(train_loader), len(train_loader_ext)),
                desc=f"Epoch {epoch}")
    
    for i, ((input, target), (input_ext, target_ext)) in enumerate(pbar):
        # 合并草图和照片batch
        input_all = torch.cat([input, input_ext], dim=0)
        tag_zeros = torch.zeros(input.size()[0], 1)
        tag_ones = torch.ones(input_ext.size()[0], 1)
        target_all = torch.cat([target, target_ext], dim=0)
        tag_all = torch.cat([tag_zeros, tag_ones], dim=0)  # [0, 0, ..., 1, 1, ...]
        
        # 随机打乱
        shuffle_idx = np.arange(input_all.size()[0])
        np.random.shuffle(shuffle_idx)
        input_all = input_all[shuffle_idx]
        tag_all = tag_all[shuffle_idx]
        target_all = target_all[shuffle_idx]

        # 移动到GPU
        input_all = input_all.to(args.device)
        tag_all = tag_all.to(args.device)
        target_all = target_all.type(torch.LongTensor).view(-1,).to(args.device)

        # 定期刷新类别嵌入（如果使用文本PEFT）
        if use_text_peft and refresh_interval and i % refresh_interval == 0 and i > 0:
            with torch.no_grad():
                class_embs_all = generate_class_embeddings(
                    class_names, model['text_encoder'], tokenizer, 
                    prompt_type=args.prompt, normalize=True, device=args.device
                )
        
        # 前向传播
        if args.use_mixed_precision:
            with autocast():
                # 学生模型前向
                clip_space_features, stu_out_kd = model['student'](input_all, tag_all)
                
                # 教师模型前向（无梯度）
                with torch.no_grad():
                    teacher_out = model['teacher'](input_all, tag_all)
                
                # 计算logits
                logits_stu = loss.compute_logits(clip_space_features, class_embs_all, 
                                                 comman_modality=True) * logits_scale
                
                # 计算损失
                loss_target = criterion['target'](logits_stu / args.temperature, target_all)
                loss_kd = criterion['source'](stu_out_kd, teacher_out, mask=tag_all)
                loss_total = loss_target + args.kd_lambda * loss_kd
            
            # 反向传播（混合精度）
            scaler.scale(loss_total).backward()
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # 学生模型前向
            clip_space_features, stu_out_kd = model['student'](input_all, tag_all)
            
            # 教师模型前向（无梯度）
            with torch.no_grad():
                teacher_out = model['teacher'](input_all, tag_all)
            
            # 计算logits
            logits_stu = loss.compute_logits(clip_space_features, class_embs_all, 
                                             comman_modality=True) * logits_scale
            
            # 计算损失
            loss_target = criterion['target'](logits_stu / args.temperature, target_all)
            loss_kd = criterion['source'](stu_out_kd, teacher_out, mask=tag_all)
            loss_total = loss_target + args.kd_lambda * loss_kd
            
            # 反向传播
            loss_total.backward()
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 统计准确率
        acc1, acc5 = evaluate.accuracy(logits_stu, target_all, topk=(1, 5))
        top5.update(acc5[0], input_all.size(0))
        top1.update(acc1[0], input_all.size(0))

        # 更新进度条
        pbar.set_postfix({
            'Acc@1': f'{top1.val:.2f}',
            'CE': f'{loss_target.item():.4f}',
            'KD': f'{loss_kd.item():.4f}'
        })
        
        # 记录到TensorBoard和日志
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            # TensorBoard
            writer.add_scalars(
                main_tag='Loss', 
                tag_scalar_dict={
                    'CE': loss_target.item(), 
                    'KD': loss_kd.item(),
                    'Total': loss_total.item()
                }, 
                global_step=i + (epoch + 1) * total_iters
            )
            writer.add_scalar(
                tag='ACC1', 
                scalar_value=top1.val,  
                global_step=i + (epoch + 1) * total_iters
            )
            
            # 文件日志
            with open(os.path.join(args.log_dir, args.file_name, 'logs.txt'), 'a') as f:
                f.write(f'Epoch: [{epoch+1}][{i}/{len(train_loader)}]\t'
                       f'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) | '
                       f'Loss: CE={loss_target.item():.3f} KD={loss_kd.item():.3f} Total={loss_total.item():.3f} | '
                       f'Acc@5 {top5.val:.2f} ({top5.avg:.2f}) Acc@1 {top1.avg:.3f}\n')
    
    # 处理epoch结束时剩余的累积梯度
    if (i + 1) % gradient_accumulation_steps != 0:
        if args.use_mixed_precision and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    main()
