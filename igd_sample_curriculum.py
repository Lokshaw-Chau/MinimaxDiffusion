import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
import torch.optim as optim
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from data import ImageFolder, ImageFolder_mp, MultiEpochsDataLoader, transform_imagenet
from collections import OrderedDict, defaultdict
from PIL import Image
import numpy as np
import gc
import train_models.resnet as RN
import train_models.resnet_ap as RNAP
import train_models.convnet as CN
import train_models.densenet_cifar as DN
from misc.utils import random_indices, rand_bbox, accuracy, AverageMeter
import time
from reparam_module import ReparamModule
import wandb
import torch.nn as nn
import timm
import random

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def define_model(args, nclass, logger=None, size=None):
    """Define neural network models
    """

    args.size = 256

    args.width = 1.0
    args.norm_type = 'instance'
    args.nch = 3

    if size == None:
        size = args.size

    if args.net_type == 'resnet':
        args.depth = 18
        model = RN.ResNet(args.spec,
                          args.depth,
                          nclass,
                          norm_type=args.norm_type,
                          size=size,
                          nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.spec,
                              args.depth,
                              nclass,
                              width=args.width,
                              norm_type=args.norm_type,
                              size=size,
                              nch=args.nch)

    elif args.net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif args.net_type == 'convnet':
        width = int(128 * args.width)
        model = CN.ConvNet(channel=4, num_classes=nclass, net_width=128, net_depth=3, net_act='relu', net_norm='instance', net_pooling='avgpooling', im_size=(args.size, args.size))
        # model = CN.ConvNet(nclass,
        #                 #    net_norm=args.norm_type,
        #                    net_depth=3,
        #                    net_width=128,
        #                    channel=args.nch,
        #                    im_size=(args.size, args.size))
        model.classifier = nn.Linear(2048, nclass)
    elif args.net_type == 'convnet6':
        width = int(128 * args.width)
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=6, net_act='relu', net_norm='instance', net_pooling='avgpooling', im_size=(args.size, args.size))
    elif args.net_type == 'convnet4':
        width = int(128 * args.width)
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=4, net_act='relu', net_norm='instance', net_pooling='avgpooling', im_size=(args.size, args.size))
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    if logger is not None:
        logger(f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}")

    return model

def rand_ckpts(args):
    expert_path = './%s/%s/%s/'%(args.ckpt_path, args.spec, args.net_type)
    expert_files = os.listdir(expert_path)
    # rand_id1 = np.random.choice(len(expert_files))
    rand_id1 = 0
    state = torch.load(expert_path + expert_files[rand_id1])
    print('file name:',expert_path + expert_files[rand_id1])
    # ckpts = state[np.random.choice(len(state))]
    ckpts = state[0]

    if args.spec == 'woof':
        if args.ckpt_path.startswith('ckpts'):
            if args.net_type == 'convnet6':
                #idxs = [0,5,16,40]
                idxs = [0, 4, 14, 36, -1]
            # elif args.ckpt_path.startswith('ema'):
            #     idxs = [0,7,25]
            elif args.net_type == 'resnet_ap':
                idxs = [0,6,16,39]
            elif args.net_type == 'resnet':
                idxs = [0,16,33]
        elif args.ckpt_path.startswith('cut_ckpts'):
            if args.net_type == 'convnet4':
                # idxs = [0,7,16,29,52]
                idxs = [1,4,13,27,57]
            elif args.net_type == 'convnet6':
                # idxs = [0,7,16,29,52]
                idxs = [0,10,26,60]

    elif  args.spec == 'nette':
        if args.ckpt_path.startswith('ckpt'):
            if args.net_type == 'convnet6':
                idxs = [0,2, 8, 22, -1]
            elif args.net_type == 'resnet_ap':
                idxs = [0,6,16,39]
            elif args.net_type == 'resnet':
                idxs = [0,8,27]
                
    elif  args.spec == '1k':
        if args.ckpt_path.startswith('ckpt'):
            if args.net_type == 'convnet6':
                idxs = [0,5,18,52]

    # select_idxs = np.arange(0, len(ckpts), 20).tolist()
    # # select_idxs = np.random.choice(int(len(ckpts)*0.6),size=5)
    # # print('select_idxs',select_idxs)
    # ckpts = [ckpts[idx] for idx in select_idxs]
    print('ckpt idxs:',idxs)
    return [ckpts[ii] for ii in idxs]

def get_curriculum_grads(args, sel_classes, class_labels, sel_class, ckpts, surrogate, curriculum_filter, device='cuda'):
    # Setup data:
    criterion_ce = nn.CrossEntropyLoss(reduction='none').to(device)
    correspond_labels = defaultdict(list)
    grads_memory = []
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder_mp(args.data_path, transform=transform, nclass=args.nclass,
                          ipc=args.real_ipc, spec=args.spec, phase=args.phase,
                          seed=0, return_origin=True, sel_class=sel_class) # target idx [0-nclss]

    real_loader = DataLoader(
        dataset,
        batch_size=200,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    print('load real grad memory ')
    for x, ry, y in real_loader: # ry 是0-1的，y是在ori 1000个里的真实index
        # ry = ry.numpy()
        assert torch.all(y == y[0]), "Tensor y contains different values"
        x = x.to(device)
        y = int(y.numpy()[0])
        grads_memory.extend(x.detach().split(1))
        correspond_labels[y] = ry[0].cpu().numpy()

    real_gradients = defaultdict(list)
    correspond_y = correspond_labels[y]
    
    # judge by a trained model
    # easiest sample from misclassified samples
    if curriculum_filter is not None:
        curriculum_filter.eval()
        loss_list = []
        is_correct_list = []
        gap = 200
        gap_idxs = np.arange(0, len(grads_memory), gap).tolist()
        for gi in gap_idxs:
            cur_embd0 = torch.stack(grads_memory[gi:gi+gap]).cpu().numpy()
            cur_embds = torch.from_numpy(cur_embd0).squeeze(1).to(device).requires_grad_(False)
            # cur_params = torch.cat([p.data.to(device).reshape(-1) for p in ckpts[-1]], 0).requires_grad_(True)
            # real_pred = surrogate(cur_embds, flat_param=cur_params)
            with torch.no_grad():
                pred = curriculum_filter(cur_embds)
                pred = curriculum_filter(cur_embds)
                real_target = torch.tensor([np.ones(len(cur_embds))*correspond_y], dtype=torch.long, device=args.device).view(-1) 
                is_correct = pred.argmax(1) == real_target
                real_loss = criterion_ce(pred, real_target)
                loss_list.extend(real_loss.detach())
                is_correct_list.extend(is_correct)
         
        misclassified_idxs = torch.where(torch.tensor(is_correct_list) == 0)[0]
        true_idxs = torch.where(torch.tensor(is_correct_list) == 1)[0]
        print("number of misclassified samples:", len(misclassified_idxs))
        # origin_loss_list = torch.stack(loss_list)
        misclassified_loss_list = [loss_list[idx] for idx in misclassified_idxs]
        misclassified_loss_list = torch.stack(misclassified_loss_list)
        sorted_misclassified_idxs = torch.argsort(misclassified_loss_list)
        if len(sorted_misclassified_idxs) < args.grad_ipc:
            true_idxs = true_idxs.tolist()
            sorted_misclassified_idxs = sorted_misclassified_idxs.tolist()
            sorted_misclassified_idxs.extend(random.sample(true_idxs, args.grad_ipc - len(misclassified_idxs)))
            idxs = sorted_misclassified_idxs
        else:
            idxs = sorted_misclassified_idxs[0:args.grad_ipc]
            
    else:
        idxs = np.arange(0, len(grads_memory)).tolist()
        random.shuffle(idxs)
        idxs = idxs[0:args.grad_ipc]
    
    # get grad
    gap = 100
    gap_idxs = np.arange(0, args.grad_ipc, gap).tolist()
    sampled_mem = [grads_memory[idx] for idx in idxs]
    ckpt_grads = []
    for ii, ckpt in enumerate(ckpts[:-1]):
        for gi in gap_idxs:
            cur_embd0 = torch.stack(sampled_mem[gi:gi+gap]).cpu().numpy()
            cur_embds = torch.from_numpy(cur_embd0).squeeze(1).to(device).requires_grad_(True)
            cur_params = torch.cat([p.data.to(device).reshape(-1) for p in ckpt], 0).requires_grad_(True)
            if gi == 0:
                acc_grad = torch.zeros(cur_params.shape)
            real_pred = surrogate(cur_embds, flat_param=cur_params)
            real_target = torch.tensor([np.ones(len(cur_embds))*correspond_y], dtype=torch.long, device=args.device).view(-1) 
            real_loss = criterion_ce(real_pred, real_target).mean()
            real_grad = torch.autograd.grad(real_loss, cur_params)[0] #.detach().clone().requires_grad_(False)
            acc_grad += real_grad.detach().data.cpu()
        ckpt_grads.append(acc_grad / len(gap_idxs)) 
    real_gradients[y]=ckpt_grads
    # del cur_imgs
    del cur_params
    del real_grad
    # del real_imgs
    del correspond_y
    del grads_memory
    gc.collect()
    surrogate.zero_grad()
    print('end')
    print('all real memory len', sum(len(lst) for lst in real_gradients.values()))
    
    return real_gradients, y, correspond_labels

def train_filter(args, image_per_class):
    total_sample_nums = image_per_class * args.nclass
    # hyperparameters
    bsz = 64
    if total_sample_nums < 64:
        bsz = 10
    
    if image_per_class <= 10:
        epochs = 2000
    elif image_per_class <= 50:
        epochs = 1500
    elif image_per_class <= 200:
        epochs = 1000

    # load data
    traindir = args.save_dir
    valdir = '/root/share/ImageNet/val'
    train_transform, test_transform = transform_imagenet(augment=True,
                                                        from_tensor=False)
    def is_valid_file(file_path):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']
        return any(file_path.endswith(ext) for ext in valid_extensions)
    
    train_dataset = ImageFolder(traindir,
                                train_transform,
                                nclass=args.nclass,
                                load_memory=True,
                                ipc=image_per_class,
                                spec=args.spec,
                                is_valid_file=is_valid_file)
    
    val_dataset = ImageFolder(valdir,
                              test_transform,
                              nclass=args.nclass,
                              load_memory=True,
                              spec=args.spec)
    
    nclass = len(train_dataset.classes)
    print("Subclass is extracted: ")
    print(" #class: ", nclass)
    print(" #train: ", len(train_dataset.targets))
    
    train_loader = MultiEpochsDataLoader(train_dataset,
                                         batch_size=bsz,
                                         shuffle=True,
                                         num_workers=8,
                                         persistent_workers=True,
                                         pin_memory=True)
    
    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=64,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4,
                                       pin_memory=True)
    
    # train
    criterion = nn.CrossEntropyLoss().cuda()
    args.net_type = args.curriculum_filter
    print("Initialize a new curriculum filter model")
    model = define_model(args, args.target_nclass)
    model.cuda()
    optimizer = optim.SGD(model.parameters(),
                          0.01,
                          momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * epochs // 3, 5 * epochs // 6], gamma=0.2)
    for epoch in range(epochs):
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda()
            target = target.cuda()
            # generate mixed sample
            lam = np.random.beta(1.0, 1.0)
            rand_index = random_indices(target, nclass=args.nclass)

            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        if epoch % 100 == 0:
            print('Epoch %d, Loss: %.4f' % (epoch, loss.item()))
    
    # validate the model
    top1 = AverageMeter()
    top5 = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
    
    print(f'Validation: Top1:{top1.avg} Top5:{top5.avg}')
    return model


def main(args):
    # set wandb
    wandb.init(project='guidance sampling', name='igd', config=args)


    # Setup PyTorch:
    torch.manual_seed(args.seed)
    # torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == '1k':
        file_list = './misc/class_indices.txt'
    else:
        file_list = './misc/class100.txt'
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class)) # 0-1000

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval() # mc add

    # define gm resources 
    args.device = 'cuda'
    surrogate = define_model(args, args.target_nclass).to(args.device)
    surrogate = ReparamModule(surrogate)
    surrogate.eval()
    ckpts = rand_ckpts(args)
    encoder = None
    criterion_ce = nn.CrossEntropyLoss().to(args.device)
    
    batch_size = 1
    if args.num_samples % args.bootstrap_gap == 0:
        images_per_curriculum = args.bootstrap_gap
        curriculum_num = args.num_samples // images_per_curriculum
    else:
        raise ValueError('num_samples should be divisible by bootstrap_gap')

    for curriculum in range(curriculum_num):
        # curriculum generation
        images_in_each_class = []
        for class_label, sel_class in zip(class_labels, sel_classes):
            class_path = os.path.join(args.save_dir, sel_class)
            if not os.path.exists(class_path):
                os.makedirs(class_path, exist_ok=True)
            images_in_each_class.append(max(0, len(os.listdir(class_path)) - 1))
        min_images_in_each_class = min(images_in_each_class)
        if min_images_in_each_class >= (curriculum + 1) * images_per_curriculum:
            print("skip curriculum %d"%(curriculum))
            continue
        else:
            print(min_images_in_each_class)
            if args.ablation == 0:
                curriculum_filter = train_filter(args, min_images_in_each_class)
            else:
                curriculum_filter = None
            generation_range = range(min_images_in_each_class, (curriculum + 1) * images_per_curriculum)

        for class_label, sel_class in zip(class_labels, sel_classes):
            class_path = os.path.join(args.save_dir, sel_class)
            if len(os.listdir(class_path)) - 1 >= (curriculum + 1) * images_per_curriculum:
                print("skip class %s"%(sel_class))
                continue

            print(sel_class)
            # real_gradients, cur_cls, correspond_labels  = get_grads(args, sel_classes, class_labels, sel_class, surrogate=surrogate, ckpts=ckpts)
            real_gradients, cur_cls, correspond_labels  = get_curriculum_grads(args, sel_classes, class_labels, sel_class, 
                                                                               surrogate=surrogate, curriculum_filter=curriculum_filter,ckpts=ckpts)
            assert class_label == cur_cls
            pseudo_memory_c = []
            # load all generated images if not empty
            if curriculum > 0:
                old_pseudo_memory_c = torch.load(os.path.join(class_path, 'pseudo_memory.pt'))
                pseudo_memory_c.extend(old_pseudo_memory_c.split(1))
            for shift in tqdm(generation_range):
                
                # Create sampling noise:
                z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
                y = torch.tensor([class_label], device=device)

                # Setup classifier-free guidance:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * batch_size, device=device)
                y = torch.cat([y, y_null], 0)

                gm_resource = [vae, surrogate, ckpts, real_gradients[class_label], correspond_labels[class_label], criterion_ce, args.repeat, args.repeat, args.gm_scale, args.f_scale, encoder]
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, gm_resource=gm_resource, gen_type=args.guide_type, low=args.low, high=args.high, pseudo_memory_c=pseudo_memory_c, neg_e=args.lambda_neg)

                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
                # print('samples',samples.shape)
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                # add psuedo samples to the memory
                if args.guide_type == 'igd':
                    # latent space'
                    pseudo_memory_c.extend(samples.detach().split(1))
                    
                samples = vae.decode(samples / 0.18215).sample
                # Save and display images:
                while len(pseudo_memory_c) > args.memory_size:
                        pseudo_memory_c.pop(0)

                for image_index, image in enumerate(samples):
                    file_number = image_index + shift * batch_size + args.total_shift
                    save_image(image, os.path.join(args.save_dir, sel_class,
                                                f"{file_number:04d}.png"), normalize=True, value_range=(-1, 1))
            # save the pseudo memory
            pseudo_memory_c = torch.cat(pseudo_memory_c, 0)
            torch.save(pseudo_memory_c, os.path.join(class_path, 'pseudo_memory.pt'))
        
        print('following is the result of pos_e %s and neg_e %s'%(args.lambda_pos, args.lambda_neg))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--target_nclass", type=int, default=1000, help='the class number for generation')
    parser.add_argument("--depth", type=int, default=10, help='the network depth')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    parser.add_argument("--memory-size", type=int, default=100, help='the memory size')
    parser.add_argument("--real_ipc", type=int, default=1000, help='the number of samples participating in the fine-tuning')
    parser.add_argument("--grad-ipc", type=int, default=80, help='the number of samples participating in the fine-tuning')
    parser.add_argument('--lambda-pos', default=0.03, type=float, help='weight for representativeness constraint')
    parser.add_argument('--lambda-neg', default=0.01, type=float, help='weight for diversity constraint')
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--net-type", type=str, default='convnet6')
    parser.add_argument("--gm-scale", type=float, default=0.02)
    parser.add_argument("--f-scale", type=float, default=0.02)
    parser.add_argument("--feat-extractor", type=str, default='dinov2')
    parser.add_argument("--low", type=int, default=500, help='allowed lowest time step for gm guidance')
    parser.add_argument("--high", type=int, default=800, help='allowed highest time step for gm guidance')
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--repeat", type=int, required=True, help='repeat for the GM recursive during low-high sampling steps')
    parser.add_argument("--guide_type", type=str, default='igd', help='the guidance type for the generation')
    # parser.add_argument("--schedule_gap", type=int, default=200, help='the gap for the curriculum schedule')

    parser.add_argument("--curriculum_filter", type=str, default='convnet6', help='the guidance type for the generation')
    parser.add_argument("--bootstrap_gap", type=int, default=4, help='the number of curriculums')
    parser.add_argument("--ablation", type=int, default=0)

    args = parser.parse_args()
    print('args\n',args)
    main(args)
