"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from sklearn.cluster import MiniBatchKMeans
from data import ImageFolder, ImageFolder_mp, MultiEpochsDataLoader
import torchvision.transforms as transforms
import numpy as np
import math

def initialize_km_models(label_list, args):
        print("Initialize K-means models")
        km_models = {}
        for prompt in label_list:
            model_name = f"MiniBatchKMeans_{prompt}"
            model = MiniBatchKMeans(n_clusters=args.num_samples, random_state=0, batch_size=(
            args.km_expand * args.num_samples),n_init="auto")
            km_models[model_name] = model
        return km_models

def prototype_kmeans(vae, km_models, args):
    print("Start prototype k-means")
    traindir = '/root/share/ImageNet/train'
    train_transform = transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    train_dataset = ImageFolder(traindir,
                                train_transform,
                                nclass=args.nclass,
                                load_memory=True,
                                spec=args.spec)
    idx2class = train_dataset.idx_to_class
    data_loader = MultiEpochsDataLoader(train_dataset,
                                         batch_size=args.km_expand * args.num_samples,
                                         shuffle=True,
                                         num_workers=8,
                                         persistent_workers=True,
                                         pin_memory=True)
    
    latents = {}
    for label in idx2class.keys():
        latents[idx2class[label]] = []
    
    for images, labels in tqdm(data_loader, total=len(data_loader), position=0):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        prompts = []
        for label in labels:
            prompt = idx2class[label.item()]
            prompts.append(prompt)

        # init_latents, _ = pipe(prompt=prompts, image=images, strength=0.7, guidance_scale=8)
        with torch.no_grad():
            init_latents = vae.encode(images).latent_dist.sample().mul_(0.18215)
        
        for latent, prompt in zip(init_latents, prompts):
            latent = latent.view(1, -1).cpu().numpy()
            latents[prompt].append(latent)
            if len(latents[prompt]) == args.km_expand * args.num_samples:
                km_models[f"MiniBatchKMeans_{prompt}"].partial_fit(np.vstack(latents[prompt]))
                latents[prompt] = []  # save the memory, avoid repeated computation
    if len(latents[prompt]) >= args.num_samples:
        km_models[f"MiniBatchKMeans_{prompt}"].partial_fit(np.vstack(latents[prompt]))
    return km_models

def gen_prototype(label_list, km_models):
    prototype = {}
    for prompt in label_list:
        model_name = f"MiniBatchKMeans_{prompt}"
        model = km_models[model_name]
        cluster_centers = model.cluster_centers_
        N = int(math.sqrt(cluster_centers.shape[1] / 4))
        num_clusters = cluster_centers.shape[0]
        reshaped_centers = []
        for i in range(num_clusters):
            reshaped_center = cluster_centers[i].reshape(4, N, N)
            reshaped_centers.append(reshaped_center.tolist())
        prototype[prompt] = reshaped_centers
    return prototype

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
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
        class_labels.append(all_classes.index(sel_class))

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
    # TODO k-means for prototype init
    km_models = initialize_km_models(sel_classes, args)
    km_models = prototype_kmeans(vae, km_models, args)
    prototype_dict = gen_prototype(sel_classes, km_models)

    # generate images
    batch_size = 1

    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        for shift in tqdm(range(args.num_samples // batch_size)):
            # Create sampling noise:
            #z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            z = prototype_dict[sel_class][shift]
            z = torch.tensor(z, device=device)
            z = z.unsqueeze(0)
            y = torch.tensor([class_label], device=device)
            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, gen_type=args.guide_type)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            for image_index, image in enumerate(samples):
                save_image(image, os.path.join(args.save_dir, sel_class,
                                               f"{image_index + shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000) # for DiT class table
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')

    parser.add_argument("--guide_type", type=str, default='none', help='the guidance type for the generation')
    parser.add_argument("--km_expand", type=int, default=10, help='expand ration for minibatch k-means model')

    args = parser.parse_args()
    main(args)
