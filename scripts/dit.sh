
export CUDA_VISIBLE_DEVICES=1
# dit

# woof 

nsample=200
phase=0 # end 7
nclass=10

spec=woof

save_path=./results/dit/${spec}-${nsample}
tag=dit

python sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt \
    --save-dir ${save_path} --spec ${spec}  --num-samples ${nsample} --nclass ${nclass} --phase ${phase} --guide_type None 

python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag ${tag} --slct_type random --spec ${spec} --repeat 3

python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag ${tag} --slct_type random --spec ${spec} --repeat 3

python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 100 --tag ${tag} --slct_type random --spec ${spec} --repeat 3
        
python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 200 --tag ${tag} --slct_type random --spec ${spec} --repeat 3

# nette 

nsample=200
phase=0 # end 7
nclass=10

spec=nette

save_path=./results/dit/${spec}-${nsample}
tag=dit

python sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt \
    --save-dir ${save_path} --spec ${spec}  --num-samples ${nsample} --nclass ${nclass} --phase ${phase} --guide_type None 

python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag ${tag} --slct_type random --spec ${spec} --repeat 3

python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag ${tag} --slct_type random --spec ${spec} --repeat 3

python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 100 --tag ${tag} --slct_type random --spec ${spec} --repeat 3
        
python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
        -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 200 --tag ${tag} --slct_type random --spec ${spec} --repeat 3