
export CUDA_VISIBLE_DEVICES=2
# m3d

phase=0 # end 7
nclass=10

for spec in woof nette
do
    for km_expand in 1 10
    do
        for nsample in 10 50 100 200
        do
            save_path=./results/dit-m3d/${spec}-${nsample}-km_expand-${km_expand}
            tag=m3d-km-expand-${km_expand}

            python m3d_sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt \
                --save-dir ${save_path} --spec ${spec}  --num-samples ${nsample} --nclass ${nclass} --phase ${phase} --guide_type None --km_expand ${km_expand}

            python train.py -d imagenet --imagenet_dir ${save_path} /root/share/ImageNet \
                    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc ${nsample} --tag ${tag} --slct_type random --spec ${spec} --repeat 3
        done
    done
done