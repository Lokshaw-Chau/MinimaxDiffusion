
export CUDA_VISIBLE_DEVICES=1

k=5
low=30
high=45
gi=200
cp=ckpts
r=1
spec=woof
nsample=100
ntype=convnet6
gamma=120
phase=0 # end 7
nclass=10
tart_ncls=10
d=6
cn=10

python igd_sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt \
    --save-dir "./results/dit-igd-schedule/easy-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path /root/share/ImageNet/train \
    --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} --depth 10 \
    --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --lambda-neg ${gamma} \
    --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls} --guide_type igd --memory-size 100 --curriculum_num ${cn}

# python train.py -d imagenet --imagenet_dir ./results/dit-igd-schedule/easy-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag igd-easy --slct_type random --spec woof --repeat 3

# python train.py -d imagenet --imagenet_dir ./results/dit-igd-schedule/easy-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag igd-easy --slct_type random --spec woof --repeat 3

# python train.py -d imagenet --imagenet_dir ./results/dit-igd-schedule/easy-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 100 --tag igd-easy --slct_type random --spec woof --repeat 1

# k=5
# # f=50
# low=30
# high=45
# gi=200
# cp=ckpts
# r=1

# nsample=100
# ntype=convnet6
# gamma=120
# phase=0 # end 7
# nclass=10
# tart_ncls=10
# d=6
# guide_type=igd
# spec=nette

# python igd_sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt \
#         --save-dir "./results/dit-igd-schedule/easy-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path /root/share/ImageNet/train \
#         --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} --depth 10 \
#         --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --lambda-neg ${gamma} \
#         --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls} --guide_type ${guide_type} --schedule_gap 200

# python train.py -d imagenet --imagenet_dir ./results/dit-igd-schedule/easy-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag igd-easy --slct_type random --spec ${spec} --repeat 3

# python train.py -d imagenet --imagenet_dir ./results/dit-igd-schedule/easy-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag igd-easy --slct_type random --spec ${spec} --repeat 3

# python train.py -d imagenet --imagenet_dir ./results/dit-igd-schedule/easy-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 100 --tag igd-easy} --slct_type random --spec ${spec} --repeat 1