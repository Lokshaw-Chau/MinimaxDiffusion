
export CUDA_VISIBLE_DEVICES=0

################## influence guided sample ###################
# k=5 # 
# low=30 # guidance interval
# high=45
# gi=200
# cp=ckpts
# r=1 # repeat sampling
# spec=woof
# nsample=50
# ntype=convnet6
# gamma=120 # weight for diversity constraint
# phase=0 # end 7
# nclass=10
# tart_ncls=10
# d=6

# python igd_sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/logs/run-0/001-DiT-XL-2-minimax/checkpoints/0012000.pt \
#     --save-dir "./results/dit-distillation/${spec}-${nsample}-dit-igd-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path /root/share/ImageNet/train \
#     --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} --depth 10 \
#     --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --lambda-neg ${gamma} \
#     --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls}
# # evaluation
# python train.py -d imagenet --imagenet_dir ./results/dit-distillation/${spec}-${nsample}-dit-igd-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag igd_woof --slct_type random --spec woof

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

python igd_sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt \
    --save-dir "./results/dit-igd-schedule/easy-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path /root/share/ImageNet/train \
    --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} --depth 10 \
    --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --lambda-neg ${gamma} \
    --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls} --guide_type igd --memory-size 100 --schedule_gap 200

# python train.py -d imagenet --imagenet_dir ./results/dit-igd/${spec}-${nsample}-wo_diversity-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 50 --tag igd_woof_wo_diverstiy --slct_type random --spec woof --repeat 3

# python train.py -d imagenet --imagenet_dir ./results/dit-igd/${spec}-${nsample}-wo_diversity-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high} /root/share/ImageNet \
#         -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 100 --tag igd_woof_wo_diverstiy --slct_type random --spec woof --repeat 1

python igd_sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt \
    --save-dir "./results/dit-igd-schedule/schedule-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path /root/share/ImageNet/train \
    --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} --depth 10 \
    --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --lambda-neg ${gamma} \
    --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls} --guide_type igd --memory-size 100 --schedule_gap 50

python igd_sample.py --model DiT-XL/2 --image-size 256 --ckpt /root/workspace/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt \
    --save-dir "./results/dit-igd-schedule/schedule-${spec}-${nsample}-${cp}-${ntype}-k${k}-gamma${gamma}-r${r}-gi${gi}-low${low}-high${high}" --data-path /root/share/ImageNet/train \
    --spec ${spec} --gm-scale ${k} --grad-ipc ${gi} --net-type ${ntype} --depth 10 \
    --low ${low} --high ${high} --ckpt_path ${cp} --repeat ${r} --num-samples ${nsample} --lambda-neg ${gamma} \
    --nclass ${nclass} --phase ${phase} --target_nclass ${tart_ncls} --guide_type igd --memory-size 100 --schedule_gap 25