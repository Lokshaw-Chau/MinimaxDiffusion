# CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --data_path /root/share/ImageNet/train/ --spec woof

# CUDA_VISIBLE_DEVICES=0 python train.py --model efficientnet_b0 --data_path /root/share/ImageNet/train/ --spec woof

# CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --data_path /root/share/ImageNet/train/ --spec idc

# CUDA_VISIBLE_DEVICES=0 python train.py --model efficientnet_b0 --data_path /root/share/ImageNet/train/ --spec idc

# CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --data_path /root/share/ImageNet/imagenet-woof/train

CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --data_path /root/share/ImageNet --spec woof

CUDA_VISIBLE_DEVICES=0 python train.py --model efficientnet_b0 --data_path /root/share/ImageNet --spec woof

CUDA_VISIBLE_DEVICES=0 python train.py --model mobilenet_v2 --data_path /root/share/ImageNet --spec woof

CUDA_VISIBLE_DEVICES=0 python train.py --model shufflenet_v2_x0_5 --data_path /root/share/ImageNet --spec woof

CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --data_path /root/share/ImageNet --spec idc

CUDA_VISIBLE_DEVICES=0 python train.py --model efficientnet_b0 --data_path /root/share/ImageNet --spec idc

CUDA_VISIBLE_DEVICES=0 python train.py --model mobilenet_v2 --data_path /root/share/ImageNet --spec idc

CUDA_VISIBLE_DEVICES=0 python train.py --model shufflenet_v2_x0_5 --data_path /root/share/ImageNet --spec idc
