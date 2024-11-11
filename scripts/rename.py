import os

exp_dir = '/root/workspace/MinimaxDiffusion/results/dit-rgd-dino'
for exp in os.listdir(exp_dir):
    img_dir = os.path.join(exp_dir, exp)
    for class_id in os.listdir(img_dir):
        class_dir = os.path.join(img_dir, class_id)
        for img in os.listdir(class_dir):
            img_number = img.split('.')[0]
            # turn to int
            img_number = int(img_number)
            new_img = f'{img_number:04d}.png'
            print(os.path.join(class_dir, img), os.path.join(class_dir, new_img))
            os.rename(os.path.join(class_dir, img), os.path.join(class_dir, new_img))