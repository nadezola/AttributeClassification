import sys

DEVICE = None

task = 'mobility_devices'
vis_train_samples = True

if task == 'mobility_devices':
    dataset_root = '/media/nadesha/hdd/INTERACT-DATASET/dataset_ludwig_gpv_icg_regular_6class'
    attributes = ['person', 'stroller', 'wheelchair', 'rollator', 'crutch', 'cane']
else:
    print(f'Task {task} is not supported')
    sys.exit(1)

extracted_data_root = 'extracted_nov23'

model = 'resnet50'  # vgg16, vgg16_bn, resnet18, resnet34, resnet50, resnet152, densenet201, mobilnet, vit_b_16, vit_l_16
resize = (224, 224)
weights = 'IMAGENET1K_V1'  # IMAGENET1K_V1 (DEFAULT) | IMAGENET1K_V2 (only for resnet152, resnet50, mobilenet_v3_largeweights)

num_epochs = 10
batch_size = 32

lr = 0.0001
step_size = 2
mean = [0.3472, 0.3492, 0.3496]
std = [0.1902, 0.1912, 0.1910]

resume = 0  # experiment ID (expID) to resume training or 0 = without training resume
freeze = False

expID = None
