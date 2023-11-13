import sys

DEVICE = None

task = 'mobility_devices'
vis_train_samples = True

if task == 'mobility_devices':
    dataset_root = '/media/nadesha/hdd/INTERACT-DATASET/dataset_ludwig_mobility/extracted_nov23'
    attributes = ['person', 'stroller', 'wheelchair', 'rollator', 'crutch', 'cane']
else:
    print(f'Task {task} is not supported')
    sys.exit(1)


model = 'resnet50'  # vgg16, vgg16_bn, resnet18, resnet34, resnet50, resnet152, densenet201, mobilnet, vit_b_16, vit_l_16
resize = (224, 224)
weights = 'IMAGENET1K_V1'  # IMAGENET1K_V1 (DEFAULT) | IMAGENET1K_V2 (only for resnet152, resnet50, mobilenet_v3_largeweights)

num_epochs = 10
batch_size = 32

lr = 0.0001
step_size = 2
mean = [0.3961, 0.4001, 0.4086]  # [0.3220, 0.3190, 0.3244] for ludwig project | [0.3093, 0.3077, 0.3066] for ludwig paper | default [0.4802, 0.4481, 0.3975]
std = [0.2126, 0.2115, 0.2048]   # [0.1867, 0.1865, 0.1865] for ludwig project | [0.1909, 0.1907, 0.1911] for ludwig paper | default [0.2302, 0.2265, 0.2262]

resume = 0  # experiment ID (expID) to resume training or 0 = without training resume
freeze = False

expID = None
