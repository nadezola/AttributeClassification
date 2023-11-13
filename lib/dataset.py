from torch.utils.data import Dataset
from PIL import Image


class ImagesDataset(Dataset):
    def __init__(self, files, labels, attributes, transforms, modelinput, mode):
        super().__init__()
        self.files = files
        self.labels = labels
        self.attributes = attributes
        self.transforms = transforms
        self.mode = mode
        self.modelinput = modelinput

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pic = Image.open(self.files[index]).convert('RGB')
        #pic = self.add_margin(pic)

        if self.mode == 'train' or self.mode == 'val':
            x = self.transforms(pic)
            y = self.labels[index]
            return x, y
        elif self.mode == 'test':
            x = self.transforms(pic)
            return x, self.files[index].name

    def add_margin(self, pil_img):
        width, height = pil_img.size
        new_width = self.modelinput[1]
        new_height = self.modelinput[0]
        left_pad = int((new_width - width) / 2)
        top_pad = int((new_height - height) / 2)
        result = Image.new(pil_img.mode, (new_width, new_height))
        result.paste(pil_img, (left_pad, top_pad))
        return result


# class HumanDataset(Dataset):
#     def __init__(self, frame, human_dets, transforms):
#         super().__init__()
#         self.frame = frame
#         self.human_dets = human_dets
#         self.transforms = transforms
#
#     def __len__(self):
#         return self.human_dets.shape[0]
#
#     def __getitem__(self, index):
#         x, y, w, h = self.human_dets[index, 2:6]
#         human_image = self.frame[int(y):int(y + h), int(x):int(x + w)]
#         pic = Image.fromarray(human_image)
#         x = self.transforms(pic)
#         return x, self.human_dets[index, 1]
