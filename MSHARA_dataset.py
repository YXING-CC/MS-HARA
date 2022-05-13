import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Cardata_path import Path
from PIL import Image
import scipy.io as scio
from torchvision.transforms import transforms
from transforms import ConvertBHWCtoBCHW, ConvertBCHWtoCBHW


class CarDataset_multi_2(Dataset):
    def __init__(self, dataset='face', split='val', clip_len=16, transform1=None, transform2=None):
        print('init')
        self.root_dir, self.output_dir, self.seq_lab_dir, self.flow_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        seqlab_folder = os.path.join(self.seq_lab_dir, split)

        self.clip_len = clip_len
        self.split = split

        self.fnames, labels, self.flabels = [], [], []
        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 134
        self.resize_width = 134
        self.crop_size = 112
        self.dataset = dataset
        if dataset =='gtea':
            self.resize_height_resnet = 200
            self.resize_width_resnet = 320
        else:
            self.resize_height_resnet = 224
            self.resize_width_resnet = 224

        self.transform1 = transform1
        self.transform2 = transform2

        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                self.flabels.append(os.path.join(seqlab_folder, label, fname))

                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "face":
            if not os.path.exists('E:/Matlab_Work/Carknows/Carpy/face.txt'):
                with open('E:/Matlab_Work/Carknows/Carpy/face.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id + 1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer, buffer_resnet = self.load_frames(self.fnames[index])
        buffer, buffer_resnet = self.crop(buffer, buffer_resnet, self.clip_len, self.crop_size)

        labels = np.array(self.label_array[index])
        seq_labels = self.get_seq_labels_from_mat(self.flabels[index], self.dataset)

        buffer, buffer_resnet = self.to_tensor(buffer, buffer_resnet, self.transform1, self.transform2)

        return buffer, buffer_resnet, torch.from_numpy(labels), seq_labels

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)

        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        buffer_resnet = np.empty((frame_count, self.resize_height_resnet, self.resize_width_resnet, 3),
                                 np.dtype('float32'))

        for i, frame_name in enumerate(frames):

            frame_pil = Image.open(frame_name)
            frame_pil_re = frame_pil.resize((self.resize_height, self.resize_width))

            buffer[i] = frame_pil_re
            buffer_resnet[i] = frame_pil

        return buffer, buffer_resnet


    def to_tensor(self, buffer, buffer_resnet, use_transform1, use_transform2):
        buffer1 = torch.from_numpy(buffer) / 255

        if use_transform1 is not None:
            buffer = use_transform1(buffer1)
        else:
            buffer = buffer.transpose((3, 0, 1, 2))  # C3D
            buffer = torch.from_numpy(buffer)

        if use_transform2 is not None:
            buffer_resnet = use_transform2(buffer_resnet)
        else:
            buffer_resnet = buffer_resnet.transpose((0, 3, 1, 2))  # C3D
            buffer_resnet = torch.from_numpy(buffer_resnet)

        return buffer, buffer_resnet


    def crop(self, buffer, buffer_resnet, clip_len, crop_size):
        pred_con = 5
        bound_sep = 5

        if clip_len > 120:
            split_index = np.linspace(0, clip_len, clip_len)
            split_index = np.ceil(split_index)
            split_index = split_index.astype('int64')
        else:
            bound = np.rint(buffer.shape[0] / bound_sep)

            begin_index = np.random.randint(0, bound)
            upper_bound = buffer.shape[0] - pred_con
            end_index = np.random.randint(bound * (bound_sep - 1), upper_bound)
            split_index = np.linspace(begin_index, end_index, clip_len)
            split_index = np.ceil(split_index)
            split_index = split_index.astype('int64')

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[split_index,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        buffer_resnet = buffer_resnet[split_index, :, :, :]

        return buffer, buffer_resnet

    def get_seq_labels_from_mat(self, flabels, dataset):
        if dataset == 'face':
            mat_path = os.path.join(flabels, 'lab.mat')
            mat_lab = scio.loadmat(mat_path)
            mat_lab_seq = mat_lab['lab_gt']
            seq_lab = mat_lab_seq.astype("int32")
        elif dataset == 'gtea':
            mat_path = os.path.join(flabels, 'state.mat')
            mat_lab = scio.loadmat(mat_path)
            mat_lab_seq = mat_lab['state_p']
            seq_lab = mat_lab_seq.astype("int32")

        if seq_lab.shape[1] < 151:          # to check abnormal sequences
            print('mat_path', mat_path)
        return seq_lab


if __name__ == "__main__":
    dataset = 'face'
    clp_len = 150

    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    resize_size = 224
    crop_size = 112

    mean_res = [0.485, 0.456, 0.406]
    std_res = [0.229, 0.224, 0.225]

    trans1 = [
        ConvertBHWCtoBCHW(),
        transforms.ConvertImageDtype(torch.float32),
    ]
    trans1.extend([
        transforms.Normalize(mean=mean, std=std),
        ConvertBCHWtoCBHW()])

    trans2 = [
        ConvertBHWCtoBCHW(),
        transforms.ConvertImageDtype(torch.float32),
    ]
    trans2.extend([
        transforms.Normalize(mean=mean_res, std=std_res)])

    transform1_t = transforms.Compose(trans1)
    transform2_t = transforms.Compose(trans2)

    train_dataloader = DataLoader(CarDataset_multi_2(dataset=dataset, split='train', clip_len=clp_len,
                                                  transform1=transform1_t, transform2=None),
                                  batch_size=3, shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = DataLoader(CarDataset_multi_2(dataset=dataset, split='val', clip_len=clp_len,
                                                transform1=transform1_t, transform2=None),
                                batch_size=3, num_workers=4, pin_memory=True)

    test_dataloader = DataLoader(CarDataset_multi_2(dataset=dataset, split='test', clip_len=clp_len,
                                                 transform1=transform1_t, transform2=None),
                                 batch_size=3, shuffle=True, num_workers=4, pin_memory=True)

    for i, sample in enumerate(train_dataloader):
        inputs1 = sample[0]
        inputs2 = sample[1]
        labels = sample[2]
        labels_seq = sample[3]
        print('main input 1', inputs1.size())
        print('main input 2', inputs2.size())
        print('main label', labels, labels.size())
        print('main label seq', labels_seq.size())

        if i == 1:
            break

