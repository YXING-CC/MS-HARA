from collections import Counter
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np
from torch.autograd import Variable
import timeit
import os

rand_flg = True
# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
stop_time_init = str(timeit.default_timer())

def accuracy(output, target, topk=(1,3)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def process_sig_img(clip_img_final, org_clip_img_size):
    process_clip = torch.empty([org_clip_img_size[0],3,224,224])
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for ind in range(org_clip_img_size[0]):
        im_tmp = clip_img_final[ind, :, :, :]
        im_tmp = im_tmp.permute(1, 2, 0)
        im_tmp = im_tmp.numpy()
        img = Image.fromarray(np.uint8(im_tmp)).convert('RGB')
        img_tensor = preprocess(img)
        process_clip[ind,:,:,:] = img_tensor

    clip_img_final = process_clip

    return clip_img_final


def generate_last_prediction_data(clip_i, inputs, inputs_resnet, seq_labels):
    if rand_flg:
        rand_ind = np.random.randint(0, 15)
    else:
        rand_ind = 8

    clip_i_final = clip_i + 16
    clip_seg_final = inputs[:, :, clip_i:clip_i_final, :, :]
    clip_img_final = inputs_resnet[:, clip_i + rand_ind, :, :, :]
    org_clip_img_size = clip_img_final.size()

    clip_img_final = process_sig_img(clip_img_final, org_clip_img_size)
    lab_seq_sub_final = seq_labels[:, :, clip_i:clip_i_final]

    clip_seg_final = Variable(clip_seg_final, requires_grad=False).to(device)
    clip_img_final = Variable(clip_img_final, requires_grad=False).to(device)

    lab_seq_sub_shape = lab_seq_sub_final.shape

    seqeunce_label = np.empty([lab_seq_sub_shape[0]])

    for c_ind in range(lab_seq_sub_shape[0]):
        sub_seq = lab_seq_sub_final[c_ind, :, :]
        sub_seq = sub_seq.squeeze()
        sub_seq_np = sub_seq.numpy()
        sub_count_np = Counter(sub_seq_np)
        value, cnt = sub_count_np.most_common()[0]
        seqeunce_label[c_ind] = value - 1

    seqeunce_label = torch.from_numpy(seqeunce_label)
    seqeunce_label = seqeunce_label.clone().detach()
    seqeunce_label = Variable(seqeunce_label, requires_grad=False).to(device)
    seqeunce_label = seqeunce_label.long()

    return clip_seg_final, clip_img_final, seqeunce_label


def wrtieresults(testing_acc_long, testing_acc_activity, testing_acc_long_pred, testing_acc_activity_pred):
    txtfile_name = 'results.txt'
    stop_time = str(timeit.default_timer())
    txtfile_name = stop_time + txtfile_name

    content = np.vstack((testing_acc_long, testing_acc_activity, testing_acc_long_pred, testing_acc_activity_pred))

    # print(content)
    if os.path.exists(txtfile_name):
        os.remove(txtfile_name)

    with open(txtfile_name, "a+") as f:
        for i in range(4):
            f.writelines(str(content[i, :]))
            f.writelines("\n")

def wrtieresults_train(testing_acc_long, testing_acc_activity, testing_acc_long_pred, testing_acc_activity_pred):
    txtfile_name = 'results_train.txt'
    stop_time = str(timeit.default_timer())
    txtfile_name = stop_time + txtfile_name

    content = np.vstack((testing_acc_long, testing_acc_activity, testing_acc_long_pred, testing_acc_activity_pred))

    # print(content)
    if os.path.exists(txtfile_name):
        os.remove(txtfile_name)

    with open(txtfile_name, "a+") as f:
        for i in range(4):
            f.writelines(str(content[i, :]))
            f.writelines("\n")


def wrtieresults_top_pred(testing_acc_long, testing_acc_activity, testing_acc_long_pred, testing_acc_activity_pred):
    txtfile_name = 'results_top_prediction.txt'
    stop_time = str(timeit.default_timer())
    txtfile_name = stop_time + txtfile_name

    content = np.vstack((testing_acc_long, testing_acc_activity, testing_acc_long_pred, testing_acc_activity_pred))

    print(content)
    if os.path.exists(txtfile_name):
        os.remove(txtfile_name)

    with open(txtfile_name, "a+") as f:
        for i in range(4):
            f.writelines(str(content[i, :]))
            f.writelines("\n")


def wrtieresults_top_recog(testing_acc_long, testing_acc_activity, testing_acc_long_pred, testing_acc_activity_pred):
    txtfile_name = 'results_top_recogniton.txt'
    stop_time = str(timeit.default_timer())
    txtfile_name = stop_time + txtfile_name

    content = np.vstack((testing_acc_long, testing_acc_activity, testing_acc_long_pred, testing_acc_activity_pred))

    print(content)
    if os.path.exists(txtfile_name):
        os.remove(txtfile_name)

    with open(txtfile_name, "a+") as f:
        for i in range(4):
            f.writelines(str(content[i, :]))
            f.writelines("\n")


def wrtie_confusion_results(training_intent_confuse, training_intent_lab_confuse, training_activity_confuse, training_activity_lab_confuse, txtfile_name):
    stop_time = stop_time_init
    txtfile_name1 = stop_time + '_int' + txtfile_name
    txtfile_name2 = stop_time + '_act' + txtfile_name

    content1 = np.vstack((training_intent_confuse, training_intent_lab_confuse))
    content2 = np.vstack((training_activity_confuse, training_activity_lab_confuse))

    np.savetxt(txtfile_name1, np.round(content1), delimiter=',')
    np.savetxt(txtfile_name2, np.round(content2), delimiter = ',')