from datetime import datetime
import socket
import glob
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from MSHARA_dataset import CarDataset_multi_2
from networks import C3Dmodel, fusion_2, prediction_net, res_3_2d_net_chattn, LSTM_ANNO, res_3_2d_net
import MT_loss_multi
from transforms import ConvertBHWCtoBCHW, ConvertBCHWtoCBHW
from fusion_methods import *
from MSHARA_utils import *

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print("Device being used:", device)

nEpochs = 60            # Number of epochs for training
resume_epoch = 0        # Default is 0, change if want to resume
useTest = True          # See evolution of the test set when training
nTestInterval = 10      # Run on test set every nTestInterval epochs
snapshot = nEpochs      # Store a model every snapshot epochs
lr = 1e-4               # Learning rate

stop_time_init = str(timeit.default_timer())
dataset = 'face'  # Options: face or gtea or finegym

# For Brain4Car dataset setup
num_classes = 5
num_activtites = 4

num_tasks = 4
require_mtl = True

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
print('save_dir_root is:', save_dir_root)

exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'MSHARA'
saveName = modelName + '-' + dataset

rand_flg = True
print('savename {}'.format(saveName))


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, num_activities=num_activtites,  lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval, pred_horizon=1):
    print('current training pred_horizon', pred_horizon)
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    fusion_dim = 9 - pred_horizon

    if modelName == 'MSHARA':
        encoder_model = res_3_2d_net.res_3_2d_net()
        fusion_model = fusion_2.Fusion_net_2(num_classes=num_activities, pred_horizon=fusion_dim)
        decoder_model = LSTM_ANNO.LSTM_ANNO(num_classes=num_classes)
        predict_model = prediction_net.prediction_net(num_activity_classes=num_activities, num_intent_classes=num_classes, fusion_ind=False, pred_horizon=fusion_dim)

        if require_mtl:
            mt_loss = MT_loss_multi.MultiTaskLossWrapper(task_num=num_tasks)
            print('loss parameters', list(mt_loss.parameters()))

        train_params = [
                        {'params': encoder_model.parameters(), 'lr': lr},
                        {'params': mt_loss.parameters(), 'lr': lr},
                        {'params': fusion_2.get_1x_lr_params(fusion_model), 'lr': lr},
                        {'params': fusion_model.spat_weigh, 'lr': lr},
                        {'params': fusion_model.temp_weigh, 'lr': lr},
                        {'params': prediction_net.get_1x_lr_params(predict_model), 'lr': lr},
                        {'params': predict_model.spat_weigh, 'lr': lr },
                        {'params': predict_model.temp_weigh, 'lr': lr },
                        {'params': decoder_model.parameters(), 'lr': lr}]

        print('train_params', train_params)
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError

    print('Total params: %.2fM' % (sum(p.numel() for p in encoder_model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.Adam(train_params, lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=120,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU

        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        optimizer.load_state_dict(checkpoint['opt_dict'])

    encoder_model.to(device)
    fusion_model.to(device)
    mt_loss.to(device)
    decoder_model.to(device)
    predict_model.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    print('tensorboard log_dir', log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))

    bat_size = 3
    clp_len = 149           # to avoid abnormal/incomplete sequences

    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

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
                                  batch_size=bat_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = DataLoader(CarDataset_multi_2(dataset=dataset, split='val', clip_len=clp_len,
                                                transform1=transform1_t, transform2=None),
                                batch_size=bat_size, num_workers=4, pin_memory=True)

    test_dataloader = DataLoader(CarDataset_multi_2(dataset=dataset, split='test', clip_len=clp_len,
                                                 transform1=transform1_t, transform2=None),
                                 batch_size=bat_size, shuffle=True, num_workers=4, pin_memory=True)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    training_loss = []
    training_acc_long = []
    training_acc_activity = []
    training_acc_long_pred = []
    training_acc_activity_pred = []

    training_acc_top1 = []
    training_acc_top3 = []
    training_acc_top1_MR = []
    training_acc_top3_MR = []
    training_acc_top1_IP = []
    training_acc_top3_IP = []
    training_acc_top1_MP = []
    training_acc_top3_MP = []

    testing_loss = []
    testing_acc_long = []
    testing_acc_activity = []
    testing_acc_long_pred = []
    testing_acc_activity_pred = []
    testing_acc_top1 = []
    testing_acc_top3 = []
    testing_acc_top1_MR = []
    testing_acc_top3_MR = []
    testing_acc_top1_IP = []
    testing_acc_top3_IP = []
    testing_acc_top1_MP = []
    testing_acc_top3_MP = []

    max_acc = 0

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        print('epoch {}'.format(epoch))
        for phase in ['train']:
            start_time = timeit.default_timer()
            running_loss = 0.0
            running_corrects_long = 0.0
            running_corrects_mid = 0.0
            running_corrects_long_pred = 0.0
            running_corrects_mid_pred = 0.0

            running_corrects_top1 = 0.0
            running_corrects_top3 = 0.0

            running_corrects_top1_MR = 0.0
            running_corrects_top3_MR = 0.0

            running_corrects_top1_IP = 0.0
            running_corrects_top3_IP = 0.0

            running_corrects_top1_MP = 0.0
            running_corrects_top3_MP = 0.0

            # reset the running loss and corrects
            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                optimizer.step()
                scheduler.step()

                encoder_model.train()
                decoder_model.train()
                predict_model.train()
                mt_loss.train()
                fusion_model.train()

            else:
                encoder_model.eval()
                decoder_model.eval()
                predict_model.eval()
                mt_loss.eval()
                fusion_model.eval()

            for inputs, inputs_resnet, labels, seq_labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                labels = labels.clone().detach()
                labels = labels.long()
                labels = Variable(labels, requires_grad=False).to(device)

                batch_size, C, frames, H, W = inputs.shape

                clip_feats = torch.Tensor([]).to(device)
                clip_activity_feats = torch.Tensor([]).to(device)

                init_flag = True
                pre1_sum = torch.Tensor([]).to(device)
                pre3_sum = torch.Tensor([]).to(device)

                loss_mid_tot = 0
                ind = 0

                clip_len = len(np.arange(0, frames-(17+pred_horizon*16), 16))
                pred_ind = frames-17
                for clip_i in np.arange(0, frames-(17+pred_horizon*16), 16):
                    ind += 1
                    clip_seg, clip_img, seqeunce_label = generate_last_prediction_data(clip_i, inputs, inputs_resnet, seq_labels)

                    clip_feats_temp, sig_feats_temp = encoder_model(clip_seg, clip_img)
                    clip_feats_int, activity_pred, activity_output, [spat_weigh, temp_weigh] = fusion_model(clip_feats_temp, sig_feats_temp)

                    loss_mid = criterion(activity_pred, seqeunce_label)
                    loss_mid_tot += loss_mid

                    prec1, prec3 = accuracy(activity_pred.data, seqeunce_label.data, topk=(1, 3))

                    pre1_sum = torch.cat([pre1_sum, prec1])
                    pre3_sum = torch.cat([pre3_sum, prec3])

                    if init_flag:
                        probs_mid_sub = nn.Softmax(dim=1)(activity_pred)
                        preds_mid_sub = torch.max(probs_mid_sub, 1)[1]
                        seq_labs_mid = seqeunce_label
                        init_flag = False
                    else:
                        probs_mid_sub = nn.Softmax(dim=1)(activity_pred)
                        preds_mid_sub_s = torch.max(probs_mid_sub, 1)[1]
                        preds_mid_sub = torch.cat((preds_mid_sub, preds_mid_sub_s))
                        seq_labs_mid = torch.cat((seq_labs_mid, seqeunce_label))

                    clip_feats = torch.cat((clip_feats, clip_feats_int), 1)
                    clip_activity_feats = torch.cat((clip_activity_feats, activity_output), 1)

                print('loss_mid_tot', loss_mid_tot, loss_mid_tot.size())

                prec1_MR = torch.mean(pre1_sum)
                prec3_MR = torch.mean(pre3_sum)

                clip_seg_final, clip_img_final, lab_seq_final = generate_last_prediction_data(pred_ind, inputs, inputs_resnet, seq_labels)
                pred_final_score, gru_output = decoder_model(clip_feats)
                predict_intent_score, predict_activity_score = predict_model(clip_activity_feats, gru_output)
                print('fusion module spaial:', spat_weigh, 'fusion module temporal:', temp_weigh)

                probs_pred_long = nn.Softmax(dim=1)(predict_intent_score)
                preds_pred_long = torch.max(probs_pred_long, 1)[1]

                probs_pred_mid = nn.Softmax(dim=1)(predict_activity_score)
                preds_pred_mid = torch.max(probs_pred_mid, 1)[1]

                probs = nn.Softmax(dim=1)(pred_final_score)
                preds = torch.max(probs, 1)[1]
                preds_mid = preds_mid_sub
                seq_labs = seq_labs_mid

                prec1_LI, prec3_LI = accuracy(pred_final_score.data, labels.data, topk=(1, 3))
                prec1_LP, prec3_LP = accuracy(predict_intent_score.data, labels.data, topk=(1, 3))
                prec1_MP, prec3_MP = accuracy(predict_activity_score.data, lab_seq_final.data, topk=(1, 3))

                if require_mtl:
                    loss_long = criterion(pred_final_score, labels)
                    loss_tot = mt_loss(loss_mid_tot, loss_long, labels, predict_intent_score,
                                       predict_activity_score, lab_seq_final)
                    print('loss_tot', loss_tot, loss_tot.size())
                else:
                    loss_mid = criterion(activity_pred, seqeunce_label)
                    print('loss_mid', loss_mid)
                    loss_long = criterion(pred_final_score, labels)
                    print('loss_long', loss_long)
                    # print('loss is {}', loss)
                    loss_tot = loss_long + (loss_mid)
                    print('loss_tot', loss_tot)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss_tot.backward()
                    optimizer.step()

                running_loss += loss_tot.item() * inputs.size(0)
                running_corrects_long += torch.sum(preds == labels.data)
                running_corrects_mid += torch.sum(preds_mid == seq_labs.data)

                running_corrects_long_pred += torch.sum(preds_pred_long == labels.data)
                running_corrects_mid_pred += torch.sum(preds_pred_mid == lab_seq_final.data)

                running_corrects_top1 += torch.sum(prec1_LI)
                running_corrects_top3 += torch.sum(prec3_LI)

                running_corrects_top1_MR += torch.sum(prec1_MR)
                running_corrects_top3_MR += torch.sum(prec3_MR)

                running_corrects_top1_IP += torch.sum(prec1_LP)
                running_corrects_top3_IP += torch.sum(prec3_LP)

                running_corrects_top1_MP += torch.sum(prec1_MP)
                running_corrects_top3_MP += torch.sum(prec3_MP)

            train_epoch_loss = running_loss / trainval_sizes[phase]
            train_epoch_acc_long = running_corrects_long.double() / trainval_sizes[phase]
            train_epoch_acc_mid = running_corrects_mid.double() / (trainval_sizes[phase]*(clip_len))
            train_epoch_acc_long_pred = running_corrects_long_pred.double() / trainval_sizes[phase]
            train_epoch_acc_mid_pred = running_corrects_mid_pred.double() / trainval_sizes[phase]

            train_epoch_acc_top1 = running_corrects_top1.double() * bat_size / trainval_sizes[phase]
            train_epoch_acc_top3 = running_corrects_top3.double() * bat_size / trainval_sizes[phase]

            train_epoch_acc_top1_MR = running_corrects_top1_MR.double() * bat_size / trainval_sizes[phase]
            train_epoch_acc_top3_MR = running_corrects_top3_MR.double() * bat_size / trainval_sizes[phase]

            train_epoch_acc_top1_IP = running_corrects_top1_IP.double() * bat_size / trainval_sizes[phase]
            train_epoch_acc_top3_IP = running_corrects_top3_IP.double() * bat_size / trainval_sizes[phase]

            train_epoch_acc_top1_MP = running_corrects_top1_MP.double() * bat_size / trainval_sizes[phase]
            train_epoch_acc_top3_MP = running_corrects_top3_MP.double() * bat_size / trainval_sizes[phase]

            train_epoch_acc_cpu_long = train_epoch_acc_long.data.cpu().numpy()
            train_epoch_acc_cpu_mid = train_epoch_acc_mid.data.cpu().numpy()
            train_epoch_acc_cpu_long_pred = train_epoch_acc_long_pred.cpu().numpy()
            train_epoch_acc_cpu_mid_pred = train_epoch_acc_mid_pred.cpu().numpy()

            train_epoch_acc_cpu_top1 = train_epoch_acc_top1.cpu().numpy()
            train_epoch_acc_cpu_top3 = train_epoch_acc_top3.cpu().numpy()

            train_epoch_acc_cpu_top1_MR = train_epoch_acc_top1_MR.cpu().numpy()
            train_epoch_acc_cpu_top3_MR = train_epoch_acc_top3_MR.cpu().numpy()

            train_epoch_acc_cpu_top1_IP = train_epoch_acc_top1_IP.cpu().numpy()
            train_epoch_acc_cpu_top3_IP = train_epoch_acc_top3_IP.cpu().numpy()

            train_epoch_acc_cpu_top1_MP = train_epoch_acc_top1_MP.cpu().numpy()
            train_epoch_acc_cpu_top3_MP = train_epoch_acc_top3_MP.cpu().numpy()

            training_loss = np.append(training_loss, train_epoch_loss)
            training_acc_long = np.append(training_acc_long, train_epoch_acc_cpu_long)
            training_acc_activity = np.append(training_acc_activity, train_epoch_acc_cpu_mid)
            training_acc_long_pred = np.append(training_acc_long_pred, train_epoch_acc_cpu_long_pred)
            training_acc_activity_pred = np.append(training_acc_activity_pred, train_epoch_acc_cpu_mid_pred)

            training_acc_top1 = np.append(training_acc_top1, train_epoch_acc_cpu_top1)
            training_acc_top3 = np.append(training_acc_top3, train_epoch_acc_cpu_top3)
            training_acc_top1_MR = np.append(training_acc_top1_MR, train_epoch_acc_cpu_top1_MR)
            training_acc_top3_MR = np.append(training_acc_top3_MR, train_epoch_acc_cpu_top3_MR)

            training_acc_top1_IP = np.append(training_acc_top1_IP, train_epoch_acc_cpu_top1_IP)
            training_acc_top3_IP = np.append(training_acc_top3_IP, train_epoch_acc_cpu_top3_IP)
            training_acc_top1_MP = np.append(training_acc_top1_MP, train_epoch_acc_cpu_top1_MP)
            training_acc_top3_MP = np.append(training_acc_top3_MP, train_epoch_acc_cpu_top3_MP)

            writer.add_scalar('data/train_loss_epoch', train_epoch_loss, epoch)
            writer.add_scalar('data/train_acc_long_epoch', train_epoch_acc_long, epoch)
            writer.add_scalar('data/train_acc_mid_epoch', train_epoch_acc_mid, epoch)
            writer.add_scalar('data/train_acc_long_pred_epoch', train_epoch_acc_long_pred, epoch)
            writer.add_scalar('data/train_acc_mid_pred_epoch', train_epoch_acc_mid_pred, epoch)

            writer.add_scalar('data/train_top1_epoch', train_epoch_acc_cpu_top1, epoch)
            writer.add_scalar('data/train_top3_epoch', train_epoch_acc_cpu_top3, epoch)
            writer.add_scalar('data/train_epoch_acc_cpu_top1_MR', train_epoch_acc_cpu_top1_MR, epoch)
            writer.add_scalar('data/train_epoch_acc_cpu_top3_MR', train_epoch_acc_cpu_top3_MR, epoch)
            writer.add_scalar('data/train_epoch_acc_cpu_top1_IP', train_epoch_acc_cpu_top1_IP, epoch)
            writer.add_scalar('data/train_epoch_acc_cpu_top3_IP', train_epoch_acc_cpu_top3_IP, epoch)
            writer.add_scalar('data/train_epoch_acc_cpu_top1_MP', train_epoch_acc_cpu_top1_MP, epoch)
            writer.add_scalar('data/train_epoch_acc_cpu_top3_MP', train_epoch_acc_cpu_top3_MP, epoch)

            writer.flush()

            print("[{}] Epoch: {}/{} Loss: {} Intent Acc: {} Activity Acc: {} Pred Intent Acc: {} Pred Activity Acc: {}"
                  .format(phase, epoch + 1, nEpochs, train_epoch_loss, train_epoch_acc_long, train_epoch_acc_mid,
                          train_epoch_acc_long_pred, train_epoch_acc_mid_pred))

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder_model.state_dict(),
                'decoder_state_dict': decoder_model.state_dict(),
                'fusion_state_dict': fusion_model.state_dict(),
                'predict_state_dict': predict_model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth')))

        if useTest and epoch % test_interval == (test_interval - 1):
            encoder_model.eval()
            mt_loss.eval()
            fusion_model.eval()
            decoder_model.eval()
            predict_model.eval()

            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects_long = 0.0
            running_corrects_mid = 0.0
            running_corrects_long_pred = 0.0
            running_corrects_mid_pred = 0.0
            
            running_corrects_top1 = 0.0
            running_corrects_top3 = 0.0
            running_corrects_top1_MR = 0.0
            running_corrects_top3_MR = 0.0
            running_corrects_top1_IP = 0.0
            running_corrects_top3_IP = 0.0
            running_corrects_top1_MP = 0.0
            running_corrects_top3_MP = 0.0

            running_recog_intent = []
            running_recog_label = []

            running_recog_activity = []
            running_recog_seqlab = []

            running_pred_intent = []
            running_pred_label = []

            running_pred_activity = []
            running_pred_seqlab = []

            for inputs, inputs_resnet, labels, seq_labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)
                with torch.no_grad():
                    batch_size, C, frames, H, W = inputs.shape
                    clip_feats = torch.Tensor([]).to(device)
                    clip_activity_feats = torch.Tensor([]).to(device)
                    
                    pre1_sum = torch.Tensor([]).to(device)
                    pre3_sum = torch.Tensor([]).to(device)
                    
                    loss_mid_tot = 0

                    clip_len = len(np.arange(0, frames - (17 + pred_horizon * 16), 16))
                    pred_ind = frames - 17

                    init_flag_test = True
                    for clip_i in np.arange(0, frames - (17+pred_horizon*16), 16):

                        clip_seg, clip_img, seqeunce_label = generate_last_prediction_data(clip_i, inputs,
                                                                                           inputs_resnet, seq_labels)

                        clip_feats_temp, sig_feats_temp = encoder_model(clip_seg, clip_img)
                       
                        clip_feats_int, activity_pred, activity_output, [spat_weigh, temp_weigh] = fusion_model(clip_feats_temp, sig_feats_temp)

                        loss_mid = criterion(activity_pred, seqeunce_label)
                        prec1, prec3 = accuracy(activity_pred.data, seqeunce_label.data, topk=(1, 3))

                        loss_mid_tot += loss_mid

                        pre1_sum = torch.cat([pre1_sum, prec1])
                        pre3_sum = torch.cat([pre3_sum, prec3])

                        if init_flag_test:
                            probs_mid_sub = nn.Softmax(dim=1)(activity_pred)
                            preds_mid_sub = torch.max(probs_mid_sub, 1)[1]
                            seq_labs_mid = seqeunce_label
                            init_flag_test = False
                        else:
                            probs_mid_sub = nn.Softmax(dim=1)(activity_pred)
                            preds_mid_sub_s = torch.max(probs_mid_sub, 1)[1]
                            preds_mid_sub = torch.cat((preds_mid_sub, preds_mid_sub_s))
                            seq_labs_mid = torch.cat((seq_labs_mid, seqeunce_label))

                        clip_feats = torch.cat((clip_feats, clip_feats_int), 1)
                        clip_activity_feats = torch.cat((clip_activity_feats, activity_output), 1)

                prec1_MR = torch.mean(pre1_sum)
                prec3_MR = torch.mean(pre3_sum)

                clip_seg_final, clip_img_final, lab_seq_final = generate_last_prediction_data(pred_ind,inputs,inputs_resnet,seq_labels)

                preds_mid = preds_mid_sub
                seq_labs = seq_labs_mid

                pred_final_score, gru_output = decoder_model(clip_feats)
                predict_intent_score, predict_activity_score = predict_model(clip_activity_feats, gru_output)

                probs_pred_long = nn.Softmax(dim=1)(predict_intent_score)
                preds_pred_long = torch.max(probs_pred_long, 1)[1]

                probs_pred_mid = nn.Softmax(dim=1)(predict_activity_score)
                preds_pred_mid = torch.max(probs_pred_mid, 1)[1]

                probs = nn.Softmax(dim=1)(pred_final_score)
                preds = torch.max(probs, 1)[1]

                prec1_LI, prec3_LI = accuracy(pred_final_score.data, labels.data, topk=(1, 3))
                prec1_LP, prec3_LP = accuracy(predict_intent_score.data, labels.data, topk=(1, 3))
                prec1_MP, prec3_MP = accuracy(predict_activity_score.data, lab_seq_final.data, topk=(1, 3))

                # Recognize intent
                recog_tmp_intent = preds.clone().detach().cpu().numpy()
                recog_lab_intent = labels.clone().detach().cpu().numpy()
                # Recognize activity
                recog_tmp_activity = preds_mid.clone().detach().cpu().numpy()
                lab_seq_act_recog = seq_labs.clone().detach().cpu().numpy()
                # predict intent
                preds_tmp_intent = preds_pred_long.clone().detach().cpu().numpy()
                lab_tmp_intent = recog_lab_intent
                # predict activity
                preds_tmp_act = preds_pred_mid.clone().detach().cpu().numpy()
                lab_seq_act_final = lab_seq_final.data.clone().detach().cpu().numpy()

                if require_mtl:
                    loss_long = criterion(pred_final_score, labels)
                    loss_tot = mt_loss(loss_mid_tot, loss_long, labels, predict_intent_score, predict_activity_score, lab_seq_final)
                else:
                    loss_mid = criterion(activity_pred, seqeunce_label)
                    loss_long = criterion(pred_final_score, labels)
                    loss_tot = loss_long + (loss_mid)

                running_loss += loss_tot.item() * inputs.size(0)
                running_corrects_long += torch.sum(preds == labels.data)
                running_corrects_mid += torch.sum(preds_mid == seq_labs.data)

                running_corrects_long_pred += torch.sum(preds_pred_long == labels.data)
                running_corrects_mid_pred += torch.sum(preds_pred_mid == lab_seq_final.data)

                running_corrects_top1 += torch.sum(prec1_LI)
                running_corrects_top3 += torch.sum(prec3_LI)

                running_corrects_top1_MR += torch.sum(prec1_MR)
                running_corrects_top3_MR += torch.sum(prec3_MR)

                running_corrects_top1_IP += torch.sum(prec1_LP)
                running_corrects_top3_IP += torch.sum(prec3_LP)

                running_corrects_top1_MP += torch.sum(prec1_MP)
                running_corrects_top3_MP += torch.sum(prec3_MP)

                # Generate confusion

                running_recog_intent = np.append(running_recog_intent, recog_tmp_intent)
                running_recog_label = np.append(running_recog_label, recog_lab_intent)

                running_recog_activity = np.append(running_recog_activity, recog_tmp_activity)
                running_recog_seqlab = np.append(running_recog_seqlab, lab_seq_act_recog)

                running_pred_intent = np.append(running_pred_intent, preds_tmp_intent)
                running_pred_label = np.append(running_pred_label, lab_tmp_intent)

                running_pred_activity = np.append(running_pred_activity, preds_tmp_act)
                running_pred_seqlab = np.append(running_pred_seqlab, lab_seq_act_final)

            epoch_loss = running_loss / test_size
            epoch_acc_long = running_corrects_long.double() / test_size
            epoch_acc_mid = running_corrects_mid.double() / (test_size*(clip_len))
            epoch_acc_long_pred = running_corrects_long_pred.double() / test_size
            epoch_acc_mid_pred = running_corrects_mid_pred.double() / test_size

            epoch_acc_top1 = running_corrects_top1.double() * bat_size / test_size
            epoch_acc_top3 = running_corrects_top3.double() * bat_size / test_size

            epoch_acc_top1_MR = running_corrects_top1_MR.double() * bat_size / test_size
            epoch_acc_top3_MR = running_corrects_top3_MR.double() * bat_size / test_size

            epoch_acc_top1_IP = running_corrects_top1_IP.double() * bat_size / test_size
            epoch_acc_top3_IP = running_corrects_top3_IP.double() * bat_size / test_size

            epoch_acc_top1_MP = running_corrects_top1_MP.double() * bat_size / test_size
            epoch_acc_top3_MP = running_corrects_top3_MP.double() * bat_size / test_size

            epoch_acc_cpu_long = epoch_acc_long.data.cpu().numpy()
            epoch_acc_cpu_mid = epoch_acc_mid.data.cpu().numpy()
            epoch_acc_cpu_long_pred = epoch_acc_long_pred.data.cpu().numpy()
            epoch_acc_cpu_mid_pred = epoch_acc_mid_pred.data.cpu().numpy()

            epoch_acc_cpu_top1 = epoch_acc_top1.data.cpu().numpy()
            epoch_acc_cpu_top3 = epoch_acc_top3.data.cpu().numpy()
            epoch_acc_cpu_top1_MR = epoch_acc_top1_MR.data.cpu().numpy()
            epoch_acc_cpu_top3_MR = epoch_acc_top3_MR.data.cpu().numpy()

            epoch_acc_cpu_top1_IP = epoch_acc_top1_IP.data.cpu().numpy()
            epoch_acc_cpu_top3_IP = epoch_acc_top3_IP.data.cpu().numpy()
            epoch_acc_cpu_top1_MP = epoch_acc_top1_MP.data.cpu().numpy()
            epoch_acc_cpu_top3_MP = epoch_acc_top3_MP.data.cpu().numpy()

            testing_loss = np.append(testing_loss, epoch_loss)
            testing_acc_long = np.append(testing_acc_long, epoch_acc_cpu_long)
            testing_acc_activity = np.append(testing_acc_activity, epoch_acc_cpu_mid)
            testing_acc_long_pred = np.append(testing_acc_long_pred, epoch_acc_cpu_long_pred)
            testing_acc_activity_pred = np.append(testing_acc_activity_pred, epoch_acc_cpu_mid_pred)

            testing_acc_top1 = np.append(testing_acc_top1, epoch_acc_cpu_top1)
            testing_acc_top3 = np.append(testing_acc_top3, epoch_acc_cpu_top3)
            testing_acc_top1_MR = np.append(testing_acc_top1_MR, epoch_acc_cpu_top1_MR)
            testing_acc_top3_MR = np.append(testing_acc_top3_MR, epoch_acc_cpu_top3_MR)

            testing_acc_top1_IP = np.append(testing_acc_top1_IP, epoch_acc_cpu_top1_IP)
            testing_acc_top3_IP = np.append(testing_acc_top3_IP, epoch_acc_cpu_top3_IP)
            testing_acc_top1_MP = np.append(testing_acc_top1_MP, epoch_acc_cpu_top1_MP)
            testing_acc_top3_MP = np.append(testing_acc_top3_MP, epoch_acc_cpu_top3_MP)

            if epoch_acc_cpu_long > max_acc:
                max_acc = epoch_acc_cpu_long
                wrtie_confusion_results(running_recog_intent, running_recog_label, running_recog_activity,
                                        running_recog_seqlab, '_recog_max_confusion.txt')
                wrtie_confusion_results(running_pred_intent, running_pred_label, running_pred_activity,
                                        running_pred_seqlab, '_pred_max_confusion.txt')

            writer.add_scalar('data/test_max_acc_epoch', max_acc, epoch)
            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_long_epoch', epoch_acc_long, epoch)
            writer.add_scalar('data/test_acc_mid_epoch', epoch_acc_mid, epoch)
            writer.add_scalar('data/test_acc_long_pred_epoch', epoch_acc_long_pred, epoch)
            writer.add_scalar('data/test_acc_mid_pred_epoch', epoch_acc_mid_pred, epoch)

            writer.add_scalar('data/test_top1_epoch', epoch_acc_cpu_top1, epoch)
            writer.add_scalar('data/test_top3_epoch', epoch_acc_cpu_top3, epoch)
            writer.add_scalar('data/epoch_acc_cpu_top1_MR', epoch_acc_cpu_top1_MR, epoch)
            writer.add_scalar('data/epoch_acc_cpu_top3_MR', epoch_acc_cpu_top3_MR, epoch)
            writer.add_scalar('data/epoch_acc_cpu_top1_IP', epoch_acc_cpu_top1_IP, epoch)
            writer.add_scalar('data/epoch_acc_cpu_top3_IP', epoch_acc_cpu_top3_IP, epoch)
            writer.add_scalar('data/epoch_acc_cpu_top1_MP', epoch_acc_cpu_top1_MP, epoch)
            writer.add_scalar('data/epoch_acc_cpu_top3_MP', epoch_acc_cpu_top3_MP, epoch)

            writer.flush()

            print("[test] Epoch: {}/{} Loss: {} Intent Acc: {} Activity Acc: {} Pred Intent Acc: {} Pred Activity Acc: {}"
                  .format(epoch + 1, nEpochs, epoch_loss, epoch_acc_long, epoch_acc_mid, epoch_acc_long_pred, epoch_acc_mid_pred))

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    wrtieresults_train(training_acc_long, training_acc_activity, training_acc_long_pred, training_acc_activity_pred)
    wrtieresults(testing_acc_long, testing_acc_activity, testing_acc_long_pred, testing_acc_activity_pred)

    wrtieresults_top_recog(testing_acc_top1, testing_acc_top3, testing_acc_top1_MR, testing_acc_top3_MR)
    wrtieresults_top_pred(testing_acc_top1_IP, testing_acc_top3_IP, testing_acc_top1_MP, testing_acc_top3_MP)

    writer.close()

if __name__ == "__main__":
    pred_horizon = np.arange(1,2)   # select prediction horizon
    print(pred_horizon)
    for pre_h in pred_horizon:
        train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, num_activities=num_activtites,  lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval, pred_horizon=pre_h)