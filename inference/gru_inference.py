import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
import os
from torch.autograd import Variable
import csv
import torch.utils.data as data_utils
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda")


#
# Write the threshold criteria currently in use for the deployment in this section
# area >2000 and color<=1

#
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1)
        self.gru = nn.LSTM(816, hidden_dim, n_layers, batch_first=True, dropout=drop_prob,bidirectional=True)
        self.fc = nn.Linear(816*2, output_dim)
        self.relu = nn.ReLU()
        self.output_dim = output_dim
    
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h):
        x = x.permute((0, 2, 1))  # batch size, channels, sequence
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = out.permute((0, 2, 1))
        out, h = self.gru(out)
        out = self.fc(self.relu(out))  # after GRU relu activation
        # print(out.shape)
        # out = self.softmax(out) # softmax is not necessary and seems ineffective with GRU
        return out.reshape(-1, self.output_dim)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class CSVDataset(Dataset):
    """Chromosome MicroArray dataset."""

    def __init__(self, csv_file, root_dir, transform=None, input_dim=None):
        """
        Args:
            csv_file (string): Path to the a outline csv file.
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.csv_frame = os.listdir(csv_file+"/"+root_dir)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.input_dim = input_dim
    def __len__(self):
        # print(len(self.csv_frame))
        return 1

    def padding(self, csv_path, maxlen=3390):
        df1 = pd.read_csv(csv_path)
        # dfzero = pd.DataFrame(0, index=np.arange(maxlen), columns=df1.columns)
        # dfzero.iloc[0:df1[df1.columns[0]].count()] = df1
        return df1

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # csv_name = os.path.join(os.path.join(self.csv_file,
        #                         self.root_dir),self.csv_frame[idx])
        csv_data = self.padding(self.csv_file)
        # sample = (csv_data[csv_data.columns[0:13]].values, csv_data['Phase'].values)
        # print(csv_data.columns)
        # print(csv_data[csv_data.columns[0:17]].columns)
        sample = (csv_data[csv_data.columns[:self.input_dim]].values, [])

        if self.transform:
            sample = self.transform(sample)
        # print(sample[0].shape)
        return sample


class Rescale(object):
    """
    Rescale by using StandardScaler
    """

    def __call__(self, sample):
        (feature, target) = sample
        scaler = StandardScaler()  # MinMaxScaler()
        # scaler = MinMaxScaler()
        feature_scaled = scaler.fit_transform(feature)
        # print(feature_scaled)
        # return (feature, target)
        return (feature_scaled, target)


def my_collate(batch):
    (feature, target) = zip(*batch)
    x_lens = [len(x) for x in feature]
    y_lens = [len(y) for y in target]
    feature = [item[0] for item in batch]
    feature = [torch.from_numpy(item).float() for item in feature]
    feature = pad_sequence(feature, batch_first=True, padding_value=0)
    target = [item[1] for item in batch]
    target = [torch.from_numpy(item).long() for item in target]
    target = pad_sequence(target, batch_first=True, padding_value=0)

    # return {'feature': feature.numpy(), 'target': target.numpy()}
    return (feature, target)


def predict_phase(csv_path):
    start=time.time()


    hidden_dim = 816  # at stride = 4
    # hidden_dim = 1632 # at stride = 2
    # hidden_dim = 1088 # at stride = 3

    # input_dim = 13
    # input_dim = 16
    input_dim = 15
    output_dim = 13
    n_layers = 2

    val_dataset = CSVDataset(csv_file=csv_path, root_dir='test', transform=transforms.Compose([Rescale()]), input_dim=input_dim)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)  # , collate_fn=my_collate)

    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    if cuda:
        model.cuda()

    TensorF = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    TensorL = torch.cuda.LongTensor if cuda else torch.LongTensor

    epochs = 1

    model.train()
    accuracy = 0
    # ckpt = os.path.join("/workspace/save_model/bundang.ckpt")
    ckpt = os.path.join("/workspace/save_model/ldg_paper_2.ckpt")
    model.load_state_dict(torch.load(ckpt))

    for epoch in range(epochs):

        model.eval()
        confusion_matrix = torch.zeros(15, 15)
        # print("number of epoch: %d" % epoch + " acc.: %f" % accuracy_list[epoch] + " loss: %f" % loss_list[epoch])
        tot_sum = 0
        tot_sample = 0
        for i_batch, sample_batched in enumerate(val_loader):
            feature, target = sample_batched
            test_X = feature.type(TensorF)
            # test_y = target[:, [0]].type(TensorL)
            # print(test_y)
            batch_size = len(feature)
            h = model.init_hidden(batch_size)
            h = h.data

            output = model(test_X, h)
            correct = torch.argmax(output, dim=1)
            # print(correct.shape)
            # print(correct)
            correct = torch.squeeze(correct).cpu().detach().numpy()
            print(correct.shape)
            # pred_v, pred_idx = torch.topk(output, k=3)
            # sum = 0
    # path="/workspace/WGAN/test/SNUH_DC07_JCW0_RLPN_0010_retrain_area.csv"

    # df = pd.read_csv(csv_path)
    # df_temp = pd.DataFrame()
    # df_temp["Frame_No."] = df["Frame_No."]    
    # df_temp["phase"] = df["phase"]    
    df_temp = pd.read_csv(csv_path)

    surgery_dict = {
    0:0,
    1:"PC01",
    2:"PC02",
    3:"PC03",
    4:"PC04",
    5:"PC05",
    6:"PC06",
    7:"PC07",
    8:"PC08",
    9:"PC09",
    10:"PC10",
    11:"PC11",
    12:"PC12",
    13:"PC13",
}
    gt_list=[]
    for i in correct.tolist():
        gt_list.append(surgery_dict.get(i))

    # df_temp["phase"]= gt_list
    df_temp["predict"] = gt_list
    # print(csv_path[:-4] + "_with_phase.csv")
    return df_temp
    # df.to_csv( "GGHB_DC16_LWK0_LDG0_0016_최종_test.csv", index=False)
    # print(len(df.index) // 1000)
    # print(len(df.index) % 1000)
    # print(time.time()-start)


# predict_phase("/workspace/data_refined_full/test/GGHB_DC16_LWK0_LDG0_0016_최종.csv")


test_path = "/workspace/data_nia_comb/comb/ldg_paper/val"

# df = pd.read_csv(os.path.join(test_path,"comb_test.csv"))
files = os.listdir(test_path)
# phase_dict = ""
# files = df['csv_path'].to_list()

for file in files:
    print(file)
    df = predict_phase(os.path.join(test_path,file))
    os.makedirs("output/ldg_paper_full",exist_ok=True)
    df.to_csv(f"output/ldg_paper_full/{file}",index = False)