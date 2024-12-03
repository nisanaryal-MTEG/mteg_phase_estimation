import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
import os
from torch.autograd import Variable
import csv
import torch.utils.data as data_utils

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt

plt.style.use('ggplot')
from pandas.plotting import scatter_matrix
# import plotly.express as px
from torch.nn.utils.rnn import pad_sequence

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:1")


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1)
        self.gru = nn.LSTM(816, hidden_dim, n_layers, batch_first=True, dropout=drop_prob,bidirectional=True)
        self.fc = nn.Linear(816*2, output_dim)
        self.relu = nn.ReLU()
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
        return out.reshape(-1,self.output_dim)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden



class CSVDataset(Dataset):
    """Chromosome MicroArray dataset."""

    def __init__(self, csv_file, root_dir, transform=None,input_dim =None ):
        """
        Args:
            csv_file (string): Path to the a outline csv file.
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_frame = os.listdir(csv_file+"/"+root_dir)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.input_dim = input_dim

    def __len__(self):
        return len(self.csv_frame)

    def padding(self, csv_path, maxlen=3390):
        df1 = pd.read_csv(csv_path)

        df_map = pd.read_csv(os.path.join(self.csv_file,"map.csv"))
        map_label = df_map["class"].to_list()
        phase = df1["phase"].values.tolist()

        phase = [str(i) for i in phase]
        map_label = [str(i) for i in map_label]
        new_phase = []
        for i in phase:
            for j in map_label:
                if j == i:
                    new_phase.append(map_label.index(j))
                    break

        df1["phase"] = new_phase
        # print(new_phase)
        # dfzero = pd.DataFrame(0, index=np.arange(maxlen), columns=df1.columns)
        # dfzero.iloc[0:df1[df1.columns[0]].count()] = df1
        return df1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print(self.csv_frame[idx])
        csv_name = os.path.join(os.path.join(self.csv_file,
                                self.root_dir),self.csv_frame[idx])
        csv_data = self.padding(csv_name)
        # sample = (csv_data[csv_data.columns[0:13]].values, csv_data['Phase'].values)
        # print(csv_data.columns)
        # print(csv_data[csv_data.columns[0:10]].columns)
        sample = (csv_data[csv_data.columns[0:self.input_dim]].values, csv_data['phase'].values)
        # print(csv_data.columns[0:self.input_dim])
        if self.transform:
            sample = self.transform(sample)
        # print(sample[0].shape)
        return sample
class Rescale(object):
    """
    Rescale by using StandardScaler
    """
    def __call__(self, sample):

        (feature,target) = sample
        scaler = StandardScaler() #MinMaxScaler()
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


hidden_dim = 816 # at stride = 4
# hidden_dim = 1632 # at stride = 2
# hidden_dim = 1088 # at stride = 3

# input_dim = 13
input_dim = 27 # no. of tools plus 
output_dim = 13
n_layers = 2
# transformed_dataset = CSVDataset(csv_file='./split/15/validity', root_dir='train', transform=transforms.Compose([Rescale()]))
# transformed_dataset = CSVDataset(csv_file='./split/random_', root_dir='train', transform=transforms.Compose([Rescale()]))
# transformed_dataset = CSVDataset(csv_file='./cholec/split/validity', root_dir='train', transform=transforms.Compose([Rescale()]))
# transformed_dataset = CSVDataset(csv_file='./cholec/split/validity-minus-mean', root_dir='train', transform=transforms.Compose([Rescale()]))
# transformed_dataset = CSVDataset(csv_file='./48_split', root_dir='train', transform=transforms.Compose([Rescale()]))
# transformed_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/combine_ldg_test', root_dir='train', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
# transformed_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/combine_ralp', root_dir='train', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
# transformed_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/RALP_gt', root_dir='train', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
# transformed_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/bundang', root_dir='train', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
transformed_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/global_voucher_20241031', root_dir='train', transform=transforms.Compose([Rescale()]),input_dim=input_dim)


# val_dataset = CSVDataset(csv_file='./split/15/validity', root_dir='test', transform=transforms.Compose([Rescale()]))
# val_dataset = CSVDataset(csv_file='./split/random_', root_dir='test', transform=transforms.Compose([Rescale()]))
# val_dataset = CSVDataset(csv_file='./cholec/split/validity', root_dir='test', transform=transforms.Compose([Rescale()]))
# val_dataset = CSVDataset(csv_file='./cholec/split/validity-minus-mean', root_dir='test', transform=transforms.Compose([Rescale()]))
# val_dataset = CSVDataset(csv_file='./48_split', root_dir='test', transform=transforms.Compose([Rescale()]))
# val_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/combine_ldg_test', root_dir='test', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
# val_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/combine_ralp', root_dir='test', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
# val_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/RALP_gt', root_dir='test', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
# val_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/bundang', root_dir='test', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
val_dataset = CSVDataset(csv_file='/workspace/data_nia_comb/comb/global_voucher_20241031', root_dir='test', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=False)#, collate_fn=my_collate)


val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)#, collate_fn=my_collate)



model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
criterion = nn.CrossEntropyLoss()
#
optimizer = optim.Adam(model.parameters(), lr=0.0001)
if cuda:
    model.to(device)
    criterion.to(device)
#
TensorF = torch.FloatTensor
# TensorF = torch.cuda.FloatTensor if cuda else torch.FloatTensor
TensorL = torch.LongTensor
# TensorL = torch.cuda.LongTensor if cuda else torch.LongTensor

epochs = 150

accuracy_list = np.zeros((epochs,))
loss_list = np.zeros((epochs,))
# ckpt = os.path.join("./gru_cholec80_25fps.ckpt")
# model.load_state_dict(torch.load(ckpt))
model.train()
accuracy=0
# ckpt = os.path.join("save_model/bundang.ckpt")
ckpt = os.path.join("/workspace/save_model/global_voucher_20241031_v2.ckpt")
# save_ckpt = os.path.join("/workspace/save_model/global_voucher_20241031_v2.ckpt")
model.load_state_dict(torch.load(ckpt))

for epoch in range(epochs):
    running_loss = 0
    running_accuracy = 0
    running_correct_len = 0
    model.train()
    for i_batch, sample_batched in enumerate(dataloader):
        (feature, target) = sample_batched
        train_X = feature.type(TensorF).to(device)
        train_y = torch.squeeze(target.type(TensorL)).to(device)

        batch_size = len(feature)
        # print(train_X.shape)
        # print(train_y.shape)
        h = model.init_hidden(batch_size)
        h = h.data
        model.zero_grad()
        # optimizer.zero_grad()

        output= model(train_X, h)
        # print(train_y)
        loss = criterion(output, train_y.squeeze())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        with torch.no_grad():
            correct = (torch.argmax(output, dim=1) == train_y.squeeze()).type(TensorF)
            # running_accuracy += correct.mean()
            for i in range(len(correct)):
                running_accuracy += correct[i]
            running_correct_len+=len(correct)

    accuracy_list[epoch] = running_accuracy / running_correct_len
    loss_list[epoch] = running_loss / len(dataloader)
    model.eval()
    confusion_matrix = torch.zeros(output_dim, output_dim)
    # print("number of epoch: %d" % epoch + " acc.: %f" % accuracy_list[epoch] + " loss: %f" % loss_list[epoch])
    tot_sum = 0
    tot_sample = 0
    for i_batch, sample_batched in enumerate(val_loader):

        feature, target = sample_batched
        test_X = feature.type(TensorF).to(device)
        test_y = target.type(TensorL).to(device)
        # print(test_y)
        batch_size = len(feature)
        h = model.init_hidden(batch_size)
        h = h.data

        output = model(test_X, h)
        correct = (torch.argmax(output, dim=1) == test_y.squeeze()).type(TensorF)
        # print(len(correct))
        # pred_v, pred_idx = torch.topk(output, k=3)
        sum = 0
        _, preds = torch.max(output.data, 1)
        for t, p in zip(test_y.view(-1), preds.view(-1)):
            # print(t.cpu().detach().numpy())
            # print(p.cpu().detach().numpy())
            confusion_matrix[t.cpu().detach().numpy(), p.cpu().detach().numpy()] += 1

        
        for i in range(len(correct)):
            sum += correct[i]
        print(sum/ len(correct))
        tot_sum += sum
        tot_sample += len(correct)
    print('total test accuracy: {}'.format(tot_sum / tot_sample))
    print(confusion_matrix.numpy().astype(int))
    # print(np.sum(confusion_matrix.numpy(), axis=0).astype(int))
    # print(np.sum(confusion_matrix.numpy(), axis=1).astype(int))
    # print(np.diagonal(confusion_matrix.numpy()).astype(int))
    # print(np.diagonal(confusion_matrix.numpy()).astype(int)/np.sum(confusion_matrix.numpy(), axis=1).astype(int) )
    # if (tot_sum/tot_sample)>accuracy:
        # print("saving.......")
        # accuracy=tot_sum/tot_sample
        # torch.save(model.state_dict(), ckpt)

