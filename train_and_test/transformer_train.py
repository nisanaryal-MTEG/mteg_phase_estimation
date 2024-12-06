from re import X
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
import os
from torch.autograd import Variable
import csv
import torch.utils.data as data_utils
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


plt.style.use('ggplot')
from pandas.plotting import scatter_matrix
# import plotly.express as px
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore")
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class transformer_model(nn.Module):
    def __init__(self, input_dim, hidden_dim ,nhead,n_layers,output_dim, drop_prob=0.0):
        super(transformer_model, self).__init__()

        self.embedding = nn.Linear(input_dim,hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=drop_prob, max_len= 25000)
        encoder_layers = TransformerEncoderLayer(d_model= hidden_dim, nhead= nhead, dim_feedforward=hidden_dim, dropout=drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.ninp=nhead


        self.decoding = nn.Linear(hidden_dim, output_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute((1, 0, 2))  # changing to sequence, batch, feature

        x = self.embedding(x)  # embedding input feature to fix shape.
        # x = self.embedding(x)* math.sqrt(self.ninp)  # embedding input feature to fix shape.

        x = self.pos_encoder(x) # applying positional encoding

        x = self.transformer_encoder(x, None) #transformer , None is mask

        x = self.decoding(x) # changing feature size to output dim

        x = x.permute((1, 0, 2)) # changing to batch, sequence, output dim

        return x


class CSVDataset(Dataset):
    """Chromosome MicroArray dataset."""

    def __init__(self, root_dir, csv_file, input_dim, transform=None):
        """
        Args:
            csv_file (string): Path to the a outline csv file.
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(root_dir + "/" + csv_file)
        self.csv_frame = self.df["csv_path"].values.tolist()

        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.input_dim = input_dim

    def __len__(self):
        return len(self.csv_frame)

    def padding(self, csv_path, maxlen=3390):
        df1 = pd.read_csv(csv_path)
        df_map = pd.read_csv(os.path.join(self.root_dir,"map.csv"))
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
        # dfzero = pd.DataFrame(0, index=np.arange(maxlen), columns=df1.columns)
        # dfzero.iloc[0:df1[df1.columns[0]].count()] = df1

        return df1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.csv_frame[idx])
        csv_name = os.path.join(self.root_dir,self.csv_frame[idx])
        csv_data = self.padding(csv_name)
        # print(csv_name)
        # sample = (csv_data[csv_data.columns[0:13]].values, csv_data['Phase'].values)
        # print(csv_data)
        # print(csv_data[csv_data.columns[0:self.input_dim]].columns)
        sample = (csv_data[csv_data.columns[0:self.input_dim]].values, csv_data['phase'].values)
        # print(csv_data)
        if self.transform:
            sample = self.transform(sample)
        # print(sample[0].shape)
        return sample
    def calculate_weight(self):
        tot_label=[]
        for file in self.csv_frame:
            csv_name = os.path.join(os.path.join(self.csv_file,
                                                 self.root_dir), file)
            csv_data = self.padding(csv_name)
            target = csv_data['phase'].values
            tot_label.extend(target)
        weight = compute_class_weight(class_weight='balanced', classes=np.unique(np.array(tot_label)),
                             y=np.array(tot_label))

        return weight

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


def start_training(args):
    transformed_dataset = CSVDataset(root_dir=args.dataset, csv_file='comb_train.csv', transform=transforms.Compose([Rescale()]),
                                     input_dim=args.input_dim)
    val_dataset = CSVDataset(root_dir=args.dataset, csv_file='comb_val.csv', transform=transforms.Compose([Rescale()]),
                             input_dim=args.input_dim)
    test_dataset = CSVDataset(root_dir=args.dataset, csv_file='comb_test.csv', transform=transforms.Compose([Rescale()]),
                             input_dim=args.input_dim)

    dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=True,num_workers=16)  # , collate_fn=my_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True,num_workers=8)  # , collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True,num_workers=8)  # , collate_fn=my_collate)



    input_dim = args.input_dim
    output_dim = args.output_dim
    n_layers = args.n_layers
    hidden_dim = 100
    nhead = 4
    n_layers = 2
    drop_prob = 0.4

    model = transformer_model(input_dim,hidden_dim,nhead,n_layers,output_dim,drop_prob=drop_prob)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    #
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.to(args.device)

    epochs = args.epoch

    accuracy_list = np.zeros((epochs,))
    loss_list = np.zeros((epochs,))
    # ckpt = os.path.join("./gru_cholec80_25fps.ckpt")
    # model.load_state_dict(torch.load(ckpt))
    model.train()
    accuracy = 0
    ckpt_file_name = args.dataset.split("/")[-1]
    # ckpt = os.path.join(f"./save/{ckpt_file_name}_TVT_1.ckpt")
    # model.load_state_dict(torch.load(ckpt))

    for epoch in range(epochs):
        running_loss = 0
        running_accuracy = 0
        running_correct_len = 0
        model.train()
        for i_batch, sample_batched in enumerate(dataloader):
            (feature, target) = sample_batched
            train_X = feature.type(torch.FloatTensor).to(args.device)
            train_y = torch.squeeze(target.type(torch.LongTensor)).to(args.device)

            batch_size = len(feature)
            # print(train_X.shape)
            # print(train_y.shape)
            # h = model.init_hidden(batch_size)
            # h = h.data
            model.zero_grad()
            # optimizer.zero_grad()

            output = model(train_X)
            # print(output.squeeze().shape,train_y.squeeze().shape)
            output = output.squeeze()
            loss = criterion(output, train_y.squeeze())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(running_loss,len(dataloader))
            with torch.no_grad():
                correct = (torch.argmax(output, dim=1) == train_y.squeeze()).type(torch.FloatTensor).to(args.device)
                # running_accuracy += correct.mean()
                for i in range(len(correct)):
                    running_accuracy += correct[i]
                running_correct_len += len(correct)
            # if i_batch == 2: break
        accuracy_list = running_accuracy / running_correct_len
        loss_list = running_loss / len(dataloader)
        # confusion_matrix = torch.zeros(10, 10)
        print("number of epoch: %d" % epoch + " acc.: %f" % accuracy_list + " loss: %f" % loss_list)

        model.eval()
        tot_sum = 0
        tot_sample = 0
        for i_batch, sample_batched in enumerate(val_loader):
            feature, target = sample_batched
            test_X = feature.type(torch.FloatTensor).to(args.device)
            test_y = target.type(torch.LongTensor).to(args.device)
            # print(target)
            batch_size = len(feature)

            output = model(test_X)
            output = output.squeeze()

            correct = (torch.argmax(output, dim=1) == test_y.squeeze()).type(torch.FloatTensor).to(args.device)

            tot_sum += torch.sum(correct)
            tot_sample += len(correct)
            # if i_batch == 2:break
        print('total validation accuracy: {}'.format(tot_sum / tot_sample))

        if (tot_sum / tot_sample) > accuracy:
            print("saving.......")
            accuracy = tot_sum / tot_sample
            # torch.save(model.state_dict(), ckpt)
        model.eval()
        tot_sum = 0
        tot_sample = 0
        for i_batch, sample_batched in enumerate(test_loader):
            feature, target = sample_batched
            test_X = feature.type(torch.FloatTensor).to(args.device)
            test_y = target.type(torch.LongTensor).to(args.device)
            # print(test_y)
            batch_size = len(feature)

            output = model(test_X)
            output = output.squeeze()
            correct = (torch.argmax(output, dim=1) == test_y.squeeze()).type(torch.FloatTensor).to(args.device)

            tot_sum += torch.sum(correct)
            tot_sample += len(correct)
            # if i_batch == 2:break
        print('total test accuracy: {}'.format(tot_sum / tot_sample))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="the device to train the model") # "cuda:0"
    parser.add_argument("--dataset", help="the datset name") # /comb/LDG0
    parser.add_argument("--input_dim", type=int, help="input dimension for the model") # 33
    parser.add_argument("--output_dim", type=int, help="output dimension") # 13
    parser.add_argument("--n_layers", type=int, default=2, help="layer of lstm") 
    parser.add_argument("--epoch", type=int, default=100, help="epoch to train") 

    args = parser.parse_args()
    print(args)
    start_training(args)
