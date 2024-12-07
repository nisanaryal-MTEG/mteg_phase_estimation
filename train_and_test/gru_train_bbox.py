import torch.nn as nn
from torch import optim
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import ast
plt.style.use('ggplot')
# import plotly.express as px
from torch.nn.utils.rnn import pad_sequence

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0")


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
        # print("out",out.shape)
        # out = self.softmax(out) # softmax is not necessary and seems ineffective with GRU
        # out = out.
        return out

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
        df = pd.read_csv(csv_path)

        # for bbox only
        df_bbox = df.iloc[:, 27:-1]
        df_tool = df.iloc[:, :27]
        df_phase = df.iloc[:, -1]
        # print(df_bbox[100:150])

        def process_cell(cell):
            # Convert string to list
            coord_list = ast.literal_eval(cell)
            # Return [0, 0, 0, 0] for empty list or the first bounding box
            return list(coord_list[0]) if coord_list else [0, 0, 0, 0]

        # Create a new DataFrame with expanded bounding boxes
        expanded_data = []
        for column in df_bbox.columns:
            # Process each column to extract bounding box data
            processed_column = df[column].apply(process_cell)
            expanded_data.append(pd.DataFrame(processed_column.tolist(), columns=[f"{column}_x1", f"{column}_y1", f"{column}_x2", f"{column}_y2"]))

        # Concatenate all expanded data into a new DataFrame
        result_df = pd.concat(expanded_data, axis=1)


        final_df = pd.concat([df_tool, result_df,df_phase], axis=1)
        # Display the resulting DataFrame
        # print(final_df[100:120])
        return final_df

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.csv_frame[idx])
        csv_name = os.path.join(os.path.join(self.csv_file,
                                self.root_dir),self.csv_frame[idx])
        csv_data = self.padding(csv_name)

        sample = (csv_data[csv_data.columns[0:self.input_dim]].values, csv_data['phase'].values)
        if self.transform:
            sample = self.transform(sample)
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

input_dim = 131 # no. of tools plus frame + 4*no. of tools
output_dim = 7
n_layers = 2

transformed_dataset = CSVDataset(csv_file='data/baba_box_20241204', root_dir='train', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
val_dataset = CSVDataset(csv_file='data/baba_box_20241204', root_dir='val', transform=transforms.Compose([Rescale()]),input_dim=input_dim)
test_dataset = CSVDataset(csv_file='data/baba_box_20241204', root_dir='test', transform=transforms.Compose([Rescale()]),input_dim=input_dim)

dataloader = DataLoader(transformed_dataset, batch_size=8, shuffle=True,num_workers=8,collate_fn=my_collate)#, collate_fn=my_collate)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True,num_workers=3)#, collate_fn=my_collate)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True,num_workers=3)#, collate_fn=my_collate)


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

epochs = 200

accuracy_list = np.zeros((epochs,))
loss_list = np.zeros((epochs,))

model.train()
accuracy=0
# ckpt = os.path.join("save_model/bundang.ckpt")
ckpt = os.path.join("save/bundang_baba_20241028_exp_box.ckpt")
# model.load_state_dict(torch.load(ckpt))

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
        loss = criterion(output.view(-1, 7), train_y.view(-1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        with torch.no_grad():
            pred_classes = torch.argmax(output, dim=2)  # Shape: [2, 11621]

            correct = (pred_classes == train_y).type(torch.float32)  # Shape: [2, 11621]
            # running_accuracy += correct.mean()
            running_accuracy += correct.sum().item()
            running_correct_len += correct.numel()

    accuracy_list[epoch] = running_accuracy / running_correct_len
    loss_list[epoch] = running_loss / len(dataloader)
    model.eval()
    print("number of epoch: %d" % epoch + " acc.: %f" % accuracy_list[epoch] + " loss: %f" % loss_list[epoch])

    confusion_matrix = torch.zeros(output_dim, output_dim)
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

        pred_classes = torch.argmax(output, dim=2)  # Shape: [2, 11621]
        # print(output.shape, test_y.shape)
        correct = (pred_classes == test_y).type(torch.float32)  # Shape: [2, 11621]
        # running_accuracy += correct.mean()
        running_accuracy = correct.sum().item()
        running_correct_len = correct.numel()
        # print(running_accuracy,running_correct_len)
        sum = 0
        # _, preds = torch.max(output.data, dim=2)  # Shape: [2, 11621]
        for t, p in zip(test_y.view(-1), pred_classes.view(-1)):
            confusion_matrix[t.item(), p.item()] += 1

        tot_sum += running_accuracy
        tot_sample += running_correct_len
    print('total validation accuracy: {}'.format(tot_sum / tot_sample))
    print(confusion_matrix.numpy().astype(int))

    if (tot_sum/tot_sample)>accuracy:
        print("saving.......")
        accuracy=tot_sum/tot_sample
        torch.save(model.state_dict(), ckpt)

    model.eval()
    confusion_matrix = torch.zeros(output_dim, output_dim)
    # print("number of epoch: %d" % epoch + " acc.: %f" % accuracy_list[epoch] + " loss: %f" % loss_list[epoch])
    tot_sum = 0
    tot_sample = 0
    for i_batch, sample_batched in enumerate(test_loader):

        feature, target = sample_batched
        test_X = feature.type(TensorF).to(device)
        test_y = target.type(TensorL).to(device)
        # print(test_y)
        batch_size = len(feature)
        h = model.init_hidden(batch_size)
        h = h.data

        output = model(test_X, h)
        pred_classes = torch.argmax(output, dim=2)  # Shape: [2, 11621]

        correct = (pred_classes == test_y).type(torch.float32)  # Shape: [2, 11621]
        # running_accuracy += correct.mean()
        running_accuracy = correct.sum().item()
        running_correct_len = correct.numel()
        sum = 0
        # _, preds = torch.max(output.data, dim=2)  # Shape: [2, 11621]
        for t, p in zip(test_y.view(-1), pred_classes.view(-1)):
            confusion_matrix[t.item(), p.item()] += 1

        tot_sum += running_accuracy
        tot_sample += running_correct_len
    print('total test accuracy: {}'.format(tot_sum / tot_sample))
    # print(confusion_matrix.numpy().astype(int))
    # if (tot_sum/tot_sample) >0.62 and (tot_sum/tot_sample) < 0.65:
    #     print("saving.......")
    #     accuracy=tot_sum/tot_sample
    #     torch.save(model.state_dict(), ckpt)
    # print(np.sum(confusion_matrix.numpy(), axis=1).astype(int))
    # print(np.diagonal(confusion_matrix.numpy()).astype(int))


