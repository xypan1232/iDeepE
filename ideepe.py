import sys
import os
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import random
import gzip
import pickle
import timeit
from seq_motifs import get_motif
import argparse
#from sklearn.externals import joblib

if torch.cuda.is_available():
        cuda = True
        #torch.cuda.set_device(1)
        print('===> Using GPU')
else:
        cuda = False
        print('===> Using CPU')
#cuda = False        
def padding_sequence_new(seq, max_len = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq

def read_rna_dict(rna_dict = 'rna_dict'):
    odr_dict = {}
    with open(rna_dict, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict

def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def split_overlap_seq(seq, window_size = 101):
    
    overlap_size = 20
    #pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(num_ins):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            #pdb.set_trace()
            #start = len(seq) -window_size
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, max_len = window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs

def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    
    return seq_list, labels

def get_RNA_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def load_graphprot_data(protein, train = True, path = './GraphProt_CLIP_sequences/'):
    data = dict()
    tmp = []
    listfiles = os.listdir(path)
    
    key = '.train.'
    if not train:
        key = '.ls.'
    mix_label = []
    mix_seq = []
    mix_structure = []    
    for tmpfile in listfiles:
        if protein not in tmpfile:
            continue
        if key in tmpfile:
            if 'positive' in tmpfile:
                label = 1
            else:
                label = 0
            seqs, labels = read_seq_graphprot(os.path.join(path, tmpfile), label = label)
            #pdb.set_trace()
            mix_label = mix_label + labels
            mix_seq = mix_seq + seqs
    
    data["seq"] = mix_seq
    data["Y"] = np.array(mix_label)
    
    return data

def loaddata_graphprot(protein, train = True, ushuffle = True):
    #pdb.set_trace()
    data = load_graphprot_data(protein, train = train)
    label = data["Y"]
    rna_array = []
    #trids = get_6_trids()
    #nn_dict = read_rna_dict()
    for rna_seq in data["seq"]:
        #rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        
        seq_array = get_RNA_seq_concolutional_array(seq)
        #tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
        rna_array.append(seq_array)
    
    return np.array(rna_array), label

def get_bag_data(data, channel = 7, window_size = 101):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        bag_seqs = split_overlap_seq(seq, window_size = window_size)
        #flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)
        
        if num_of_ins >channel:
            start = (num_of_ins - channel)/2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) <channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                #bag_subt.append(random.choice(bag_subt))
		tri_fea = get_RNA_seq_concolutional_array('N'*window_size)
                bag_subt.append(tri_fea.T)
        
        bags.append(np.array(bag_subt))
    
        
    return bags, labels
    #for data in pairs.iteritems():
    #    ind1 = trids.index(key)
    #    emd_weight1 = embedding_rna_weights[ord_dict[str(ind1)]]

def get_bag_data_1_channel(data, max_len = 501):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        #bag_seqs = split_overlap_seq(seq)
        bag_seq = padding_sequence(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)

        
        bags.append(np.array(bag_subt))
    
        
    return bags, labels

def batch(tensor, batch_size = 1000):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        acc_list = []
    	for idx, (X, y) in enumerate(train_loader):
            #for X, y in zip(X_train, y_train):
            #X_v = Variable(torch.from_numpy(X.astype(np.float32)))
             #y_v = Variable(torch.from_numpy(np.array(ys)).long())
    	    X_v = Variable(X)
    	    y_v = Variable(y)
            if cuda:
                X_v = X_v.cuda()
                y_v = y_v.cuda()
                
            self.optimizer.zero_grad()
            y_pred = self.model(X_v)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.data[0])

        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        #X_list = batch(X, batch_size)
        #y_list = batch(y, batch_size)
        #pdb.set_trace()
        print X.shape
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                              torch.from_numpy(y.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        self.model.train()
        for t in range(nb_epoch):
            loss = self._fit(train_loader)
            print loss
            #rint("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    def evaluate(self, X, y, batch_size=32):
        
        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        if cuda:
            y_v = y_v.cuda()
        loss = self.loss_f(y_pred, y_v)
        predict = y_pred.data.cpu().numpy()[:, 1].flatten()
        auc = roc_auc_score(y, predict)
        #lasses = torch.topk(y_pred, 1)[1].data.numpy().flatten()
        #cc = self._accuracy(classes, y)
        return loss.data[0], auc

    def _accuracy(self, y_pred, y):
        return float(sum(y_pred == y)) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(X.astype(np.float32)))
        if cuda:
            X= X.cuda()        
        y_pred = self.model(X)
        return y_pred        

    def predict_proba( X):
	self.model.eval()
        return self.model.predict_proba(X)
        
class CNN(nn.Module):
    def __init__(self, nb_filter, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 12, hidden_size = 200, stride = (1, 1), padding = 0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        out1_size = (window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        maxpool_size = (out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size = (1, 10), stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride = stride))
        out2_size = (maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        maxpool2_size = (out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1
        self.drop1 = nn.Dropout(p=0.25)
	print 'maxpool_size', maxpool_size
        self.fc1 = nn.Linear(maxpool2_size*nb_filter, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
    
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]
    
class CNN_LSTM(nn.Module):
    def __init__(self, nb_filter, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 12, hidden_size = 200, stride = (1, 1), padding = 0, num_layers = 2):
        super(CNN_LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        out1_size = (window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        maxpool_size = (out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1
        self.downsample = nn.Conv2d(nb_filter, 1, kernel_size = (1, 10), stride = stride, padding = padding)
        input_size = (maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        self.layer2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional=True)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.downsample(out)
        out = torch.squeeze(out, 1)
        #pdb.set_trace()
        if cuda:
            x = x.cuda()
            h0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size)).cuda() 
            c0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size)).cuda()
        else:
            h0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size)) 
            c0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size))
        out, _  = self.layer2(out, (h0, c0))
        out = out[:, -1, :]
        #pdb.set_trace()
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

def convR(in_channels, out_channels, kernel_size, stride=1, padding = (0, 1)):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     padding=padding, stride=stride, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, nb_filter = 16, kernel_size = (1, 3), stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = convR(in_channel, nb_filter, kernel_size = kernel_size, stride = stride)
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convR(nb_filter, nb_filter, kernel_size = kernel_size, stride = stride)
        self.bn2 = nn.BatchNorm2d(nb_filter)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, nb_filter = 16, channel = 7, labcounts = 12, window_size = 36, kernel_size = (1, 3), pool_size = (1, 3), num_classes=2, hidden_size = 200):
        super(ResNet, self).__init__()
        self.in_channels = channel
        self.conv = convR(self.in_channels, nb_filter, kernel_size = (4, 10))
        cnn1_size = window_size - 7
        self.bn = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, nb_filter, layers[0],  kernel_size = kernel_size)
        self.layer2 = self.make_layer(block, nb_filter*2, layers[1], 1, kernel_size = kernel_size, in_channels = nb_filter)
        self.layer3 = self.make_layer(block, nb_filter*4, layers[2], 1, kernel_size = kernel_size, in_channels = 2*nb_filter)
        self.avg_pool = nn.AvgPool2d(pool_size)
        avgpool2_1_size = (cnn1_size - (pool_size[1] - 1) - 1)/pool_size[1] + 1
        last_layer_size = 4*nb_filter*avgpool2_1_size
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1,  kernel_size = (1, 10), in_channels = 16):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                convR(in_channels, out_channels, kernel_size = kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, kernel_size = kernel_size, stride = stride, downsample = downsample))
        #self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size = kernel_size))
        return nn.Sequential(*layers)
     
    def forward(self, x):
        #print x.data.cpu().numpy().shape
        #x = x.view(x.size(0), 1, x.size(1), x.size(2))
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        #pdb.set_trace()
        #print self.layer2
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        #pdb.set_trace()
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = F.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
            kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(nn.Module):
    def __init__(self, labcounts = 4, window_size = 107, channel = 7, growth_rate=6, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=12, bn_size=2, drop_rate=0, avgpool_size=(1, 8),
                 num_classes=2):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(channel, num_init_features, kernel_size=(4, 10), stride=1, bias=False)),
        ]))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        last_size = window_size - 7
        # Each denseblock
        num_features = num_init_features
        #last_size =  window_size
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features
                                                            * compression))
                last_size = (last_size - (2 - 1) - 1)/2 + 1
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        avgpool2_1_size = (last_size - (self.avgpool_size[1] - 1) - 1)/self.avgpool_size[1] + 1
        num_features = num_features*avgpool2_1_size
        print num_features
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        #x = x.view(x.size(0), 1, x.size(1), x.size(2))
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
                           features.size(0), -1)
        out = self.classifier(out)
        out = F.sigmoid(out)
        return out
    
    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.features[0](x)
        out = self.features[1](out)
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

def get_all_data(protein, channel = 7):

    data = load_graphprot_data(protein)
    test_data = load_graphprot_data(protein, train = False)
    #pdb.set_trace()
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data)
        test_bags, true_y = get_bag_data_1_channel(test_data)
    else:
        train_bags, label = get_bag_data(data)
    #pdb.set_trace()
        test_bags, true_y = get_bag_data(test_data)

    return train_bags, label, test_bags, true_y

def run_network(model_type, X_train, test_bags, y_train, channel = 7, window_size = 107):
    print 'model training for ', model_type
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter = 16, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = 16, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = 16, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size = window_size, channel = channel, labcounts = 4)
    else:
        print 'only support CNN, CNN-LSTM, ResNet and DenseNet model'

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001),
                loss=nn.CrossEntropyLoss())
    clf.fit(X_train, y_train, batch_size=100, nb_epoch=50)
    
    print 'predicting'         
    pred = model.predict_proba(test_bags)
    return pred, model

def run_ideepe_on_graphprot(model_type = 'CNN', local = False, ensemble = False):
    data_dir = './GraphProt_CLIP_sequences/'
    
    finished_protein = set()
    start_time = timeit.default_timer()
    if local:
        window_size = 107
        channel = 7
        lotext = 'local'
    else:
        window_size = 507
        channel = 1
        lotext = 'global'
    if ensemble:
        outputfile = lotext + '_result_adam_ensemble_' + model_type
    else:
        outputfile = lotext + '_result_adam_individual_' + model_type
        
    fw = open(outputfile, 'w')
    
    for protein in os.listdir(data_dir):
        protein = protein.split('.')[0]
        if protein in finished_protein:
                continue
        finished_protein.add(protein)
        print protein
        fw.write(protein + '\t')
        hid = 16
        if not ensemble:
            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, channel = channel)
            predict = run_network(model_type, np.array(train_bags), np.array(test_bags), np.array(train_labels), protein, channel = channel, window_size = window_size)
        else:
            print 'ensembling'
            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, channel = 1)
            predict1 = run_network(model_type, np.array(train_bags), np.array(test_bags), np.array(train_labels), channel = 1, window_size = 507)
            train_bags, train_labels, test_bags, test_labels = [], [], [], []
            train_bags, train_labels, test_bags, test_labels = get_all_data(protein, channel = 7)
            predict2 = run_network(model_type, np.array(train_bags), np.array(test_bags), np.array(train_labels), channel = 7, window_size = 107)
            predict = (predict1 + predict2)/2.0
            train_bags, train_labels, test_bags = [], [], []
        
        auc = roc_auc_score(test_labels, predict)
        print 'AUC:', auc
        fw.write(str(auc) + '\n')
        mylabel = "\t".join(map(str, test_labels))
        myprob = "\t".join(map(str, predict))  
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
    fw.close()
    end_time = timeit.default_timer()
    print "Training final took: %.2f s" % (end_time - start_time)

def read_data_file(posifile, negafile = None, train = True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label = 1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    
    return data

def get_data(posi, nega = None, channel = 7,  window_size = 101, train = True):
    data = read_data_file(posi, nega, train = train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len = window_size)

    else:
        train_bags, label = get_bag_data(data, channel = channel, window_size = window_size)
    
    return train_bags, label

def detect_motifs(model, test_seqs, X_train, output_dir = 'motifs', channel = 1):
    if channel == 1:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for param in model.parameters():
            layer1_para =  param.data.cpu().numpy()
            break
        	#test_data = load_graphprot_data(protein, train = True)
        	#test_seqs = test_data["seq"]
        N = len(test_seqs)
        if N > 15000: # do need all sequence to generate motifs and avoid out-of-memory
        	sele = 15000
        else:
        	sele = N
        ix_all = np.arange(N)
        np.random.shuffle(ix_all)
        ix_test = ix_all[0:sele]
        
        X_train = X_train[ix_test, :, :, :]
        test_seq = []
        for ind in ix_test:
        	test_seq.append(test_seqs[ind])
        test_seqs = test_seq
        filter_outs = model.layer1out(X_train)[:,:, 0, :]
        get_motif(layer1_para[:,0, :, :], filter_outs, test_seqs, dir1 = output_dir)

def train_network(model_type, X_train, y_train, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16, motif = False, motif_seqs = [], motif_outdir = 'motifs'):
    print 'model training for ', model_type
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size = window_size, channel = channel, labcounts = 4)
    else:
        print 'only support CNN, CNN-LSTM, ResNet and DenseNet model'

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
                loss=nn.CrossEntropyLoss())
    clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs)
    
    if motif and channel == 1:
        detect_motifs(model, motif_seqs, X_train, motif_outdir)

    torch.save(model.state_dict(), model_file)
    #print 'predicting'         
    #pred = model.predict_proba(test_bags)
    #return model

def predict_network(model_type, X_test, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16):
    print 'model training for ', model_type
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'DenseNet':
        model = DenseNet(window_size = window_size, channel = channel, labcounts = 4)
    else:
        print 'only support CNN, CNN-LSTM, ResNet and DenseNet model'

    if cuda:
        model = model.cuda()
                
    model.load_state_dict(torch.load(model_file))
    try:
        pred = model.predict_proba(X_test)
    except: #to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis = 0)
    return pred
        
def run_ideepe(parser):
    #data_dir = './GraphProt_CLIP_sequences/'
    posi = parser.posi
    nega = parser.nega
    model_type = parser.model_type
    ensemble = parser.ensemble
    out_file = parser.out_file
    train = parser.train
    model_file = parser.model_file
    predict = parser.predict
    motif = parser.motif
    motif_outdir = parser.motif_dir
    max_size = parser.maxsize
    channel = parser.channel
    local = parser.local
    window_size = parser.window_size
    ensemble = parser.ensemble
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    glob = parser.glob
    start_time = timeit.default_timer()
    #pdb.set_trace() 
    if predict:
        train = False
        if testfile == '':
            print 'you need specify the fasta file for predicting when predict is True'
            return
    if train:
        if posi == '' or nega == '':
            print 'you need specify the training positive and negative fasta file for training when train is True'
            return
    motif_seqs = []
    if motif:
            train = True
	    local = False
	    glob = True
	    #pdb.set_trace()
            data = read_data_file(posi, nega)
            motif_seqs = data['seq']
            if posi == '' or nega == '':
                print 'To identify motifs, you need training positive and negative sequences using global CNNs.'
                return
   
    if local:
        #window_size = window_size + 6
    	channel = channel
        ensemble = False
        #lotext = 'local'
    elif glob:
        #window_size = maxsize + 6
        channel = 1
        ensemble = False
        window_size = max_size
        #lotext = 'global'
    #if local and ensemble:
    #	ensemble = False
	
    if train:
        if not ensemble:
            train_bags, train_labels = get_data(posi, nega, channel = channel, window_size = window_size)
            model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = channel, window_size = window_size + 6,
                                         model_file = model_file, batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir) 
        else:
            print 'ensembling'
            train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = max_size)
            model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = max_size + 6,
                                  model_file = model_file + '.global', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir)
            train_bags, train_labels = [], []
            train_bags, train_labels = get_data(posi, nega, channel = 7, window_size = window_size)
            model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = window_size + 6,
                                        model_file = model_file + '.local', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir)

            
            end_time = timeit.default_timer()
            print "Training final took: %.2f s" % (end_time - start_time)
    elif predict:
        fw = open(out_file, 'w')
        if not ensemble:
            X_test, X_labels = get_data(testfile, nega = None, channel = channel, window_size = window_size)
            predict = predict_network(model_type, np.array(X_test), channel = channel, window_size = window_size + 6, model_file = model_file, batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        else:
            X_test, X_labels = get_data(testfile, nega = None, channel = 1, window_size = max_size)
            predict1 = predict_network(model_type, np.array(X_test), channel = 1, window_size = max_size + 6, model_file = model_file+ '.global', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
            X_test, X_labels = get_data(testfile, nega = None, channel = 7, window_size = window_size)
            predict2 = predict_network(model_type, np.array(X_test), channel = 7, window_size = window_size + 6, model_file = model_file+ '.local', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
                        
            predict = (predict1 + predict2)/2.0
	#pdb.set_trace()
        #auc = roc_auc_score(X_labels, predict)
        #print auc        
        myprob = "\n".join(map(str, predict))  
        fw.write(myprob)
        fw.close()
    else:
        print 'please specify that you want to train the mdoel or predict for your own sequences'


def parse_arguments(parser):
    parser.add_argument('--posi', type=str, metavar='<postive_sequecne_file>', help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='<negative_sequecne_file>', help='The fasta file of negative training samples')
    parser.add_argument('--model_type', type=str, default='CNN', help='it supports the following deep network models: CNN, CNN-LSTM, ResNet and DenseNet, default model is CNN')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='The output file used to store the prediction probability of the testing sequences')
    parser.add_argument('--motif', type=bool, default=False, help='It is used to identify binding motifs from sequences.')
    parser.add_argument('--motif_dir', type=str, default='motifs', help='The dir used to store the prediction binding motifs.')
    parser.add_argument('--train', type=bool, default=True, help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='model.pkl', help='The file to save model parameters. Use this option if you want to train on your sequences or predict for your sequences')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--testfile', type=str, default='',  help='the test fast file for sequences you want to predict for, you need specify it when using predict')
    parser.add_argument('--maxsize', type=int, default=501, help='For global sequences, you need specify the maxmimum size to padding all sequences, it is only for global CNNs (default value: 501)')
    parser.add_argument('--channel', type=int, default=7, help='The number of channels for breaking the entire RNA sequences to multiple subsequences, you can specify this value only for local CNNs (default value: 7)')
    parser.add_argument('--window_size', type=int, default=101, help='The window size used to break the entire sequences when using local CNNs, eahc subsequence has this specified window size, default 101')
    parser.add_argument('--local', type=bool, default=False, help='Only local multiple channel CNNs for local subsequences')
    parser.add_argument('--glob', type=bool, default=False, help='Only global multiple channel CNNs for local subsequences')
    parser.add_argument('--ensemble', type=bool, default=True, help='It runs the ensembling of local and global CNNs, you need specify the maxsize (default 501) for global CNNs and window_size (default: 101) for local CNNs')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--num_filters', type=int, default=16, help='The number of filters for CNNs (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs (default value: 50)')

    args = parser.parse_args()
    return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print args
#model_type = sys.argv[1]
run_ideepe(args)

