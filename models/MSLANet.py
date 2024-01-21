import torch
from torch import nn
from models.LANet import LANet
from models.GradCAM import GradCAM
from utils.utils import select_device
from config import DROPOUT_P, NUM_CLASSES, HIDDEN_SIZE, NUM_DROPOUT_LAYERS

class MSLANet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_layers=HIDDEN_SIZE, dropout_num=NUM_DROPOUT_LAYERS, dropout_p=DROPOUT_P):
        super(MSLANet, self).__init__()
        self.device = select_device()
        self.lanet_model = LANet().to(self.device)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout_num = dropout_num
        self.hidden_layers = hidden_layers
        self.dropout_p = dropout_p
        self.first_fc_layers = [nn.Linear(1, num_classes).to(self.device) for _ in range(dropout_num)] if len(hidden_layers) == 0 else [nn.Linear(1, hidden_layers[0]).to(self.device) for _ in range(dropout_num)]
        self.next_fc_layers = [[nn.Linear(hidden_layers[j-1], hidden_layers[j], bias=False).to(self.device) for j in range(1, len(hidden_layers))] for _ in range(dropout_num)] if len(hidden_layers) > 0 else []
        self.final_fc_layers = [nn.Linear(hidden_layers[-1], num_classes, bias=False).to(self.device) for _ in range(dropout_num)] if len(hidden_layers) > 0 else []
        self.first_batch_norm_layer = nn.BatchNorm1d(hidden_layers[0]).to(self.device) if len(hidden_layers) > 0 else nn.BatchNorm1d(num_classes).to(self.device)
        self.next_batch_norm_layers = [nn.BatchNorm1d(hidden_layers[j]).to(self.device) for j in range(1, len(hidden_layers))] if len(hidden_layers) > 0 else []
        self.final_batch_norm_layer = nn.BatchNorm1d(num_classes).to(self.device) if len(hidden_layers) > 0 else None

    def forward(self, x):
        lanet_output = self.lanet_model(x).to(self.device)
        dropout_preds = []

        for i in range(self.dropout_num):
            if len(self.hidden_layers) == 0:
                if self.first_fc_layers[i].in_features == 1:
                    lanet_shape = lanet_output.shape
                    input_size = lanet_shape[1] * lanet_shape[2] * lanet_shape[3]
                    self.first_fc_layers[i] = nn.Linear(input_size, NUM_CLASSES).to(self.device)

                dropout_layer = self.flatten(lanet_output)
                dropout_layer = self.dropout(dropout_layer)
                dropout_layer = self.relu(dropout_layer)
                dropout_layer = self.first_fc_layers[i](dropout_layer)
                dropout_layer = self.first_batch_norm_layer(dropout_layer)
                dropout_preds.append(dropout_layer)
            else:
                if self.first_fc_layers[i].in_features == 1:
                    lanet_shape = lanet_output.shape
                    input_size = lanet_shape[1] * lanet_shape[2] * lanet_shape[3]
                    self.first_fc_layers[i] = nn.Linear(input_size, self.hidden_layers[0]).to(self.device)

                dropout_layer = self.flatten(lanet_output)
                dropout_layer = self.dropout(dropout_layer)
                for j in range(len(self.hidden_layers)):
                    if j == 0:
                        dropout_layer = self.first_fc_layers[i](dropout_layer)
                        dropout_layer = self.relu(dropout_layer)
                        dropout_layer = self.first_batch_norm_layer(dropout_layer)
                    else:
                        dropout_layer = self.next_fc_layers[i][j-1](dropout_layer)
                        dropout_layer = self.relu(dropout_layer)
                        dropout_layer = self.next_batch_norm_layers[j-1](dropout_layer)
                dropout_layer = self.final_fc_layers[i](dropout_layer)
                dropout_layer = self.final_batch_norm_layer(dropout_layer)
            dropout_preds.append(dropout_layer)
        
        average_dropout_layers = torch.stack(dropout_preds).mean(dim=0)
        return average_dropout_layers