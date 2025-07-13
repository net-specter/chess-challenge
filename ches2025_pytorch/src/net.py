import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import numpy as np

# Custom activation functions for better performance
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# Noise reduction algorithms
class NoiseReduction(nn.Module):
    """
    Implements multiple noise reduction techniques:
    1. Savitzky-Golay Filter - for smoothing while preserving peaks
    2. Gaussian Filter - for general noise reduction
    3. Median Filter - for impulse noise removal
    4. Adaptive Denoising - learnable noise reduction
    """
    def __init__(self, method='savgol', window_length=11, polyorder=3):
        super(NoiseReduction, self).__init__()
        self.method = method
        self.window_length = window_length
        self.polyorder = polyorder
        
        # Learnable denoising layer
        if method == 'adaptive':
            self.denoising_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2)
            nn.init.constant_(self.denoising_conv.weight, 1/5)  # Initialize as averaging filter
    
    def forward(self, x):
        if self.method == 'savgol':
            return self.savitzky_golay_filter(x)
        elif self.method == 'gaussian':
            return self.gaussian_filter(x)
        elif self.method == 'median':
            return self.median_filter(x)
        elif self.method == 'adaptive':
            return self.denoising_conv(x)
        else:
            return x
    
    def savitzky_golay_filter(self, x):
        """Savitzky-Golay filter for noise reduction while preserving features"""
        batch_size, channels, length = x.shape
        filtered = torch.zeros_like(x)
        
        for b in range(batch_size):
            for c in range(channels):
                trace = x[b, c, :].cpu().numpy()
                if len(trace) >= self.window_length:
                    filtered_trace = signal.savgol_filter(trace, self.window_length, self.polyorder)
                    filtered[b, c, :] = torch.from_numpy(filtered_trace).to(x.device)
                else:
                    filtered[b, c, :] = x[b, c, :]
        
        return filtered
    
    def gaussian_filter(self, x):
        """Gaussian filter for general noise reduction"""
        kernel_size = 5
        sigma = 1.0
        kernel = torch.exp(-0.5 * (torch.arange(kernel_size, dtype=torch.float32) - kernel_size//2)**2 / sigma**2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1).to(x.device)
        
        padding = kernel_size // 2
        x_padded = F.pad(x, (padding, padding), mode='reflect')
        return F.conv1d(x_padded, kernel, groups=1)
    
    def median_filter(self, x):
        """Median filter for impulse noise removal"""
        kernel_size = 5
        padding = kernel_size // 2
        x_padded = F.pad(x, (padding, padding), mode='reflect')
        
        # Unfold to create sliding windows
        unfolded = x_padded.unfold(2, kernel_size, 1)
        # Apply median along the last dimension
        filtered, _ = torch.median(unfolded, dim=-1)
        return filtered

class MLP(nn.Module):
    def __init__(self, search_space,num_sample_pts, classes):
        super(MLP, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]

        self.layers = nn.ModuleList()

        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(num_sample_pts, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))

            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax_layer(x) #F.softmax()
        x = x.squeeze(1)
        return x



class CNN(nn.Module):
    def __init__(self, search_space,num_sample_pts, classes):
        super(CNN, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]
        self.conv_layers = search_space["conv_layers"]

        self.layers = nn.ModuleList()
        #CNN
        self.kernels, self.strides, self.filters, self.pooling_type, self.pooling_sizes, self.pooling_strides, self.paddings = create_cnn_hp(search_space)
        num_features = num_sample_pts
        for layer_index in range(0, self.conv_layers):
            #Convolution layer
            new_out_channels = self.filters[layer_index]
            if layer_index == 0:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.kernels[layer_index]
                new_num_features = cal_num_features_conv1d(num_features,kernel_size = self.kernels[layer_index], stride = self.kernels[layer_index], padding = self.paddings[layer_index])
                if new_num_features <=0:
                    conv1d_kernel = 1
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=1,
                                                               stride=1,
                                                               padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(nn.Conv1d(in_channels=1, out_channels=new_out_channels, kernel_size=conv1d_kernel,
                                             stride=conv1d_stride, padding=self.paddings[layer_index]))

            else:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.kernels[layer_index]
                new_num_features = cal_num_features_conv1d(num_features, kernel_size=self.kernels[layer_index],
                                                       stride=self.kernels[layer_index],
                                                       padding=self.paddings[layer_index])
                if new_num_features <= 0:
                    conv1d_kernel = 1
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=1,
                                                               stride=1,
                                                               padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(nn.Conv1d(in_channels=prev_out_channels, out_channels=new_out_channels, kernel_size=conv1d_kernel,
                                             stride=conv1d_stride, padding=self.paddings[layer_index]))
            #Activation Function
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
            #Pooling Layer
            if self.pooling_type[layer_index] == "max_pool":
                layer_pool_size = self.pooling_sizes[layer_index]
                layer_pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_maxpool1d(num_features, layer_pool_size, layer_pool_stride)

                if new_num_features <= 0:
                    layer_pool_size = 1
                    layer_pool_stride = 1
                    new_num_features = cal_num_features_maxpool1d(num_features, 1, 1)
                num_features = new_num_features
                self.layers.append(nn.MaxPool1d(kernel_size=layer_pool_size, stride=layer_pool_stride))
            elif self.pooling_type[layer_index] == "average_pool":
                pool_size = self.pooling_sizes[layer_index]
                pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_avgpool1d(num_features, pool_size, pool_stride)
                if new_num_features <= 0:
                    pool_size = 1
                    pool_stride = 1
                    new_num_features = cal_num_features_maxpool1d(num_features, 1, 1)
                num_features = new_num_features
                self.layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride))
            #BatchNorm
            self.layers.append(nn.BatchNorm1d(new_out_channels))
            prev_out_channels = new_out_channels
        #MLP
        self.layers.append(nn.Flatten())
        #Flatten
        flatten_neurons =prev_out_channels*num_features
        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(flatten_neurons, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))
            #Activation layer
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax_layer(x) #F.softmax()
        x = x.squeeze(1)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, search_space, num_sample_pts, classes):
        super(CNN_LSTM, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]
        self.conv_layers = search_space["conv_layers"]
        self.lstm_layers = search_space.get("lstm_layers", 2)
        self.lstm_hidden_size = search_space.get("lstm_hidden_size", 256)
        self.dropout = search_space.get("dropout", 0.1)

        # Add noise reduction as first layer
        self.noise_reduction = NoiseReduction(method='savgol', window_length=11, polyorder=3)
        
        self.layers = nn.ModuleList()
        
        # CNN Feature Extraction Part
        self.kernels, self.strides, self.filters, self.pooling_type, self.pooling_sizes, self.pooling_strides, self.paddings = create_cnn_hp(search_space)
        num_features = num_sample_pts
        
        prev_out_channels = 1
        for layer_index in range(0, self.conv_layers):
            # Convolution layer
            new_out_channels = self.filters[layer_index]
            if layer_index == 0:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.strides[layer_index]  # Use strides instead of kernels
                new_num_features = cal_num_features_conv1d(num_features, kernel_size=conv1d_kernel, 
                                                         stride=conv1d_stride, padding=self.paddings[layer_index])
                if new_num_features <= 0:
                    conv1d_kernel = 3
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=3, stride=1, 
                                                             padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(nn.Conv1d(in_channels=1, out_channels=new_out_channels, 
                                           kernel_size=conv1d_kernel, stride=conv1d_stride, 
                                           padding=self.paddings[layer_index]))
            else:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.strides[layer_index]  # Use strides instead of kernels
                new_num_features = cal_num_features_conv1d(num_features, kernel_size=conv1d_kernel,
                                                         stride=conv1d_stride, padding=self.paddings[layer_index])
                if new_num_features <= 0:
                    conv1d_kernel = 3
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=3, stride=1,
                                                             padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(nn.Conv1d(in_channels=prev_out_channels, out_channels=new_out_channels, 
                                           kernel_size=conv1d_kernel, stride=conv1d_stride, 
                                           padding=self.paddings[layer_index]))
            
            # Improved Activation Functions
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
            elif self.activation == 'gelu':
                self.layers.append(GELU())
            elif self.activation == 'swish':
                self.layers.append(Swish())
            
            # Pooling Layer
            if self.pooling_type[layer_index] == "max_pool":
                layer_pool_size = self.pooling_sizes[layer_index]
                layer_pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_maxpool1d(num_features, layer_pool_size, layer_pool_stride)
                if new_num_features <= 0:
                    layer_pool_size = 2
                    layer_pool_stride = 2
                    new_num_features = cal_num_features_maxpool1d(num_features, 2, 2)
                num_features = new_num_features
                self.layers.append(nn.MaxPool1d(kernel_size=layer_pool_size, stride=layer_pool_stride))
            elif self.pooling_type[layer_index] == "average_pool":
                pool_size = self.pooling_sizes[layer_index]
                pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_avgpool1d(num_features, pool_size, pool_stride)
                if new_num_features <= 0:
                    pool_size = 2
                    pool_stride = 2
                    new_num_features = cal_num_features_avgpool1d(num_features, 2, 2)
                num_features = new_num_features
                self.layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride))
            
            # BatchNorm
            self.layers.append(nn.BatchNorm1d(new_out_channels))
            prev_out_channels = new_out_channels
        
        # Store final CNN output dimensions for LSTM
        self.cnn_output_channels = prev_out_channels
        self.cnn_output_length = num_features
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=prev_out_channels,  # CNN output channels
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Final classification layers with improved activations
        self.fc_layers = nn.ModuleList()
        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.fc_layers.append(nn.Linear(self.lstm_hidden_size, self.neurons))
            else:
                self.fc_layers.append(nn.Linear(self.neurons, self.neurons))
            
            # Improved activation layers
            if self.activation == 'relu':
                self.fc_layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.fc_layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.fc_layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.fc_layers.append(nn.ELU())
            elif self.activation == 'gelu':
                self.fc_layers.append(GELU())
            elif self.activation == 'swish':
                self.fc_layers.append(Swish())
            
            # Add batch normalization for better training stability
            self.fc_layers.append(nn.BatchNorm1d(self.neurons))
            # Add dropout for regularization
            self.fc_layers.append(nn.Dropout(self.dropout * 0.5))  # Reduced dropout for FC layers
        
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # Apply noise reduction first
        x = self.noise_reduction(x)
        
        # CNN feature extraction
        for layer in self.layers:
            x = layer(x)
        
        # Prepare for LSTM: transpose to (batch_size, sequence_length, features)
        x = x.transpose(1, 2)  # From (batch, channels, length) to (batch, length, channels)
        
        # LSTM processing with improved handling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output of LSTM for classification
        x = lstm_out[:, -1, :]  # Take the last timestep
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Final classification layers with improved activations
        for layer in self.fc_layers:
            x = layer(x)
        
        x = self.softmax_layer(x)
        x = x.squeeze(1) if x.dim() > 1 and x.size(1) == 1 else x
        return x


def cal_num_features_conv1d(n_sample_points,kernel_size, stride,padding = 0, dilation = 1):
        L_in = n_sample_points
        L_out = math.floor(((L_in +(2*padding) - dilation *(kernel_size -1 )-1)/stride )+1)
        return L_out


def cal_num_features_maxpool1d(n_sample_points, kernel_size, stride, padding=0, dilation=1):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return L_out

def cal_num_features_avgpool1d(n_sample_points,kernel_size, stride, padding = 0):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - kernel_size ) / stride) + 1)
    return L_out


def create_cnn_hp(search_space):
    pooling_type = search_space["pooling_types"]
    pool_size = search_space["pooling_sizes"] #size == stride
    conv_layers = search_space["conv_layers"]
    init_filters = search_space["filters"]
    init_kernels = search_space["kernels"] #stride = kernel/2
    init_padding = search_space["padding"] #only for conv1d layers.
    kernels = []
    strides = []
    filters = []
    paddings = []
    pooling_types = []
    pooling_sizes = []
    pooling_strides = []
    for conv_layers_index in range(1, conv_layers + 1):
        if conv_layers_index == 1:
            filters.append(init_filters)
            kernels.append(init_kernels)
            strides.append(int(init_kernels / 2))
            paddings.append(init_padding)
        else:
            filters.append(filters[conv_layers_index - 2] * 2)
            kernels.append(kernels[conv_layers_index - 2] // 2)
            strides.append(int(kernels[conv_layers_index - 2] // 4))
            paddings.append(init_padding)
        pooling_sizes.append(pool_size)
        pooling_strides.append(pool_size)
        pooling_types.append(pooling_type)
    return kernels, strides, filters, pooling_type, pooling_sizes, pooling_strides, paddings




def weight_init(m, type = 'kaiming_uniform_'):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if type == 'xavier_uniform_':
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('selu'))
        elif type == 'he_uniform':
            nn.init.kaiming_uniform_(m.weight)
        elif type == 'random_uniform':
            nn.init.uniform_(m.weight)
        if m.bias != None:
            nn.init.zeros_(m.bias)




def create_hyperparameter_space(model_type):
    if model_type == "mlp":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                                                   "lr": random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                                    "optimizer": random.choice( ["RMSprop", "Adam"]),
                                                    "layers": random.randrange(1, 8, 1),
                                                    "neurons": random.choice( [10, 20, 50, 100, 200, 300, 400, 500]),
                                                    "activation": random.choice(  ["relu", "selu", "elu", "tanh"]),
                                                    "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
                                                }
        return search_space
    elif model_type == "cnn":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                                              "lr":random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                              "optimizer":random.choice(["RMSprop", "Adam"]),
                                              "layers": random.randrange(1, 8, 1),
                                              "neurons": random.choice( [10, 20, 50, 100, 200, 300, 400, 500]),
                                              "activation": random.choice( ["relu", "selu", "elu", "tanh"]),
                                              "kernel_initializer": random.choice( ["random_uniform", "glorot_uniform", "he_uniform"]),
                                              "pooling_types": random.choice(["max_pool", "average_pool"]),
                                              "pooling_sizes":random.choice(  [2,4,6,8,10]), #size == strides
                                              "conv_layers": random.choice( [1,2,3,4]),
                                              "filters": random.choice( [4,8,12,16]),
                                              "kernels": random.choice( [i for i in range(26,53,2)]), #strides = kernel/2
                                              "padding": random.choice(  [0,4,8,12,16]),
                                        }

        return search_space
    elif model_type == "cnn-lstm":
        search_space = {"batch_size": random.choice([64, 128, 256, 512]),  # Better batch sizes for deep learning
                                              "lr": random.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]),  # Higher learning rates
                                              "optimizer": random.choice(["Adam", "AdamW"]),  # Better optimizers
                                              "layers": random.randrange(2, 4, 1),  # More reasonable number of FC layers
                                              "neurons": random.choice([128, 256, 512, 1024]),  # Larger neurons for complex patterns
                                              "activation": random.choice(["relu", "gelu", "swish"]),  # Better activations
                                              "kernel_initializer": random.choice(["he_uniform", "xavier_uniform"]),
                                              # CNN parameters - optimized for feature extraction
                                              "pooling_types": random.choice(["max_pool", "average_pool"]),
                                              "pooling_sizes": random.choice([2, 3, 4]),  # Smaller pooling for better feature preservation
                                              "conv_layers": random.choice([2, 3, 4]),  # More conv layers for better feature extraction
                                              "filters": random.choice([32, 64, 128]),  # More filters for richer features
                                              "kernels": random.choice([i for i in range(8, 17, 2)]),  # Smaller kernels for local patterns
                                              "padding": random.choice([2, 4, 6, 8]),  # Better padding
                                              # LSTM specific parameters - optimized
                                              "lstm_layers": random.choice([2, 3]),  # More LSTM layers for temporal learning
                                              "lstm_hidden_size": random.choice([256, 512, 1024]),  # Larger hidden sizes
                                              "dropout": random.choice([0.1, 0.2, 0.3]),  # Lower dropout for better learning
                                        }
        return search_space