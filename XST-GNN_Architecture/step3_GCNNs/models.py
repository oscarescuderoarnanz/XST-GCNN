import torch
import torch.nn.functional as F
import torch.nn as nn


####################################################################
######## Standard GCNN with Normalized Adjacency Matrix ############
###################################################################

class standard_gcnn_layer(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN) acting as a low-pass filter.

    Args:
        S (torch.Tensor): Graph shift operator (e.g., adjacency matrix).
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        seed (int): Seed value for reproducibility.
    """
    
    def __init__(self, S, in_dim, out_dim, seed):
        super().__init__()
        
        torch.manual_seed(seed)

        # Assign values from the configuration and dataset to self for later use
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.S = S.clone()
        
        # Add self-loops by adding an identity matrix to the graph shift operator
        self.S += torch.eye(self.S.shape[0], device=self.S.device)
        self.d = self.S.sum(1)
        self.D_inv = torch.diag(1 / torch.sqrt(self.d))
        # Compute the symmetric normalized adjacency matrix for the message passing algorithm:
        self.S = self.D_inv @ self.S @ self.D_inv
        self.S = nn.Parameter(self.S, requires_grad=False)
        
        # In this step, the parameter requires gradient
        # Generate a weight matrix
        self.W = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        # Initialize weights using Kaiming uniform initialization
        self.initial_W = nn.init.kaiming_uniform_(self.W.data)

        # Initialize the bias
        self.b = nn.Parameter(torch.empty(self.out_dim))
        # Calculate the standard deviation
        std = 1/(self.in_dim*self.out_dim)
        # Initialize bias uniformly between -std and std
        self.initial_b = nn.init.uniform_(self.b.data, -std, std)
        
        
    def forward(self, x):
        """
        Forward pass of the GCN layer.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output features after applying the GCN layer.
        """
        return self.S @ x @ self.W + self.b[None, :]

class standard_gcnn(nn.Module):
    """
    A multi-layer Graph Convolutional Network (GCN) acting as a low-pass filter.

    Args:
        n_layers (int): Number of GCN layers.
        dropout (float): Dropout rate for regularization.
        hid_dim (int): Hidden layer dimensionality.
        S (torch.Tensor): Graph shift operator.
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        fc_layer (list of tuples): Configuration of the fully connected layer.
        seed (int): Seed value for reproducibility.
        nonlin (nn.Module): Non-linearity to apply after each GCN layer (default: nn.LeakyReLU).
    """
    
    def __init__(self, n_layers, dropout, hid_dim, S,
                 in_dim, out_dim, fc_layer, seed,
                 nonlin=nn.LeakyReLU):
        
        super().__init__()

        self.convs = nn.ModuleList()
        self.nonlin = nonlin()
        self.n_layers = n_layers
        self.dropout = dropout

        if self.n_layers > 1:
            self.convs.append(standard_gcnn_layer(S, in_dim, hid_dim, seed))
            for i in range(self.n_layers - 2):
                in_dim = hid_dim
                self.convs.append(standard_gcnn_layer(S, in_dim, hid_dim, seed))
            in_dim = hid_dim
            self.convs.append(standard_gcnn_layer(S, in_dim, out_dim, seed))
        else:
            self.convs.append(standard_gcnn_layer(S, in_dim, out_dim, seed))
            
        self.classifier = nn.Sequential(
            nn.Linear(fc_layer[0][0], fc_layer[0][1]),
            nn.Dropout(dropout),
        )
        self.first_layer_weights = self.classifier[0].weight

        
    def forward(self, x):
        """
        Forward pass through the multi-layer GCN.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple: Contains the output after the fully connected layer, importance before FC,
                   weights of the first FC layer, pre-sigmoid output, and first layer's GCN weights.
        """
        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = self.convs[-1](x)
        importance_pre_fc = x
        x = x.view(x.shape[0], x.shape[1])
        
        output = self.classifier(x)
        pre_sigmoid = output
        output = torch.sigmoid(output)    

        return output, importance_pre_fc, self.first_layer_weights, pre_sigmoid, self.convs[0].W
        
        
    def get_classifier(self):
        """
        Get the classifier part of the network.

        Returns:
            nn.Sequential: The fully connected classifier.
        """
        return self.classifier

    
###################################################################
################ Higher-Order Polynomial GCNN #####################
###################################################################
class higher_order_polynomial_gcnn_layer(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN) acting as a bank of filters.

    Args:
        S (torch.Tensor): Graph shift operator (e.g., adjacency matrix).
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        K (int): The order of the filter (number of powers of S).
        seed (int): Seed value for reproducibility.
        bias (bool): Whether to include a bias term (default: True).
    """
    
    def __init__(self, S, in_dim, out_dim, K, seed, bias=True):
        super().__init__()
 
        torch.manual_seed(seed)
    
        self.S = S.clone()
        self.N = self.S.shape[0]
        self.S += torch.eye(self.N, device=self.S.device)
        self.d = self.S.sum(1)
        self.D_inv = torch.diag(1 / torch.sqrt(self.d))
        self.S = self.D_inv @ self.S @ self.D_inv
 
        self.K = K
        self.Spow = torch.zeros((self.K, self.N, self.N), device=self.S.device)
        self.Spow[0,:,:] = torch.eye(self.N, device=self.S.device)
        for k in range(1, self.K):
            self.Spow[k,:,:] = self.Spow[k-1,:,:] @ self.S
 
        self.Spow = nn.Parameter(self.Spow, requires_grad=False)
        self.S = nn.Parameter(self.S, requires_grad=False)
 
        self.in_dim = in_dim
        self.out_dim = out_dim
 
        # Parameters alpha that multiply each S
        self.W = nn.Parameter(torch.empty(self.K, self.in_dim, self.out_dim))
        self.initial_W = nn.init.kaiming_uniform_(self.W.data)
 
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            std = 1/(self.in_dim*self.out_dim)
            self.initial_b = nn.init.uniform_(self.b.data, -std, std)
 
 
    def forward(self, x):
        """
        Forward pass of the GCN layer using a bank of filters.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output features after applying the bank of filters.
        """
        out = torch.zeros((x.shape[0], self.N, self.out_dim), device=x.device)        
        for k in range(self.K):
            out += self.Spow[k,:,:] @ x @ self.W[k,:,:]
 
        if self.bias:
            return out + self.b[None, :]
        else:
            return out
        
class higher_order_polynomial_gcnn(nn.Module):
    """
    A multi-layer Graph Convolutional Network (GCN) acting as a bank of filters.

    Args:
        n_layers (int): Number of GCN layers.
        dropout (float): Dropout rate for regularization.
        hid_dim (int): Hidden layer dimensionality.
        S (torch.Tensor): Graph shift operator.
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        K (int): The order of the filter.
        fc_layer (list of tuples): Configuration of the fully connected layer.
        seed (int): Seed value for reproducibility.
        nonlin (nn.Module): Non-linearity to apply after each GCN layer (default: nn.LeakyReLU).
    """
    
    def __init__(self, n_layers, dropout, hid_dim, S,
                 in_dim, out_dim, K, fc_layer,
                 seed, nonlin=nn.LeakyReLU):
        
        super().__init__()

        self.convs = nn.ModuleList()
        self.nonlin = nonlin()
        self.n_layers = n_layers
        self.dropout = dropout

        if self.n_layers > 1:
            self.convs.append(higher_order_polynomial_gcnn_layer(S, in_dim, hid_dim, K, seed))
            for i in range(self.n_layers - 2):
                in_dim = hid_dim
                self.convs.append(higher_order_polynomial_gcnn_layer(S, in_dim, hid_dim, K, seed))
            in_dim = hid_dim
            self.convs.append(higher_order_polynomial_gcnn_layer(S, in_dim, out_dim, K, seed))
        else:
            self.convs.append(higher_order_polynomial_gcnn_layer(S, in_dim, out_dim, K, seed))
            
        self.classifier = nn.Sequential(
            nn.Linear(fc_layer[0][0], fc_layer[0][1]),
            nn.Dropout(dropout),
        )
        self.first_layer_weights = self.classifier[0].weight.clone().detach()

    def forward(self, x):
        """
        Forward pass through the multi-layer GCN using a bank of filters.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple: Contains the output after the fully connected layer, importance before FC,
                   weights of the first FC layer, pre-sigmoid output, and the first layer's GCN weights and S.
        """
        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = self.convs[-1](x)
        importance_pre_fc = x
        x = x.view(x.shape[0], x.shape[1])
        
        output = self.classifier(x)   
        pre_sigmoid = output
        output = torch.sigmoid(output)    

        return output, importance_pre_fc, self.first_layer_weights, pre_sigmoid, [self.convs[0].W, self.convs[0].S]
      
    def get_classifier(self):
        """
        Get the classifier part of the network.

        Returns:
            nn.Sequential: The fully connected classifier.
        """
        return self.classifier
