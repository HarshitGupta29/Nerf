from math import gamma
from tkinter import HIDDEN
import torch
import extras

class nerf(torch.nn.Module): 
    def init(self, hidder_layer = 8, layer_func = torch.nn.Linear, layer_size = 256, output_layer_size = 128):
        super(nerf, self).init()
        self.hidder_layer = hidder_layer
        self.layer_func = layer_func
        self.layer_size = layer_size
        self.input_size = EMBEDD_X * 3 * 2
        self.output_layer_size = output_layer_size
        self.layers = torch.nn.ModuleDict()
        ## each layer corresponds to one hidder layer 
        self.layers[0] = layer_func(self.input_size, layer_size)
        for i in range(1, hidder_layer):
            self.layers[i] = layer_func(layer_size, layer_size) if i != 4 else layer_func(layer_size + self.input_size, layer_size)
        self.layers[hidder_layer] = layer_func(layer_size, layer_size) # for 256 feature vector
        self.alpha = torch.nn.Linear(layer_size, 1)
        self.layers[hidder_layer + 1] = layer_func(layer_size + EMBEDD_D * 3 * 2, output_layer_size)
        for i in range(3):
            self.layers[hidder_layer + 2 + i] = layer_func(output_layer_size, output_layer_size)
        self.layers[hidder_layer + 5] = layer_func(output_layer_size,3)
        self.activation = torch.nn.functional.relu
        

    def forward(self, gamma_x: torch.Tensor, gamma_d: torch.Tensor) -> torch.Tensor:
        x = self.layers[i](gamma_x)
        for i in range(1,self.hidden_layer):
            x = self.layers[i](x) if i != 4 else self.layers[i](torch.cat((x, gamma_x), -1))
            x = self.activation(x)
        feature = self.layers[self.hidder_layer](x)
        alpha = self.alpha(feature)
        feature = self.activation(feature)
        x = self.layers[self.hidder_layer+1](torch.cat((feature, gamma_d), -1))
        for i in range(3):
            x = self.activation(self.layers[self.hidder_layer + 2 + i](x))
        rgb = self.layers[self.hidder_layer + 5](x) 
        return torch.cat((rgb, alpha),-1)

    def backprop(self):
        return 

    





