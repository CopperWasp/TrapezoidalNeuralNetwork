import torch
import numpy as np
import torch.nn.functional as F
import scipy
from torch.autograd import Variable
import trapez_net_parameters as tp


# plottable statistics are global for easy access
entropy_list = []
feature_masking = True
edge_masking = True


class TrapezNet(torch.nn.Module):
    def __init__(self, layer_sizes): # layer_sizes is a list such as = [10, 5, 5, 2]
        super(TrapezNet, self).__init__()

        # init layers, node masks and edge masks
        self.layer_sizes = layer_sizes
        self.layers = torch.nn.ModuleList()
        self.edge_masks = []
        self.node_masks = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
            self.edge_masks.append(np.ones((layer_sizes[i], layer_sizes[i+1])))
            self.node_masks.append(np.ones(layer_sizes[i]))
            
        np.fill_diagonal(self.edge_masks[0], 0)
            
        # init input feature stats for the trapezoidal stream
        self.feature_stats = np.zeros(layer_sizes[0])
        

    def forward(self, x):
        output1 = x
        output2 = x
        for i in range(len(self.layers)):
            if i == len(self.layers) -1:
                output1 = F.softmax(self.layers[i](output1), dim=0)
                output2 = F.softmax(self.layers[i](output2), dim=0)# softmax
                #output = self.layers[i](output)
            else:
                output1 = self.layers[i](output1)#.clamp(min=0) # reLu
                output2 = self.layers[i](output2).clamp(min=0) # reLu 
                
        e1 = scipy.stats.entropy(output1.data.numpy())
        e2 = scipy.stats.entropy(output2.data.numpy())
        
        if e1 > e2:
            output = output2
            #print("1")
        else:
            output = output1
            #print("2")
        # record the entropy of the output
        entropy_list.append(scipy.stats.entropy(output.data.numpy()))
        
        # update feature stats
        self.updateFeatureStats(x)
        return output # must return for loss
        

    def expandLayer(self, new_size, layer=0): # new_input_size = x.size()[0]
       # only works with layer=0 for now
       # add new nodes and weights to a layer to be expanded to the same size as new_size
       # corresponding edge and node masks should also be expanded
       growing_layer = self.layers[layer]
       current_size = growing_layer.weight.size()[1]
       
       # generate a random set of weights to append to current
       appendix = torch.randn(growing_layer.weight.size()[0], new_size-current_size)
       growed_wts = torch.cat((growing_layer.weight.data, appendix), 1) # order of this is important
       
       # replace the updated layer
       self.layers[layer] = torch.nn.Linear(growed_wts.size()[1], growed_wts.size()[0], bias=False)
       self.layers[layer].weight.data = growed_wts
       
       # expand edge masks
       change = self.layers[layer].weight.size()[1] - self.edge_masks[layer].shape[1]
       appendee = np.ones((self.edge_masks[layer].shape[0], change))
       self.edge_masks[layer] = np.concatenate((self.edge_masks[layer], appendee), axis=1)
       
       # expand node masks
       self.node_masks[layer] = np.append(self.node_masks[layer], ([1]*(new_size-current_size)))
       
       # expand the feature stats
       self.feature_stats = np.append(self.feature_stats, [0]*(new_size-current_size))
       
       
    def updateFeatureStats(self, x):
        self.feature_stats += 1
        for i in range(len(self.feature_stats)):
            if(x.data.numpy()[i]!=0):
                self.feature_stats[i]+=1
                    

    def maskEdges(self): # needs to be called at the end of each iteration
        if edge_masking == True:
            for i in range(len(self.edge_masks)):
                mask = self.edge_masks[i]
                wts = self.layers[i].weight.data.numpy()
                self.layers[i].weight.data = torch.from_numpy(np.multiply(wts, mask.T)).float()
            
        
    # masks the input layer features        
    def maskFeatures(self, x): # needs to be called to transform input data into masked one
        if feature_masking == True:
            masked = torch.mul(x, Variable(torch.FloatTensor(self.node_masks[0])))
            return masked
        else:
            return x
        
     
    def projectWeights(self, weights):
        weights = weights * (tp.lamb / np.linalg.norm(abs(weights), ord=1))
        return weights
        
    # prune nodes and prune edges -> works on masks
    def pruneEdges(self, x, layer=0):
        if tp.edge_pruning_switch == False:
            return
        
        #counts = self.feature_stats / np.linalg.norm(self.feature_stats)
        
        mask = self.edge_masks[layer].T
        #print(layer)
        #print(mask.shape)
        #projected_wts = self.projectWeights(self.layers[layer].weight.data.numpy()*mask)

        wts = abs(self.layers[layer].weight.data.numpy() * mask)
        #print(wts.shape)
        #print()
        proto_wts = wts
        proto_wts[proto_wts==0] +=1000
        sorted_indices = np.argsort(proto_wts, axis=None)
        
        
        nonzero_count = np.count_nonzero(mask)
        #allowed_nonzero = np.count_nonzero(x.data.numpy())
        allowed_nonzero = int(len(x.data.numpy())*tp.pruning_coefficient)


        for i in range(nonzero_count - allowed_nonzero):
            mindex = np.unravel_index(sorted_indices[i], wts.shape)
            mask[mindex] = 0
            
        nonzero_count = np.count_nonzero(mask)
        
        #print(nonzero_count)
        #for i in range(allowed_nonzero - nonzero_count):
            # find a way to generate connections
           # mask[mindex] = 1
            
        self.edge_masks[layer] = mask.T
        #self.layers[layer].weight.data = (torch.FloatTensor(projected_wts))
        
        #print(wts.shape)
        #print(mask.shape)
        #print(max_num_edges)
        #print(wts)
        
