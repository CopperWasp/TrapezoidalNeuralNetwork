import random
import torch
from torch.autograd import Variable
import preprocess as p
import preprocess_multi as pm
import copy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import olsf2 as sf


# 1. try splice, isolet, hapt in nn. If looks better find a way to plot a comparison.
# 2. if any of the above works, you can also go for their real life data streams rcv1 and URL.




seed = 123


#per layer learning rates
# make sure its generalized for multiclass
folds = 5

full_input_size = 240
d_in_multiplier = int(np.sqrt(full_input_size))



adaptive_lr_switch = 1
prune_switch = 0

lamb = 10
l2_regularizer = 0.1
learning_rate = 0.1
decay = 0.98



# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_out = 1, 3

# list of entropies
entropy_list = []
feature_list = []
loss_list = []
feature_dif_list = [0] # needs an initial value

# label range
max_label = -1




def crossValidate(data, datasetName):

    #final_result_olsf = [0 for i in range(len(data))]
    final_result = [0 for i in range(len(data))]
    for i in range(folds):
        X_list, y_list = getShuffledData(data)
        y_list = [int(i) for i in y_list]
        # for multiclass
        global max_label
        max_label = np.max(y_list)
        # olsf
        #olsf = sf.olsf(X_list, y_list)
        #current_result_olsf = olsf.fit()
        #final_result_olsf = np.add(final_result_olsf, current_result_olsf)
        # olsf end
        current_result = experiment(X_list, y_list)
        final_result = np.add(final_result, current_result)
        #
    #final_result_olsf = np.multiply(final_result_olsf, 1/folds)   
    final_result = np.multiply(final_result, 1/folds)
    
    x_axis = range(0, len(y_list))
    plt.suptitle(datasetName, fontsize=11)
    
    plt.xlabel('Instance', fontsize=10)
    plt.ylabel('Error Rate (%)', fontsize=10)
    
    plt.plot(x_axis, final_result, label="NN")
    #plt.plot(x_axis, final_result_olsf, label="OLSF")
    plt.legend(fontsize=11)
    return final_result   
        

def set_learning_rate(optimizer, epoch, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    
def experiment(X_list, y_list):
    D_in = len(truncTrapez(X_list[0], 0, len(X_list))) # input dimension is # features
    H = D_in*d_in_multiplier
    model = DynamicNet(D_in, H, D_out)
    criterion = torch.nn.L1Loss(size_average=False)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = l2_regularizer)
    error_list = []
    false_count = 0
    prev_rate = learning_rate
    for t in range(len(y_list)):
        
        # print lr
        #for param_group in optimizer.param_groups:
         #   print(param_group['lr'])
         
        # Forward pass: Compute predicted y by passing x to the model
        x = Variable(torch.FloatTensor(truncTrapez(X_list[t], t, len(X_list))))
        
        # record number of features received for plot
        feature_list.append(x.data.shape[0])
        feature_dif_list.append(feature_list[t]-feature_list[t-1])
 
        # keep track of lengths
        len_prev = feature_list[t-1]

        # adaptive LR
        if adaptive_lr_switch == 1:
            if len(x)>len_prev:
                #rate = learning_rate*(np.sqrt(len(x)-len_prev))
                rate = learning_rate / (entropy_list[t-1])
                
                if prune_switch == 1:
                     #model.projectWeights(int(np.sqrt(len(x))))
                     model.pruneEdges(len(x))
                # update the mask
                for i in range(len(x)-len_prev):
                    model.mask.append(1)
                    
            else:
                rate = prev_rate * decay

            optimizer = torch.optim.SGD(model.parameters(), lr=rate, weight_decay = l2_regularizer)
            #optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay = l2_regularizer)
            prev_rate = rate
        # update length  
        len_prev = len(x)
        
        # extending the weights
        if model.input_linear.weight.data.shape[1]<x.data.shape[0]:
            model.transferWeights(x)
            
        # update weight stats right after extending, if happends    
        model.updateWeightStats()
        model.updateWeightMask()
        
        
        # do prediction
        #t_mask = Variable(torch.FloatTensor(model.mask))
        
        # masked input
       #y_pred = model(torch.mul(x, t_mask))
        y_pred = model(x)
        
        
        # Prepare y, binary class label
        y = [0] * (max_label+1)
        y[y_list[t]] = 1

        
        y = Variable(torch.FloatTensor(y))
    
        # Calculate loss
        loss = criterion(y_pred, y)
        #print(loss)
        loss_list.append(loss.data.numpy())
        
        
        # Accuracy
        #print(np.argmax(y_pred.data.numpy()))
        #print(np.argmax(y.data.numpy()))
        
        if np.argmax(y_pred.data.numpy())!=np.argmax(y.data.numpy()):
            false_count+=1
            
        error_list.append(false_count/(t+1))

        
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.maskWeights()
    print("final error rate: "+str(false_count/len(y_list)))
    return error_list


# generate data for cross validation
def getShuffledData(data): 
    copydata = copy.deepcopy(data)
    global seed
    random.seed(seed)
    random.shuffle(copydata)
    dataset = p.removeDataTrapezoidal(copydata)
    #dataset = copydata
    all_keys = set().union(*(d.keys() for d in dataset))

    X,y = [],[]
    for row in dataset:
        for key in all_keys:
            if key not in row.keys() : row[key]=0
        y.append(row['class_label'])
        del row['class_label']
    if 0 not in row.keys(): start=1
    if 0 in row.keys(): start=0
    for row in dataset:
        X_row=[]
        for i in range(start, len(row)):
            X_row.append(row[i])
        X.append(X_row)
    return X, y   


def truncTrapez(array, i, rows):
    d = len(array)
    multiplier=int(i/int(rows/40))+1
    increment=int(d/40)
    features_left=multiplier*increment
    return array[:features_left]
      
        
class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
       # self.middle_linear1 = torch.nn.Linear(H, H)
       # self.middle_linear2 = torch.nn.Linear(H, H*2)
       # self.middle_linear3 = torch.nn.Linear(H*2, H)
        self.output_linear = torch.nn.Linear(H, D_out)
        self.weight_stats = np.zeros((H, D_in))
        self.mask = [1] * D_in
        self.weight_mask = np.ones((H, D_in))

        
    def forward(self, x):
        h_relu = self.input_linear(x)#.clamp(min=0)
       # h_relu = self.middle_linear1(h_relu).clamp(min=0)
       # h_relu = self.middle_linear2(h_relu).clamp(min=0)
       # h_relu = self.middle_linear3(h_relu).clamp(min=0)
        y_pred = F.softmax(self.output_linear(h_relu))
        
        # calculate the entropy of the output
        global entropy_list
        entropy_list.append(scipy.stats.entropy(y_pred.data.numpy()))   
        return y_pred
    
    
    def transferWeights(self, x, layer=None):
        if layer==None:
            layer = self.input_linear
        current_input_size = layer.weight.size()[1] #input dimension?
        new_input_size = x.size()[0]   
        a = torch.randn(layer.weight.size()[0], new_input_size-current_input_size)
        b = torch.cat((a, self.input_linear.weight.data), 1)

        self.input_linear = torch.nn.Linear(b.size()[1], b.size()[0])
        self.input_linear.weight.data = b
        
        
    def updateWeightStats(self):
        self.weight_stats += 1
        if(self.input_linear.weight.size()[1] != self.weight_stats.shape[1]):
            change = self.input_linear.weight.size()[1] - self.weight_stats.shape[1]
            appendee = np.ones((self.weight_stats.shape[0], change))
            self.weight_stats = np.concatenate((self.weight_stats, appendee), axis=1)
        
    
    def getNthLargestWeight(self, n):
        wts = self.input_linear.weight.data.numpy()
        flat = np.absolute(wts.flatten())
        flat.sort()
        return flat[-n]
    
        
    def pruneNodes(self, n):
        threshold = int(np.sqrt(n))
        entropies = []
        wts = self.input_linear.weight.data.numpy()

        # calculate entropies of each wt.
        for f in range(wts.shape[1]):
            feature_weights = wts[:, f]
            #absoluted = np.absolute(feature_weights) / np.sum(np.absolute(feature_weights))
            #entropy = scipy.stats.entropy(absoluted)
            entropies.append(np.sum(np.square(feature_weights)) / (self.weight_stats[0, f]))
            #entropies.append(np.sum(np.square(feature_weights)))
            
        entropies.sort()
        lower_bound = entropies[-threshold]
        for f in range(wts.shape[1]):
            feature_weights = wts[:, f]
            #absoluted = np.absolute(feature_weights) / np.sum(np.absolute(feature_weights))
            entropy = (np.sum(np.square(feature_weights)) / (self.weight_stats[0, f]))
            #entropy = (np.sum(np.square(feature_weights)))
            if entropy<lower_bound:
                self.mask[f] = 0
                
    
    def updateWeightMask(self):
        change = self.input_linear.weight.size()[1] - self.weight_mask.shape[1]
        appendee = np.ones((self.weight_mask.shape[0], change))
        self.weight_mask = np.concatenate((self.weight_mask, appendee), axis=1)
        
    
    def maskWeights(self):
        masked = np.multiply(self.input_linear.weight.data.numpy(), self.weight_mask)
        self.input_linear.weight.data = torch.from_numpy(masked).float()
        
            
    def pruneEdges(self, n):
        wts = self.input_linear.weight.data.numpy()
        norm = np.linalg.norm(np.absolute(wts),ord=2)
        projected_wts = np.multiply(lamb/norm, wts)
        lower_bound = sorted(np.absolute(wts.flatten()))[-n]
        
        for row in range(wts.shape[0]):
            for col in range(wts.shape[1]):
                if(wts[row, col]<lower_bound):
                    self.weight_mask[row, col] = 0
        
        self.input_linear.weight.data = torch.from_numpy(np.multiply(projected_wts, self.weight_mask)).float()
                       
        
    def projectWeights(self, n):
        threshold = int(np.sqrt(n))
        wts = self.input_linear.weight.data.numpy()


        # calculate entropies of each wt.
        for f in range(wts.shape[1]):
            feature_weights = wts[:, f]
            #print(feature_weights)
            if np.linalg.norm(feature_weights, ord=1) != 0:

                 projected_weights = np.multiply(lamb/np.linalg.norm(np.absolute(wts), ord=1), feature_weights)
                 sorted_wts = sorted(np.absolute(projected_weights))
                 thr = sorted_wts[-threshold]
                 #print(thr)
                 projected_weights[abs(projected_weights) < thr] = 0
                 wts[:, f] = projected_weights
                 self.mask[f] = 0
            
        self.input_linear.weight.data = torch.from_numpy(wts).float()
  
        threshold = int(np.sqrt(n))
        wts = self.output_linear.weight.data.numpy()

        # calculate entropies of each wt.
        for f in range(wts.shape[1]):
            feature_weights = wts[:, f]
            if np.linalg.norm(feature_weights, ord=1) != 0:
                 projected_weights = np.multiply(lamb/np.linalg.norm(np.absolute(wts), ord=1), feature_weights)
                 sorted_wts = sorted(np.absolute(projected_weights))
                 thr = sorted_wts[-threshold]
                 wts[:, f] = projected_weights
            
        self.output_linear.weight.data = torch.from_numpy(wts).float()
            
            #check this back
        
            
            
        
        # set the new weights back
        

        
        # you can check expected absolute value of the distribution
        # make use of mean and variance
        # entropy of the absolute value of the weights
            # normalize the weight
            # expected information that this guy conveys
            # use also its weight value
            
            
            
            
        
        
        
        
        
        
        
# adjustable learning rate
# sparsity







# we are better
#error_list = crossValidate(p.readWpbcNormalized(), "Dataset: WPBC") #0.01, decay = 0.95, lamb =0.01
#error_list = crossValidate(p.readGermanNormalized(), "Dataset: German") #lr = 0.01, lamb = 0.1
#error_list = crossValidate(p.readMagicNormalized(), "Dataset: Magic04") #lamb = 1 or more
#error_list = crossValidate(p.readSvmguide3Normalized(), "Dataset: Svmguide3")
#error_list = crossValidate(p.readA8ANormalized(), "Dataset: A8A")
#error_list = crossValidate(p.readIonosphereNormalized(), "Dataset: Ionosphere") #lr =0.1, decay =0.98 #

# they are better

#error_list = crossValidate(p.readWdbcNormalized(), "Dataset: WDBC") #lr = 0.1, decay = 0.99
#error_list = crossValidate(p.readSpambaseNormalized(), "Dataset: Spambase")
#error_list = crossValidate(p.readWbcNormalized(), "Dataset: WBC")
error_list = crossValidate(pm.readSplice(), "Dataset: Splice")

# measure correlation bw. entropy and number of features
# if |spearman r| > |pearson r| there is monotonic correlation but nonlinear
#print(pearsonr(entropy_list, feature_list))
#print(spearmanr(entropy_list, feature_list))