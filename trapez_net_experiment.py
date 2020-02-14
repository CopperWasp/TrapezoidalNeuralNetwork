import trapez_net as tn
import preprocess as p
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import copy
import olsf2 as sf
import trapez_net_parameters as tp
import pickle


# printables
feature_list = []
feature_dif_list = [0]
cv_errors_nn = [] # for standard error of mean
cv_errors_olsf = []



def getShuffledTrapezoidalData(data): 
    copydata = copy.deepcopy(data)
    random.shuffle(copydata)
    dataset = p.removeDataTrapezoidal(copydata)
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


def crossValidate(data, datasetName):
    random.seed(tp.seed)
    final_result_olsf = [0 for i in range(len(data))]
    final_result = [0 for i in range(len(data))]
    for i in range(tp.folds):
        X_list, y_list = getShuffledTrapezoidalData(data)
        #print(y_list)
        y_list = [int(i) for i in y_list]
        # olsf
        olsf = sf.olsf(X_list, y_list)
        current_result_olsf = olsf.fit()
        final_result_olsf = np.add(final_result_olsf, current_result_olsf)
        # olsf end
        current_result = experiment(X_list, y_list)
        final_result = np.add(final_result, current_result)
        cv_errors_nn.append(current_result[-1] * len(data))
        cv_errors_olsf.append(current_result_olsf[-1] * len(data))
        #
    final_result_olsf = np.multiply(final_result_olsf, 1/tp.folds)   
    final_result = np.multiply(final_result, 1/tp.folds)
    
    # save both results here to pickle, using the datasetname
    pickle.dump(final_result_olsf, open("pickles/"+datasetName+"_olsf_stream_error.p", "wb"))
    pickle.dump(final_result_olsf, open("pickles/"+datasetName+"_nn_stream_error.p", "wb"))
    # save both cv errors here to pickle
    pickle.dump(cv_errors_olsf, open("pickles/"+datasetName+"_olsf_cv_error.p", "wb"))
    pickle.dump(cv_errors_nn, open("pickles/"+datasetName+"_nn_cv_error.p", "wb"))
    
    x_axis = range(0, len(y_list))
    plt.suptitle(datasetName, fontsize=11)
    plt.xlabel('Instance', fontsize=10)
    plt.ylabel('Error Rate (%)', fontsize=10)
    plt.plot(x_axis, final_result, label="NN")
    plt.plot(x_axis, final_result_olsf, label="OLSF")
    plt.legend(fontsize=11)
    return final_result


def getCriterion():
    if tp.criterion == "L1":
        return torch.nn.L1Loss(size_average=False)
    elif tp.criterion == "MSE":
        return torch.nn.MSELoss(size_average=False)
    elif tp.criterion == "KL":#    phi = 0.05
#    cross_validation_folds = 25
#    random_seed = 50
#    olvf_C = 0.1
#    olvf_Lambda = 30
#    olvf_B = 0.64
#    olvf_option = 1 # 0, 1 or 2
#    stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
#    olsf_C = 0.1
#    olsf_Lambda = 30
#    olsf_B = 0.64
#    olsf_option = 1 # 0, 1 or 2
        return torch.nn.KLDivLoss(size_average=False)
    elif tp.criterion == "SoftMargin":
        return torch.nn.SoftMarginLoss(size_average=False)
    

def getOptimizer(model, learning_rate):
    if tp.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = tp.l2_penalty)
    if tp.optimizer == "ADAM":
       return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if tp.optimizer == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)


def updateLearningRate(t, optimizer):
    for param_group in optimizer.param_groups:
        current_rate = param_group['lr']
        
    if tp.adaptive_lr_switch == True and t>0:
        if feature_dif_list[t] > 0:
            return current_rate * (tn.entropy_list[t-1]) # adaptive entropy scaling, try with t too
        else:
            return current_rate * tp.lr_decay # adaptive decay
    else:
        return current_rate # non adaptive
    

def getOneHot(value):
    if value<0: value = 0
    num_labels = tp.layer_sizes[-1]
    y = [0] * num_labels
    y[value] = 1
    
    return Variable(torch.FloatTensor(y))


def updateErrorList(y_pred, y, t, error_list, false_count):
    if np.argmax(y_pred.data.numpy())!=np.argmax(y.data.numpy()):
        false_count+=1
    error_list.append(false_count/(t+1))
    return error_list, false_count
            
        
def experiment(X_list, y_list):
    # generate model
    model = tn.TrapezNet(tp.layer_sizes)
    criterion = getCriterion()
    optimizer = getOptimizer(model, tp.learning_rate)
    # initialize stats
    error_list = []
    false_count = 0
    for t in range(len(y_list)):
        
        # Forward pass: Compute predicted y by passing x to the model
        x = Variable(torch.FloatTensor(X_list[t]))
   
        # record number of features received for plot
        feature_list.append(np.count_nonzero(X_list[t]))
        feature_dif_list.append(feature_list[t]-feature_list[t-1])
        # update learning rate
        optimizer = getOptimizer(model, updateLearningRate(t, optimizer))
        # possibly masked input features, initial mask is 1s
        masked = model.maskFeatures(x)
        y_pred = model(masked)
        y = getOneHot(y_list[t])
        # Calculate loss
        loss = criterion(y_pred, y) 
        + (tp.first_layer_penalty * np.linalg.norm(model.layers[0].weight.data.numpy(), ord=2))
        #+ (tp.second_layer_penalty) * np.linalg.norm(model.layers[1].weight.data.numpy(), ord=2)
        # Accuracy
        error_list, false_count = updateErrorList(y_pred, y, t, error_list, false_count)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # mask weights for possible pruning
        model.pruneEdges(x)
        model.pruneEdges(x, 1)
        model.maskEdges()
        
    print("final error rate: "+str(false_count/len(y_list)))
    return error_list


error_list = crossValidate(p.readWbcNormalized(), "Dataset: WBC")
#error_list = crossValidate(p.readWdbcNormalized(), "Dataset: WDBC") #lr = 0.1, decay = 0.99
#error_list = crossValidate(p.readIonosphereNormalized(), "Dataset: Ionosphere") #lr =1, decay =0.95 #
#error_list = crossValidate(p.readSpambaseNormalized(), "Dataset: Spambase")
#error_list = crossValidate(p.readA8ANormalized(), "Dataset: A8A")
#error_list = crossValidate(p.readWpbcNormalized(), "Dataset: WPBC") #0.01, decay = 0.95, lamb =0.01
#error_list = crossValidate(p.readGermanNormalized(), "Dataset: German") #lr = 0.01, lamb = 0.1
#error_list = crossValidate(p.readMagicNormalized(), "Dataset: Magic04") #lamb = 1 or more
#error_list = crossValidate(p.readSvmguide3Normalized(), "Dataset: Svmguide3")


def plot_error_bars(dataset_name = "WBC"):
    # width of the bars
    barWidth = 0.1
    # Choose the height of the blue bars
    bars1 = [np.mean(cv_errors_nn)]
    # Choose the height of the cyan bars
    bars2 = [np.mean(cv_errors_olsf)]
    # Choose the height of the error bars (bars1)
    yer1 = [np.std(cv_errors_nn)/np.sqrt(len(cv_errors_nn))]
    # Choose the height of the error bars (bars2)
    yer2 = [np.std(cv_errors_olsf)/np.sqrt(len(cv_errors_nn))]
    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    
    plt.ylim(ymin = np.minimum(np.amin(cv_errors_nn), np.amin(cv_errors_olsf)))
    plt.ylim(ymax = np.maximum(np.amax(cv_errors_nn), np.amax(cv_errors_olsf)))
     
    # Create blue bars
    plt.suptitle("Dataset: "+str(dataset_name), fontsize=11)
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer1, capsize=7, label='NN')
    # Create cyan bars
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', yerr=yer2, capsize=7, label='OLSF')
    # general layout
    #plt.xticks([r + barWidth for r in range(len(bars1))], ['cond_A', 'cond_B', 'cond_C'])
    plt.ylabel('Avg. error')
    plt.legend()
    # Show graphic
    plt.show()


    
#plot_error_bars()

    