import numpy as np
import preprocess
import random
import copy
import matplotlib.pyplot as plt
import time
import OLSF as sf
import olmf as mf
import parameters as p
import scipy

results = []
classifier_dimensions = []


def streaming_experiment(data, dataset_name):
    

    olsf = sf.olsf(data)
    olmf = mf.olmf(data)

    plot_list=[
 
               ("OLsf", olsf.fit, 1),
               ("OLmf", olmf.fit, 1)
               ]
    
    x = list(range(len(data)))
    plotlist=[]
    styles = [['0.5', '8', '-'], ['0.25', '^', '--'], ['0', '.', ':']] 

    for triple in plot_list:
        if triple[2]==1: plotlist.append((triple[1](), triple[0]))
        
    for i in range(len(plotlist)):
        plt.plot(x[1:][0::int(len(x)/10)], plotlist[i][0][1:][0::int(len(x)/10)], label=plotlist[i][1], marker=styles[i][1], linestyle=styles[i][2], color= styles[i][0])
        #plt.plot(x[1:][:], plotlist[i][0][1:][:], label=plotlist[i][1], marker='o', linestyle='--')
  
#    print("Values_vf:"+str(len(plotlist[0][0][1:][:])))
#    print(plotlist[0][0][1:][:])
#    print("Values_sf:"+str(len(plotlist[1][0][1:][:])))
#    print(plotlist[1][0][1:][:])
#
#      
#    print("OLSF mean error:"+str(np.mean(olsf.error_stats)))
#    print("OLSF stdev error:"+str(np.std(olsf.error_stats)))  
#    print("OLSF stdev error:"+str(np.std(olsf.error_stats)/np.sqrt(p.cross_validation_folds)))
#    print()
#    print("OLVF mean error:"+str(np.mean(olvf.error_stats)))    
#    print("OLVF stdev error:"+str(np.std(olvf.error_stats)))
#    print("OLVF stdev error:"+str(np.std(olvf.error_stats/np.sqrt(p.cross_validation_folds))))
#    print("welch's t-test")
#    print(scipy.stats.ttest_ind(olsf.error_stats, olvf.error_stats, equal_var = False))
    
    plt.legend(fontsize=13)
    plt.xlabel("Instance", fontsize=13)
    plt.ylabel("Test Error Rate %", fontsize=13)
    plt.title(dataset_name, fontsize=13)
    plt.grid()
    figurename = './figures/'+'olvf_stream'+str("_")+time.strftime("%H%M%S")+'.png'
    plt.savefig(figurename)
    
    plt.show()
    #X,y = getSampleData(data, p.stream_mode) # to be plotted
    #feature_summary=[np.count_nonzero(row) for row in X]
    #plotFeatures(feature_summary, dataset_name)

      
#    # width of the bars
#    barWidth = 0.1
#    # Choose the height of the blue bars
#    bars1 = [np.mean(olsf.error_stats)]
#    # Choose the height of the cyan bars
#    bars2 = [np.mean(olvf.error_stats)]
#    # Choose the height of the error bars (bars1)
#    yer1 = [np.std(olsf.error_stats)/np.sqrt(p.cross_validation_folds)]
#    # Choose the height of the error bars (bars2)
#    yer2 = [np.std(olvf.error_stats)/np.sqrt(p.cross_validation_folds)]
#    # The x position of bars
#    r1 = np.arange(len(bars1))
#    r2 = [x + barWidth for x in r1]
#    
#    plt.ylim(ymin = np.minimum(np.amin(olsf.error_stats), np.amin(olvf.error_stats)))
#    plt.ylim(ymax = np.maximum(np.amax(olsf.error_stats), np.amax(olvf.error_stats)))
#     
#    # Create blue bars
#    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer1, capsize=7, label='olsf')
#    # Create cyan bars
#    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', yerr=yer2, capsize=7, label='olvf')
#    # general layout
#    #plt.xticks([r + barWidth for r in range(len(bars1))], ['cond_A', 'cond_B', 'cond_C'])
#    plt.ylabel('Avg. error')
#    plt.legend()
#    # Show graphic
#    plt.show()
#    
#    #
#    global results
#    results.append({"C": p.olvf_C, "B": p.olvf_B, "phi": p.phi, "mean_olvf": np.mean(olvf.error_stats), "mean_olsf": np.mean(olsf.error_stats), "std": np.std(olvf.error_stats)})
##    
    
    return dataset_name


def plotFeatures(feature_summary, dataset_name):
    xx = np.array(range(0, len(feature_summary)))
    yy = np.array(feature_summary)
    plt.plot(xx, yy, label=dataset_name, marker='o', linestyle='--')
    plt.legend(loc="upper right", fontsize=20)
    plt.xlabel("Training examples", fontsize=20)
    plt.ylabel("Number of features", fontsize=20)
    plt.grid()
    plt.savefig('./figures/'+dataset_name+time.strftime("%Y%m%d-%H%M%S")+'features.png')
    plt.show()
    plt.clf()


def getSampleData(data, mode='trapezoidal'): # only for feature plot
    copydata= copy.deepcopy(data)
    random.shuffle(copydata)
    if mode=='trapezoidal': 
         dataset=preprocess.removeDataTrapezoidal(copydata)
    if mode=='variable': 
        dataset=preprocess.removeRandomData(copydata)
    else:
         "ERROR, WRONG STREAM_MODE"
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
    

def streamOverUCIDatasets():
    #figurename = streaming_experiment(preprocess.readUrlNormalized(1), "URL")
    #figurename = streaming_experiment(preprocess.readWbcNormalized(), "WBC")
    #figurename = streaming_experiment(preprocess.readGermanNormalized(), "German")
    figurename = streaming_experiment(preprocess.readIonosphereNormalized(), "Ionosphere")
   # figurename = streaming_experiment(preprocess.readMagicNormalized(), "Magic")
    #figurename = streaming_experiment(preprocess.readSpambaseNormalized(), "Spambase")
    #figurename = streaming_experiment(preprocess.readWdbcNormalized(), "WDBC")
    #figurename = streaming_experiment(preprocess.readWpbcNormalized(), "WPBC")
    #figurename = streaming_experiment(preprocess.readA8ANormalized(), "A8A")
   # figurename = streaming_experiment(preprocess.readSvmguide3Normalized(), "svmguide3")
    p.saveParameters(figurename)
 
    

start_time = time.time()
for i in range (0, 1):
     print(p.random_seed)
     streamOverUCIDatasets()
     p.random_seed+=1
     
print(results)
print("--- %s seconds ---" % (time.time() - start_time))

