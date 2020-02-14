import numpy as np
import preprocess
import random
import copy
import parameters as p


class olmf:
    def __init__(self, data):
        self.C=p.olsf_C
        self.Lambda=p.olsf_Lambda
        self.B=p.olsf_B
        self.option=p.olsf_option
        self.data=data
        self.rounds = 1
        self.error_stats = []
        
        # additions for memory
        
    def set_classifier(self):
        self.weights= np.zeros(np.count_nonzero(self.X[0]))
        self.forget= np.zeros(np.count_nonzero(self.X[0]))
        
        
    def update_forget(self, classifier1, classifier2): # after finalize
        classifier1 = np.append(classifier1, np.zeros(len(classifier2)- len(classifier1)))
        self.forget = np.subtract(classifier2, classifier1)
        if (np.linalg.norm(self.forget)!=0):
            self.forget = np.divide(self.forget, np.linalg.norm(self.forget)) # when norm is 0 what happens?
    
    
    def extend_forget(self, classifier1, classifier2):
        self.forget = np.append(self.forget, np.zeros(len(classifier2)- len(classifier1)))
    
    
    def finalize_classifier(self, memoryless):
        self.weights = np.append(self.weights, np.zeros(len(memoryless) - len(self.weights)))
        self.weights = np.multiply(self.forget, self.weights) + np.multiply(np.subtract(1, self.forget), memoryless)
        


    def parameter_set(self, i, loss):
        if self.option==0: return loss/np.dot(self.X[i], self.X[i])
        if self.option==1: return np.minimum(self.C, loss/np.dot(self.X[i], self.X[i]))
        if self.option==2: return loss/((1/(2*self.C))+np.dot(self.X[i], self.X[i]))
        
        
    def sparsity_step(self):
        projected= np.multiply(np.minimum(1, self.Lambda/np.linalg.norm(self.weights, ord=1)), self.weights)
        self.weights= self.truncate(projected)
        
        
    def truncate(self, projected):
        if np.linalg.norm(projected, ord=0)> self.B*len(projected):
            remaining= int(np.maximum(1, np.floor(self.B*len(projected))))
            for i in projected.argsort()[:(len(projected)-remaining)]:
                projected[i]=0
            return projected
        else: return projected


    def fit(self):
        print("OLMF")
        random.seed(p.random_seed)
        for i in range(0, self.rounds):
            if(i%5==0):
                print("Round: "+str(i))
            self.formatInputData() # each round get shuffled data from same seed as other algorithm
            train_error=0
            train_error_vector=[]
            total_error_vector= np.zeros(len(self.y))
            
            self.set_classifier()
            
            for i in range(0, len(self.y)):
                #if(i%10)==0:
                print("Instance: "+str(i))
                row= self.X[i][:np.count_nonzero(self.X[i])]
                if len(row)==0: continue
                y_hat= np.sign(np.dot(self.weights, row[:len(self.weights)]))
                if y_hat!=self.y[i]: train_error+=1
                loss= (np.maximum(0, 1-self.y[i]*(np.dot(self.weights, row[:len(self.weights)]))))
                tao= self.parameter_set(i, loss)
                w_1= self.weights+np.multiply(tao*self.y[i], row[:len(self.weights)])
                w_2= np.multiply(tao*self.y[i], row[len(self.weights):])
                
                
                
#                # forget at time t is the difference of prev. classifier and current memoryless, slightly better after instance 200
#                # at time t, we use forget of t-1, which is diff bw. prev memoryless and before classifier
                prev_classifier = self.weights
                self.extend_forget(prev_classifier, np.append(w_1, w_2))
                self.finalize_classifier(np.append(w_1, w_2))
                self.update_forget(prev_classifier, np.append(w_1, w_2))
                
                
                
                

                self.sparsity_step()
                train_error_vector.append(train_error/(i+1))
            total_error_vector= np.add(train_error_vector, total_error_vector)
            self.error_stats.append(train_error)
        total_error_vector= np.divide(total_error_vector, self.rounds)
        print(self.weights)
        return train_error_vector
               
    
    def predict(self, X_test):
        prediction_results=np.zeros(len(X_test))
        for i in range (0, len(X_test)):
            row= X_test[i]
            prediction_results[i]= np.sign(np.dot(self.weights, row[:len(self.weights)]))
        return prediction_results


    def formatInputData(self):
        dataset = self.data
        all_keys = set().union(*(d.keys() for d in dataset))

        int_keys = []
        for key in all_keys:
            if key!='class_label':
                int_keys.append(int(key))
    
        X,y = [],[]
        for row in dataset:
            for key in all_keys:
                if key not in row.keys() : row[key]=0
            y.append(row['class_label'])
            del row['class_label']
            
            for i in range(0, max(int_keys)):
                if i not in all_keys:
                    row[i]=0
                
        if 0 not in row.keys(): start=1
        if 0 in row.keys(): start=0
        for row in dataset:
            X_row=[]
            for i in range(start, len(row)):
                X_row.append(row[i])
            X.append(X_row)
        print("done")
        self.X, self.y = X, y

        

