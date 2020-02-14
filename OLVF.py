import numpy as np
import preprocess
import random
import copy
import parameters as p


class olvf:
    def __init__(self, data, update, sparsity, removing_possiblity=0.5):
        self.sparsity_mode = sparsity
        self.C = p.olvf_C
        self.Lambda = p.olvf_Lambda
        self.B = p.olvf_B
        self.option = p.olvf_option
        self.data = data
        self.rounds = p.cross_validation_folds
        self.error_stats = []
        self.remove = removing_possiblity
    
    def updateMetadata(self, i):
        for index in range(0, len(self.X[i])):
            if self.X[i][index]!=0:
                if self.count_vector[index]==0:
                    self.count_vector[index]=i
                else:
                    self.count_vector[index]+=1
          
                     
    def set_metadata(self):
        self.count_vector=[1 for i in range(0, len(self.X[0]))]
         
        
    def set_classifier(self):
        self.weights= np.zeros(len(self.X[0]))
        
        
    def scale_loss(self, i):
        shared=0
        overall=0
        counter=0
        
        for index in range(0, len(self.weights)):
            if self.weights[index]!=0:
                overall+=self.count_vector[index]/(i+1) #3
            else:
                counter+=1
        if overall ==0:
             overall = 1
        
        for index in range(0, len(self.X[i])):
            if self.X[i][index]!=0:
                shared+=(self.count_vector[index])/(i+1) #2
        #print((shared / overall) * (shared/len(self.X[i])))
        
        return p.phi * (shared / overall) * (shared/(shared + (counter)))# projection confidence
        #return p.phi  * ((shared + (counter))) / overall

         
       
    def parameter_set(self, i, loss, phi):
        inner_product = np.dot(self.X[i], self.X[i])
        if inner_product == 0: inner_product = 1
        if phi == 0: phi = 1
        return np.minimum(self.C, loss/(inner_product*phi))


    def sparsity_step(self, i):
        if self.sparsity_mode == "no":
            return
        if self.sparsity_mode == "variable": #changed
            freq = np.multiply(self.count_vector, 1/(i+1))
            projected= np.multiply(np.minimum(1, np.linalg.norm(freq)), self.weights)
            self.weights= self.truncate(projected)
            
        if self.sparsity_mode == "normal":
             if np.linalg.norm(self.weights, ord=1) == 0:
                 return
             freq = np.multiply(self.count_vector, 1/(i+1))
             projected= np.multiply(np.minimum(1, self.Lambda/np.linalg.norm(self.weights, ord=1)), self.weights)
             self.weights= self.truncate(projected)


    def truncate(self, projected):
        if np.linalg.norm(projected, ord=0)> self.B*len(projected):
            remaining= int(np.maximum(1, np.floor(self.B*len(projected))))
            for i in projected.argsort()[:(len(projected)-remaining)]:
                projected[i]=0
                self.count_vector[i]=0
        return projected
    

    def fit(self):
        print("OLVF: phi="+str(p.phi)+" C="+str(p.olvf_C))
        random.seed(p.random_seed)
        
        for i in range(0, self.rounds):
            self.getShuffledData()
            print("Round: "+str(i))
            self.set_classifier()
            self.set_metadata()
            train_error=0
            train_error_vector=[]
            total_error_vector= np.zeros(len(self.y))

            for i in range(0, len(self.y)):
                row= self.X[i][:len(self.X[i])]
                #predict
                if len(row)==0:
                    print("0 length")
                    continue
                if np.linalg.norm(row)==0:
                    print("0 norm")
                    train_error_vector.append(train_error/(i+1))
                    print(row)
                    continue
                
                y_hat= np.sign(np.dot(self.weights, row[:len(self.weights)]))
                
                if y_hat!=self.y[i]: train_error+=1
                phi= self.scale_loss(i)
                loss= phi * (np.maximum(0, (1-self.y[i]*(np.dot(self.weights, row[:len(self.weights)])))))
                #tao= self.parameter_set(i, loss, phi)
                #self.weights= self.weights+np.multiply(phi*tao*self.y[i], row[:len(self.weights)])
                self.updateMetadata(i)
                self.weights= self.weights+np.multiply((loss*self.y[i]*(1/(np.linalg.norm(row)))), row[:len(self.weights)])
                
                self.sparsity_step(i)
                
                train_error_vector.append(train_error/(i+1))
            self.error_stats.append(train_error)
            total_error_vector= np.add(train_error_vector, total_error_vector)
        print("Classifier length:"+str(np.count_nonzero(self.weights)))   
        total_error_vector= np.divide(total_error_vector, self.rounds)
        return train_error_vector
        
        
    def predict(self, X_test):
        prediction_results=np.zeros(len(X_test))
        for i in range (0, len(X_test)):
            row= X_test[i]
            prediction_results[i]= np.sign(np.dot(self.weights, row[:len(self.weights)]))
        return prediction_results

    
    def getShuffledData(self, mode=p.stream_mode): # generate data for cross validation
        data=self.data
        copydata= copy.deepcopy(data)
        random.shuffle(copydata)
        if mode=='trapezoidal': dataset=preprocess.removeDataTrapezoidal(copydata)
        if mode=='variable': dataset=preprocess.removeRandomData(copydata, self.remove)
        else: dataset= copydata
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

        self.X, self.y = X, y              






    
    
