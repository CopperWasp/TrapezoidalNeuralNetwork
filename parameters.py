phi = 0.1 # variable 5
cross_validation_folds = 20
random_seed = 40 #variable 100
olvf_C = 0.1 #variable 10
olvf_Lambda = 30
olvf_B = 0.64
olvf_option = 1 # 0, 1 or 2
stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
olsf_C = 0.01
olsf_Lambda = 30
olsf_B = 0.64
olsf_option = 2 # 0, 1 or 2

def saveParameters(figurename):
    text_file = open(figurename[:-3]+".txt", "w")
    param_setting = "SHARED - folds = "+str(cross_validation_folds) + "\n" + \
                    "SHARED - seed = "+str(random_seed) + "\n\n" + \
                    "OLVF - phi = "+str(phi) + "\n" + \
                    "OLVF - C = "+str(olvf_C) + "\n" + \
                    "OLVF - B = "+str(olvf_B) + "\n" + \
                    "OLVF - Lambda = "+str(olvf_Lambda) + "\n" + \
                    "OLVF - Option = "+str(olvf_option) + "\n" + \
                    "OLVF - StreamMode = "+stream_mode + "\n\n" + \
                    "OLSF - C = "+str(olsf_C) + "\n" + \
                    "OLSF - B = "+str(olsf_B) + "\n" + \
                    "OLSF - Lambda = "+str(olsf_Lambda) + "\n" + \
                    "OLSF - Option = "+str(olsf_option) + "\n"
    text_file.write(param_setting)
    text_file.close()


#def setParametersMagic():
    #phi = 0.06 # variable same
    #cross_validation_folds = 25 #variable 20
    #random_seed = 100
    #olvf_C = 0.75 #varible random
    #olvf_Lambda = 30
    #olvf_B = 0.64
    #olvf_option = 1 # 0, 1 or 2
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #olsf_C = 0.1
    #olsf_Lambda = 30
    #olsf_B = 0.64
    #olsf_option = 1 # 0, 1 or 2

    
#def setParametersA8A():
#    phi = 0.01
#    cross_validation_folds = 2 #5
#    random_seed = 20
#    olvf_C = 0.1 
#    olvf_Lambda = 50
#    olvf_B = 1
#    olvf_option = 1 # 0, 1 or 2
#    stream_mode = "variable" # or variable, decrease sparsity when variable
#    olsf_C = 0.01
#    olsf_Lambda = 30
#    olsf_B = 0.04
#    olsf_option = 2 # 0, 1 or 2
     
#def setParametersWBC():
    #phi = 0.1 # variable 5
    #cross_validation_folds = 20
    #random_seed = 40 #variable 100
    #olvf_C = 0.1 #variable 10
    #olvf_Lambda = 30
    #olvf_B = 0.64
    #olvf_option = 1 # 0, 1 or 2
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #olsf_C = 0.01
    #olsf_Lambda = 30
    #olsf_B = 0.64
    #olsf_option = 2 # 0, 1 or 2
    
    
#def setParametersWDBC():
#    phi = 0.005 #variable 0.05
#    cross_validation_folds = 20
#    random_seed = 51 #variable 100
#    olvf_C = 1
#    olvf_Lambda = 30
#    olvf_B = 1
#    olvf_option = 1 # 0, 1 or 2
#    stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
#    olsf_C = 1
#    olsf_Lambda = 30
#    olsf_B = 1
#    olsf_option = 1 # 0, 1 or 2   
    
    
#def setParametersSpambase():
    #phi = 0.1 #variable 0.1
    #cross_validation_folds = 20
    #random_seed = 100 #variable 150
    #olvf_C = 1 #variable 0.01
    #olvf_Lambda = 30
    #olvf_B = 0.64
    #olvf_option = 1 # 0, 1 or 2
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #olsf_C = 1
    #olsf_Lambda = 30
    #olsf_B = 0.64
    #olsf_option = 1 # 0, 1 or 2
    
    
#def setParametersIonosphere():
    #phi = 0.2 #variable 0.05
    #cross_validation_folds = 25
    #random_seed = 20 #variable 100
    #olvf_C = 1 #variable 0.1
    #olvf_Lambda = 30
    #olvf_B = 0.64
    #olvf_option = 1 # 0, 1 or 2
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #olsf_C = 1
    #olsf_Lambda = 30
    #olsf_B = 0.64
    #olsf_option = 2 # 0, 1 or 2
    
#def setParametersGerman():
#    phi = 0.05
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
 
   
#def setParametersWPBC():
#    phi = 0.05
#    cross_validation_folds = 20
#    random_seed = 50 #41 variable
#    olvf_C = 2.5 #o.1 variable
#    olvf_Lambda = 30
#    olvf_B = 0.08 #0.1 variable
#    olvf_option = 1 # 0, 1 or 2
#    stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
#    olsf_C = 0.1
#    olsf_Lambda = 30
#    olsf_B = 0.08
#    olsf_option = 2 # 0, 1 or 2
    
#def setParametersSVMGuide():
#    phi = 0.01 #0.1 var
#    cross_validation_folds = 20 #20 var
#    random_seed = 50 #20 var
#    olvf_C = 0.1 #0.1 var
#    olvf_Lambda = 30 #50 var
#    olvf_B = 0.1
#    olvf_option = 1 # 0, 1 or 2
#    stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
#    olsf_C = 0.1
#    olsf_Lambda = 30
#    olsf_B = 0.08
#    olsf_option = 2 # 0, 1 or 2 #sqrt of inconfi


     

