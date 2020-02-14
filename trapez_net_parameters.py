# parameters
seed = 123
criterion = "KL"
optimizer = "ADAM"
folds = 20
adaptive_lr_switch = False
edge_pruning_switch = True
lamb = 30
second_layer_penalty = 0.1
first_layer_penalty = 0.1
layer_sizes = [9,9,2]
learning_rate = 0.75
l2_penalty = 0.2
lr_decay = 0.97
pruning_coefficient = 1

######################### equals

# ionosphere
#second_layer_penalty = 0.1
#first_layer_penalty = 0.1
#layer_sizes = [34,34,2]
#learning_rate = 0.75
#l2_penalty = 0
#lr_decay = 0.96

# wbc
#second_layer_penalty = 0.1
#first_layer_penalty = 0.1
#layer_sizes = [9,9,2]
#learning_rate = 0.8
#l2_penalty = 0.25
#lr_decay = 0.97

# wdbc
#second_layer_penalty = 0.1
#first_layer_penalty = 0.1
#layer_sizes = [29,29,2]
#learning_rate = 0.2
#l2_penalty = 0
#lr_decay = 0.99

# magic04
#second_layer_penalty = 0.1
#first_layer_penalty = 0.1
#layer_sizes = [10,10,2]
#learning_rate = 0.5
#l2_penalty = 0
#lr_decay = 0.99

# spambase
#layer_sizes = [57,7,2]
#learning_rate = 0.1
#l2_penalty = 0
#lr_decay = 0.99

######################### we are better

# german
#seed = 123
#criterion = "KL"
#optimizer = "ADAM"
#folds = 10
#adaptive_lr_switch = True
#edge_pruning_switch = True
#lamb = 30
#second_layer_penalty = 0
#first_layer_penalty = 0
#layer_sizes = [24,24,2]
#learning_rate = 0.1
#lr_decay = 0.99
#l2_penalty = 0.1

# wpbc
#layer_sizes = [32,32,2]
#learning_rate = 0.4
#l2_penalty = 0
#lr_decay = 0.97

# svmguide
#second_layer_penalty = 0.01
#first_layer_penalty = 0.01
#layer_sizes = [22,22,2]
#learning_rate = 1
#l2_penalty = 0
#lr_decay = 0.99

# a8a
#second_layer_penalty = 0.1
#first_layer_penalty = 0.1
#layer_sizes = [123,123,2]
#learning_rate = 0.1
#l2_penalty = 0
#lr_decay = 0.99