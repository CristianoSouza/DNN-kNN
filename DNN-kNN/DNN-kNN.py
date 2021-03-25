from sklearn import neighbors
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import pandas as pd
import numpy as np
import sklearn
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras.preprocessing.text
from sklearn.model_selection import train_test_split

class DNN:
  model_dnn = None
  imput_dim_neurons = 0
  number_neurons_hidden_layer = 10
  activation_function_hidden_layer = "tanh"
  number_neurons_output_layer = 2
  activation_function_output_layer = "softmax"
  n_epochs = 2
  optimizer = 'adam'
  loss = 'sparse_categorical_crossentropy'

  def setImputDimNeurons(self, a):
    self.imput_dim_neurons = a
  
  def setActivationFunctionHiddenLayer(self, a):
    self.activation_function_hidden_layer = a

  def setNumNeuronsHiddenLayer(self, a):
    self.number_neurons_hidden_layer = a
  
  def setActivationFunctionOutputLayer(self, a):
    self.activation_function_output_layer = a  

  def setNumNeuronsOutLayer(self, a):
    self.number_neurons_output_layer = a

  def setNumEpochs(self, a):
    self.n_epochs = a

  def setOptimizer(self, a):
    self.optimizer = a

  def setLoss(self, a):
    self.loss = a

  def getInfos(self):
    texto = '    DNN: \n'
    texto += f'      Structure:\n'
    texto += f'        Input dimension: {self.imput_dim_neurons}\n'
    texto += f'        Number of hidden layer neurons: {self.number_neurons_hidden_layer}\n'
    texto += f'        Activation function hidden layers: {self.activation_function_hidden_layer}\n'
    texto += f'        Number of output layer neurons: {self.number_neurons_output_layer}\n'
    texto += f'        Activation function output layers: {self.activation_function_output_layer}\n'
    texto += f'      Training:\n'
    texto += f'        Epochs: {self.n_epochs}\n'
    texto += f'        Optimizer: {self.optimizer}\n'
    texto += f'        Loss: {self.loss}\n'    
    return texto  

  def generateHybridModel(self, data_set_samples, data_set_labels):

    self.model_dnn = Sequential()
    self.model_dnn.add(Dense(self.number_neurons_hidden_layer, input_dim=self.imput_dim_neurons, activation=self.activation_function_hidden_layer))
    self.model_dnn.add(Dense(self.number_neurons_hidden_layer, activation=self.activation_function_hidden_layer))
    self.model_dnn.add(Dense(self.number_neurons_output_layer, activation=self.activation_function_output_layer))

    print(self.model_dnn.summary())
    self.model_dnn.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
    csv_logger = CSVLogger('training.log')

    early_stopping = EarlyStopping(monitor='loss', patience=20)

    fit = self.model_dnn.fit(data_set_samples, data_set_labels, epochs=self.n_epochs, verbose=2, callbacks=[early_stopping])

    return 0

  def predict(self, data_set_samples):

    predictions_dnn = self.model_dnn.predict(data_set_samples)

    return predictions_dnn


class kNN:
  knn = 0
  k_neighbors = 1
  algorithm = 'kd_tree'
  weights = 'uniform'

  def setKNeighbors(self, a):
    self.k_neighbors = a

  def setAlgorithm(self, a):
    self.algorithm = a 

  def setWeights(self, a):
    self.weights = a 

  def getInfos(self):
    texto = '    kNN: \n'
    texto += f'      K Neighbors: {self.k_neighbors}\n'
    texto += f'      Algorithm: {self.algorithm}\n'
    texto += f'     Weights: {self.weights}\n'
    return texto  

  def buildExamplesBase(self, data_set_samples, data_set_labels):
    self.knn = neighbors.KNeighborsClassifier(self.k_neighbors, weights=self.weights, algorithm=self.algorithm)
    self.knn.fit(data_set_samples, data_set_labels)
    return 0

  def predict(self, data_set_samples):

    predictions_knn = self.knn.predict(data_set_samples)
    return predictions_knn





class DNNkNN:
  dnn = None
  knn = None
  ACCEPTABLE_ERROR_RATE_FP = 0
  ACCEPTABLE_ERROR_RATE_FN = 0
  normal_neuron_limit = 0
  attack_neuron_limit = 0

  def __init__(self):
    self.dnn = DNN()
    self.knn = kNN()
  
  
  def setAcceptableErrorRateFP (self, a):
    self.ACCEPTABLE_ERROR_RATE_FP = a

  def setAcceptableErrorRateFN (self, a):
    self.ACCEPTABLE_ERROR_RATE_FN = a



  def getDNN(self):
    return self.dnn

  def getKNN(self):
    return self.knn

  def getInfos(self):
    texto = '    DNN-kNN Method: \n'
    texto += dnnknn.getDNN().getInfos()
    texto += dnnknn.getKNN().getInfos()
    return texto  



  def training(self, data_set_samples, data_set_labels):
    predictions_dnn_knn_training = []
    knn_instances = []

    #Dnn training
    self.dnn.generateHybridModel(data_set_samples, data_set_labels)

    #Knn training only for the training phase of the approach (same number of DNN attributes)
    self.knn.buildExamplesBase(data_set_samples, data_set_labels)

    print('Starting definition of threshold values...')
    acc = 0


    #sets the limits very low, so no example is sent to KNN
    self.attack_neuron_limit = 0.5  
    self.normal_neuron_limit = 0.5 

    for i in range(1, 10):
      predictions_dnn_knn_training = []
      list_id_sendto_knn = []
      knn_instances = []
      self.knn_count = 0
      self.dnn_count = 0

      #print(f'New Limit Neuron Normal: {self.normal_neuron_limit}')
      #print(f'New Limit Neuron Attack: {self.attack_neuron_limit}')

      predictions_dnn = self.dnn.predict(data_set_samples)

      for j in range(0,len(data_set_samples)):
        if(predictions_dnn[j][0] > self.normal_neuron_limit):
          predictions_dnn_knn_training.append(0) 
        elif(predictions_dnn[j][1] > self.attack_neuron_limit):
          predictions_dnn_knn_training.append(1) 
        else:
          predictions_dnn_knn_training.append(-1)
          list_id_sendto_knn.append(j)
          knn_instances.append(data_set_samples[j])
      
      if (len(knn_instances) != 0):
        predictions_knn = self.knn.predict(knn_instances)

        for k in range(0, len(knn_instances)):
          predictions_dnn_knn_training[list_id_sendto_knn[k]] = predictions_knn[k]

      acc = sklearn.metrics.accuracy_score(data_set_labels, predictions_dnn_knn_training)
      matriz = confusion_matrix(data_set_labels, predictions_dnn_knn_training)

      tn = matriz[0][0]
      fp = matriz[0][1]
      fn = matriz[1][0]
      tp = matriz[1][1]
      print(f'Training DNN-kNN acc: {acc}')
      #print(f'FP Rate: {fp*(100/(fp+tp))}')
      #print(f'FN Rate: {fn*(100/(fn+tn))}')

      output_neuron_normal = []
      output_neuron_attack = []
      for p in range(0,len(data_set_samples)):
        if (predictions_dnn[p][0] > 0.5):
          output_neuron_normal.append(predictions_dnn[p][0])
        if (predictions_dnn[p][1] > 0.5):
          output_neuron_attack.append(predictions_dnn[p][1])

      if(len(output_neuron_attack) != 0):
        if ((fp*(100/(fp+tp)) > self.ACCEPTABLE_ERROR_RATE_FP) and (fp != 0)):
          self.attack_neuron_limit = np.percentile(output_neuron_attack, i*10)  
      
      if(len(output_neuron_normal) != 0):
        if ((fn*(100/(fn+tn)) > self.ACCEPTABLE_ERROR_RATE_FN) and (fn != 0)):
          self.normal_neuron_limit = np.percentile(output_neuron_normal, i*10)  

      if((fn*(100/(fn+tn)) <= self.ACCEPTABLE_ERROR_RATE_FP) and (fn*(100/(fn+tn)) <= self.ACCEPTABLE_ERROR_RATE_FN)):
        break


    print(f'Final value Normal neuron limit: {self.normal_neuron_limit}')
    print(f'Final value Attack neuron limit: {self.attack_neuron_limit}')



    #Attribute reduction (selection made with InfoGain)
    data_set_samples = np.delete(data_set_samples, np.s_[0,1,6,8,9,10,12,13,14,15,16,17,18,19,20,21,23,26,27,35,39], axis=1)

    #Final knn training with reduced attributes
    self.knn.buildExamplesBase(data_set_samples, data_set_labels)


    return 0


  def predict(self, test_data_set_samples):

    predictions_ann_knn = []
    list_id_sendto_knn = []
    knn_instances = []
    self.dnn_count = 0
    self.knn_count = 0

    predictions_dnn = self.dnn.predict(test_data_set_samples)

    #Attribute reduction (selection made with InfoGain)
    test_data_set_samples = np.delete(test_data_set_samples, np.s_[0,1,6,8,9,10,12,13,14,15,16,17,18,19,20,21,23,26,27,35,39], axis=1)


    for i in range(0,len(test_data_set_samples)):
          
      if(predictions_dnn[i][0] > self.normal_neuron_limit):
        self.dnn_count = self.dnn_count + 1
        predictions_ann_knn.append(0) 
      elif(predictions_dnn[i][1] > self.attack_neuron_limit):
          self.dnn_count = self.dnn_count + 1
          predictions_ann_knn.append(1) 
      else:
          self.knn_count = self.knn_count + 1
          predictions_ann_knn.append(-1)
          list_id_sendto_knn.append(i)
          knn_instances.append(test_data_set_samples[i])
      
    if (len(knn_instances) != 0):
      predictions_knn = self.knn.predict(knn_instances)
      for i in range(0, len(knn_instances)):
        predictions_ann_knn[list_id_sendto_knn[i]] = predictions_knn[i]
  

    return predictions_ann_knn




############################### EXAMPLE ################################



train_url = 'nsl_kdd_multilclass_5.csv'

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","Attack"]


data_set = pd.read_csv(train_url,header=1, names = col_names)


print('Dimensions of the Training set:',data_set.shape)

columns_category = ['protocol_type','service','flag']

le = LabelEncoder()
for col in columns_category:
  le.fit(data_set[col])
  data_set[col] = le.fit_transform(data_set[col])

print("mostrar forma depois da categorização:")
data_set.head()


condiction = [data_set['Attack'] == 'DoS',
            data_set['Attack'] == 'probe',
            data_set['Attack'] == 'R2L',
            data_set['Attack'] == 'U2R',            
            data_set['Attack'] == 'normal',
            ]

results = [1, 1, 1, 1, 0 ]            

data_set['Attack'] = np.select(condiction, results)
data_set.head()


print('Label distribution Training set:')
print(data_set['Attack'].value_counts())
print()


back_up = data_set[:]
data_set_labels = back_up.pop('Attack')
data_set_samples = back_up



print(data_set_samples.shape)
print(data_set_labels.shape)

data_set_samples = data_set_samples.values
data_set_labels = data_set_labels.values

data_set_training_samples, data_set_test_samples, data_set_training_labels, data_set_test_labels = train_test_split(data_set_samples, data_set_labels, test_size=0.30, random_state=42, stratify=data_set_labels)








ACCEPTABLE_ERROR_RATE_FP = 0.05
ACCEPTABLE_ERROR_RATE_FN = 0.05

dnnknn = DNNkNN()
dnnknn.getDNN().setImputDimNeurons(41)
dnnknn.getDNN().setActivationFunctionHiddenLayer("tanh")
dnnknn.getDNN().setNumNeuronsHiddenLayer(41)
dnnknn.getDNN().setActivationFunctionOutputLayer("softmax")
dnnknn.getDNN().setNumNeuronsOutLayer(41)
dnnknn.getDNN().setNumEpochs(100)
dnnknn.getDNN().setOptimizer('adam')
dnnknn.getDNN().setLoss('sparse_categorical_crossentropy')

dnnknn.getKNN().setKNeighbors(1)
dnnknn.getKNN().setAlgorithm('kd_tree')
dnnknn.getKNN().setWeights('uniform')

dnnknn.setAcceptableErrorRateFP(0.05)
dnnknn.setAcceptableErrorRateFN(0.05)


dnnknn.training(data_set_training_samples, data_set_training_labels)

predictions_dnn_knn = dnnknn.predict(data_set_test_samples)

acc = sklearn.metrics.accuracy_score(data_set_test_labels, predictions_dnn_knn)
print(f'Testing DNN-kNN acc: {acc}')