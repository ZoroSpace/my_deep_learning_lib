import numpy
import scipy.special
import matplotlib.pyplot as plt
class neural_network:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        self.wih = numpy.random.rand(hidden_nodes,input_nodes) - 0.5
#         self.wih = numpy.random.normal(0.0,pow(input_nodes,-0.5),(hidden_nodes,input_nodes))
        self.who = numpy.random.rand(output_nodes,hidden_nodes) - 0.5
#         self.who = numpy.random.normal(0.0,pow(hidden_nodes,-0.5),(output_nodes,hidden_nodes))
        self.activation_function = lambda x : scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin = 2).T
        targets = numpy.array(targets_list,ndmin = 2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who+=self.lr*numpy.dot(output_errors*final_outputs*(1-final_outputs),hidden_outputs.T)
        self.wih+=self.lr*numpy.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),inputs.T)        

    def query(self,inputs):
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs
    
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
nn = neural_network(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file=open('mnist_train.csv','r')
training_list=training_data_file.readlines()
training_data_file.close()

for record in training_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]))/255*0.99+0.01
    targets = numpy.zeros(10)+0.01
    targets[int(all_values[0])] = 0.99
    nn.train(inputs,targets)

test_data_file=open('mnist_test.csv','r')
test_list=test_data_file.readlines()
test_data_file.close()

score_card = []
for record in test_list:
    test_all_values = record.split(',')
    test_input = (numpy.asfarray(test_all_values[1:]))/255*0.99+0.01
    test_output = nn.query(test_input)
    label = numpy.argmax(test_output)
#     print(label,' ',test_all_values[0])
    if label == int(test_all_values[0]):
        score_card.append(1)
    else:
        score_card.append(0)
print(sum(score_card)/len(score_card))


# run the network backwards, given a label, see what image it produces
# label to test
label = 0
# create the output signals for this label
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] is the target label for this record
targets[label] = 0.99
print(targets)
# get image data
image_data = nn.backquery(targets)
# plot image data
plt.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
