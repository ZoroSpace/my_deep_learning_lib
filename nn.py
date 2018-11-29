import numpy
import scipy.special

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

    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin = 2).T
        targets = numpy.array(targets_list,ndmin = 2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(who.T,output_errors)
        
        self.who+=self.lr*numpy.dot(output_errors*final_outputs*(1-final_outputs),hidden_outputs.T)
        self.wih+=self.lr*numpy.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),inputs.T)        

    def query(self,inputs):
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        
