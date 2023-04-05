import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
np.random.seed(2)

data  = pd.read_csv("df.csv")
x = data.drop(["id","Personal Loan"], axis = 1).values
y = data["Personal Loan"].values.reshape(-1,1)

def activation(inputs):
    inputs = np.array(inputs)
    return 1 / (1 + np.exp(-inputs))

class Layer:
    def __init__(self, n_features, n_neurons):
        self.weights = np.random.rand(n_features+1, n_neurons)
        
    def forward(self, inputs):
        dummy  = np.ones(shape = (inputs.shape[0],1))
        inputs = np.concatenate((inputs,dummy), axis = 1)
        return np.dot(inputs, self.weights)

class NN:
    def __init__(self, network):
        self.layer = []
        self.layer_size = []
        for i in range(len(network)-1):
            self.layer.append(Layer(network[i], network[i+1])) #stores weights
            self.layer_size.append((network[i]+1, network[i+1]))

    def forward(self, x):
        output = x
        for i in range(len(self.layer)):
            output = self.layer[i].forward(output)
            output = activation(output)
        return output.round()
    
    def vectorToWeights(self, particle):
        curr_idx = 0
        for i in range(len(self.layer_size)):
            layer = self.layer[i]
            shape = self.layer_size[i]
            next_idx = (shape[0] * shape[1]) + curr_idx    
            layer.weights = np.reshape(particle[curr_idx:next_idx], shape)
            curr_idx = next_idx

class PSO:
    def __init__(self, x, y, w, c1, c2):
        self.x = x
        self.y = y
        self.w  = w
        self.c1 = c1
        self.c2 = c2
            
    def generateSwarm(self):
        self.x_swarm = np.random.rand(100, 79)
        self.ic = np.random.rand(100, 79)
        self.pb = self.x_swarm.copy()
        self.gb = self.pb[0]
           
    def evaluateScore(self, particle):
        nn.vectorToWeights(particle)
        ypred = nn.forward(self.x)
        return accuracy_score(self.y, ypred)
    
    def updatePb(self, particle_idx):
        if self.evaluateScore(self.x_swarm[particle_idx]) > self.evaluateScore(self.pb[particle_idx]):
            self.pb[particle_idx] = self.x_swarm[particle_idx]
            
    def updateGb(self):
        for i in range(len(self.pb)):
            if self.evaluateScore(self.pb[i]) > self.evaluateScore(self.gb):
                self.gb = self.pb[i]
                
    def evaluateNet(self):
        for i in range(self.pb.shape[0]):
            vpb = self.pb[i] - self.x_swarm[i]
            vgb = self.gb - self.x_swarm[i]
        r1 = np.random.random()
        r2 = np.random.random()
        dx = (self.w * self.ic) + (self.c1 * r1 * vpb) + (self.c2 * r2 * vgb)  
        self.w -= 0.01
        self.ic = dx 
        self.x_swarm += dx
 
nn = NN([11, 6, 1])
pso = PSO(x, y, 0.1, 0.5, 1.5)
pso.generateSwarm()
for epochs in range(10):
    for i in range(len(pso.x_swarm)):
        pso.updatePb(i)
    pso.updateGb()
    pso.evaluateNet()
    print("Epoch: ", epochs+1," ---> accuracy: ", pso.evaluateScore(pso.gb))


        
        






















