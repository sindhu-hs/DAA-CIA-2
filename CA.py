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
            
nn = NN([11,6,1])

class Belief:
    def __init__(self, mean, std):
        self.mean = mean #array of size 79
        self.std  = std  #array of size 79
        
    def generatePopulation(self, populationSizePerCulture):
        self.populationSizePerCulture = populationSizePerCulture
        self.population = self.mean + self.std * np.random.randn(populationSizePerCulture, 79)
        
    def updateCulture(self):
        self.mean = np.mean(self.population, axis = 0)
        self.std  = np.std(self.population, axis = 0)
        
class CulturalAlgo:
    def __init__(self, cultureParams, x, y):
        self.x = x
        self.y = y
        self.cultures = []
        for params in cultureParams:
            self.cultures += [Belief(params[0], params[1])]
        
    def generatePopulation(self, populationSizePerCulture):
        self.populationSizePerCulture = populationSizePerCulture
        self.populationSize = len(self.cultures) * populationSizePerCulture
        self.population = np.zeros((self.populationSizePerCulture, 79))
        for culture_idx in range(len(self.cultures)):
            self.cultures[culture_idx].generatePopulation(populationSizePerCulture)
            if culture_idx == 0:
                self.population += self.cultures[culture_idx].population
            else:
                self.population = np.concatenate((self.population, self.cultures[culture_idx].population))
      
    def crossOver(self, parent1, parent2):
        idx = np.random.randint(0,len(parent1)-1)
        child1 = np.concatenate((parent1[:idx],parent2[idx:]), axis = 0)
        child2 = np.concatenate((parent2[:idx],parent1[idx:]), axis = 0)
        child1 = self.mutate(child1)
        child2 = self.mutate(child2)
        return [child1, child2] 
        
    def mutate(self, child):
        idx = np.random.randint(0,len(child)-1)
        child[idx] = np.random.randint(100) * np.random.rand()
        return child
       
    def evaluateScore(self, child):
        nn.vectorToWeights(child)
        ypred = nn.forward(self.x)
        return accuracy_score(self.y, ypred)
       
    def nextGeneration(self):
        children = []
        scores = []
        for ind1 in range(len(self.population)):
            for ind2 in range(len(self.population)):
                if ind1 != ind2:
                    children += (self.crossOver(self.population[ind1], self.population[ind2]))
                    
        for child in children:
            scores.append(self.evaluateScore(child))
        children = np.array(children)
        
        scores = np.argsort(scores)
        scores = scores[::-1][:self.populationSizePerCulture]
        self.population = children[scores]
        return self.population, self.evaluateScore(self.population[0])
    
    def getZ(self, culture, individual):
        return (individual - culture.mean)/culture.std
        
    def classify(self):
        pop = np.zeros((self.populationSizePerCulture ,79))
        for culture_idx in range(len(self.cultures)):
            z_values = []
            for individual in self.population:
                z = sum(self.getZ(self.cultures[culture_idx], individual))
            z_values.append(z)
            z_values = np.array(z_values)
            z_values = np.argsort(z_values)[:self.cultures[culture_idx].populationSizePerCulture]
            self.cultures[culture_idx].population = self.population[z_values]
            if culture_idx == 0:
                pop += self.cultures[culture_idx].population
            else:
                pop = np.concatenate((pop, self.cultures[culture_idx].population))
        self.population = pop
        
    def updateCulture(self):
        for culture in self.cultures:
            culture.updateCulture()

ca = CulturalAlgo([(5,2), (0,3), (6,2)], x, y)
ca.generatePopulation(10)
for epoch in range(3):
    pop, max_score = ca.nextGeneration()
    ca.classify()
    ca.updateCulture()
    print(pop.shape)
    print("=====================")
    #print("Epoch: ",epoch+1," ---> accuracy: ", max_score)   