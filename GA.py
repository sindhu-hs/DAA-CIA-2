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
        
class GA:
    def __init__(self, pop_size, x, y):
        self.pop_size = pop_size
        self.x = x
        self.y = y
        
    def generatePopulation(self):
        self.population = np.random.randn(self.pop_size, 79)
        
    def nextGeneration(self):
        children = []
        score_children = []
        score_population = []
        
        for i in range(len(self.population)):
            score_population.append(self.evaluateScore(self.population[i]))
            for j in range(len(self.population)):
                if i != j:
                    children += (self.crossOver(self.population[i], self.population[j]))    
                    
        for child in children:
            score_children.append(self.evaluateScore(child))
        children = np.array(children)
        
        if (score_population[0] > score_children[0]) and (score_population[1] > score_children[1]):
            
            score_population = np.argsort(score_population)
            score_population = score_population[::-1][:2]
            
            score_children = np.argsort(score_children)
            score_children = score_children[::-1][:self.pop_size-2]
        
            self.population = np.concatenate((self.population[score_population], children[score_children]))
            return self.population, self.evaluateScore(self.population[0])
            
        elif score_population[0] > score_children[0]:
            
            score_population = np.argsort(score_population)
            score_population = score_population[::-1]
            
            score_children = np.argsort(score_children)
            score_children = score_children[::-1][:self.pop_size-1]
            
            self.population = np.concatenate((self.population[score_population[0]].reshape(1,79), children[score_children]))
            return self.population, self.evaluateScore(self.population[0])
            
        elif score_population[1] > score_children[1]:
            
            score_population = np.argsort(score_population)[::-1]
            score_children = np.argsort(score_children)[::-1]
            
            temp_population = np.concatenate((children[score_children[0]], self.population[score_population[1]]))
            score_children = score_children[1 : self.pop_size-2]
            temp_population = temp_population.reshape((2,79))
            
            self.population = np.concatenate((temp_population, children[score_children]))
            return self.population, self.evaluateScore(self.population[0])
        
        else:
            
            score_children = np.argsort(score_children)
            score_children = score_children[::-1][:self.pop_size]
        
            self.population = children[score_children]
            return self.population, self.evaluateScore(self.population[0])
         
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
        
ga = GA(20, x, y)
ga.generatePopulation()
for epoch in range(100):
     pop, max_score = ga.nextGeneration()
     print("Epoch: ",epoch+1," ---> accuracy: ", max_score)           