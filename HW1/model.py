# Yekta Can Tursun


from multiprocessing import reduction
from turtle import shape
import numpy as np
import torch

class ModelBase():
    def __init__(self) -> None:
        self.beta : np.array = np.zeros(shape=(2))
        pass

    def grad(self, y : np.array, y_pred : np.array, x : np.array):
        """
        This function calculates grad vector and return it
        """
        g_b0 = -2 * (y - y_pred).mean()
        g_b1 = -2 * (x * (y - y_pred)).mean()
        return g_b0, g_b1
    
    def fit(self, x, y,alpha=0.001):

        self.beta = np.random.random(2)
        np.random.seed(10)
        print("starting sgd")
        for i in range(50):
            y_pred: np.ndarray = self.beta[0] + self.beta[1] * x


            # Calculate the grad vector
            g_b0,g_b1 = self.grad(y,y_pred,x)

            print(f"({i}) beta: {self.beta}, gradient: {g_b0} {g_b1}")

            beta_prev = np.copy(self.beta)

            self.beta[0] = self.beta[0] - alpha * g_b0
            self.beta[1] = self.beta[1] - alpha * g_b1

            if np.linalg.norm(self.beta - beta_prev) < alpha/10:
                print(f"I do early stoping at iteration {i}")
                break

    def pred(self, y) -> np.array:
        # We thinnk y is a vector in 1D size
        # We will add one values for each y 
        x = np.ones(shape=(2,y.shape[0]))
        x[1,:] = y

        y_pred = self.beta @ x

        return y_pred


# Derivations


class ModelRegulizedL2(ModelBase):
       
    def grad(self, y: np.array, y_pred: np.array, x: np.array, lam=10**-1):
        g_b0 = -2 * (y - y_pred).mean() + 2 * lam * self.beta[0]
        g_b1 = -2 * (x * (y - y_pred)).mean() + 2 * lam * self.beta[1]
        return g_b0, g_b1


class ModelRegulizedL1(ModelBase):
    def grad(self, y: np.array, y_pred: np.array, x: np.array, lam=10**-1):
        if self.beta[0] >= 0:
            g_b0 = -2 * (y - y_pred).mean() + lam
        else:
            g_b0 = -2 * (y - y_pred).mean() - lam

        if self.beta[1] >= 0:
            g_b1 = -2 * (x * (y - y_pred)).mean() + lam
        else:
            g_b1 = -2 * (x * (y - y_pred)).mean() - lam
        
        return g_b0, g_b1

def sigmoid(x:np.array) -> np.array:
    z = 1/(1 + np.exp(-x))
    return z

class ThresholdBasedModel(ModelBase):
    def __init__(self, threshold, lam) -> None:
        super().__init__()
        self.lam = lam
        self.threshold = threshold
    
    def gradOLD(self, y: np.array, y_pred: np.array, x: np.array):
        g_b0 = -2 * (y - y_pred).mean() # dx
        g_b1 = -2 * (x*(y  - y_pred)).mean() # dy

        fx = self.threshold * np.mean(y_pred-y)

        loss0 =self.threshold * (np.exp(fx)*(g_b0))/(np.exp(fx)+1)**2
        loss1 =self.threshold *  (np.exp(fx)*(g_b1))/(np.exp(fx)+1)**2
        return loss0, loss1

    
    def gradOLD2(self, y: np.array, y_pred: np.array, x: np.array):
        pass
        diff = (y_pred-y)**2

        g = np.zeros(shape=(2,diff.shape[0]))
        for i,row in enumerate(diff):
            if row > self.threshold:
                g_b0 = -2 * (self.threshold) # dx
                g_b1 = -2 * (x[i] *(self.threshold)) # dy
            else:
                g_b0 = -2 * (row) # dx
                g_b1 = -2 * (x[i] *(row)) # dy

            g[:,i] = np.array([g_b0,g_b1])
            pass
        return g[0,:].mean(), g[1,:].mean()



# Torch based Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b0 = torch.nn.Parameter(torch.randn(()))
        self.b1 = torch.nn.Parameter(torch.randn(()))
    
    def forward(self, x):
        return self.b0 + self.b1 * x


class TorchBasedModel():
    def __init__(self,threshold) -> None:
        self.model = Model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.threshold = threshold
        pass
    
    def softCap(self, x : np.array, threshold=1,b=0.05):
        return threshold * (1/-torch.exp(b*x) + 1)
        
    def loss(self, y, y_pred):
        loss = (y_pred-y)**2
        loss = self.softCap(loss,threshold=self.threshold)
        return torch.mean(loss)

    def fit(self, x, y):
        pass
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        self.model.train()
        with torch.autograd.set_detect_anomaly(True):
            for i in range(100):
                y_pred = self.model(x)

                loss = self.loss(y,y_pred)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def pred(self, y) -> np.array:
        y = torch.from_numpy(y)
        self.model.eval()
        y_pred = self.model(y)
        return y_pred.detach().numpy()