# Yekta Can Tursun


from multiprocessing import reduction
from turtle import shape
from cv2 import moments
import numpy as np
import torch

# Base class
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
    
    def fit(self, x: np.array, y: np.array,alpha=0.002,epoch=100):

        self.beta = np.random.random(2)
        np.random.seed(10)
        print("starting sgd")
        for i in range(epoch):
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

    def pred(self, y: np.array) -> np.array:
        """
        Calculate results with trained model
        """
        # We thinnk y is a vector in 1D size
        # We will add one values for each y 
        x = np.ones(shape=(2,y.shape[0]))
        x[1,:] = y

        y_pred = self.beta @ x

        return y_pred


# Derivations
# In this part, I created different version of model with different loss functions
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


class ThresholdBasedModel(ModelBase):
    def __init__(self, threshold) -> None:
        super().__init__()
        self.threshold = threshold
    

    def grad(self, y: np.array, y_pred: np.array, x: np.array, b=0.2):
        g_b0 = 2 * b * self.threshold *(y-y_pred) * np.exp(b *(y-y_pred)**2)
        g_b1 = -2 * b * self.threshold* (y-y_pred) * np.exp(b *(y-y_pred)**2)
        return g_b0.mean(), g_b1.mean()


# Torch based Model
# In order to compare with my threshold model, I implemented torch version so that I can compare them 

# Linear model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b0 = torch.nn.Parameter(torch.randn(()))
        self.b1 = torch.nn.Parameter(torch.randn(()))
    
    def forward(self, x):
        return self.b0 + self.b1 * x

# Torch Based Model 
class TorchBasedModel():
    def __init__(self,threshold) -> None:
        torch.manual_seed(10)
        self.model = Model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02)
        self.threshold = threshold
        pass
    
    def softCap(self, x : np.array, threshold=1,b=0.05):
        """
        Soft capping function
        """
        return threshold * (1/-torch.exp(b*x) + 1)
        
    def loss(self, y: np.array, y_pred: np.array):
        loss = (y_pred-y)**2
        loss = self.softCap(loss,threshold=self.threshold)
        return torch.mean(loss)

    def fit(self, x: np.array, y: np.array, epoch=100):
        
        # Tranfer from numpy to torch
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        # Set model for train
        self.model.train()

        # Autograd decent
        with torch.autograd.set_detect_anomaly(True):
            for i in range(epoch):
                y_pred = self.model(x)

                # Get loss
                loss = self.loss(y,y_pred)
                
                # Gradient Decent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # End of Gradient Decent
    
    def pred(self, y: np.array) -> np.array:
        y = torch.from_numpy(y)
        self.model.eval()
        y_pred = self.model(y)
        return y_pred.detach().numpy()