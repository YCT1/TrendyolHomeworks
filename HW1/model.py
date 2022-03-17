# Yekta Can Tursun



from turtle import shape
import numpy as np


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
        for i in range(1000):
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

