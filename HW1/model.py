# Yekta Can Tursun



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

        print("starting sgd")
        for i in range(1000):
            y_pred: np.ndarray = self.beta[0] + self.beta[1] * x


            # Calculate the grad vector
            g_b0,g_b1 = self.grad(y,y_pred,x)

            print(f"({i}) beta: {self.beta}, gradient: {g_b0} {g_b1}")

            beta_prev = np.copy(self.beta)

            self.beta[0] = self.beta[0] - alpha * g_b0
            self.beta[1] = self.beta[1] - alpha * g_b1

            if np.linalg.norm(self.beta - beta_prev) < alpha:
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



class ThresholdBasedModel(ModelBase):
    def __init__(self, threshold, lam) -> None:
        super().__init__()
        self.lam = lam
        self.threshold = threshold
    
    def grad(self, y: np.array, y_pred: np.array, x: np.array):
        g_b0 = -2 * (y - y_pred).mean() + 2 * self.lam * self.beta[0]
        g_b1 = -2 * (x * (y - y_pred)).mean() + 2 * self.lam * self.beta[1]
        return g_b0, g_b1