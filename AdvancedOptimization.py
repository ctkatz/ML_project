import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the linear scores for input features.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).

        Returns:
            Tensor: Score vector of shape (n_samples,).
        """
        # Initialize the weights if not already done
        if self.w is None:
            self.w = torch.zeros(X.shape[1])

        return X @ self.w

    def predict(self, X):
        """
        Make binary predictions based on the sign of the score.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).

        Returns:
            Tensor: Binary predictions in {0.0, 1.0}, shape (n_samples,).
        """
        s = self.score(X)
        return (s >= 0).float()

class LogisticRegression(LinearModel):
    def sig(self, s):
        """
        Apply the sigmoid function element-wise.

        Args:
            s (Tensor): Input tensor.

        Returns:
            Tensor: Sigmoid activation of input tensor.
        """
        
        return 1 / (1 + torch.exp(-s))

    def loss(self, X, y):
        """
        Compute the average logistic loss over the dataset.
        """
        s = self.score(X)
        epsilon = 1e-8  # Small constant to avoid log(0)
        sig_s = self.sig(s + epsilon)
        sig_s = torch.clamp(sig_s, epsilon, 1 - epsilon) # prevent our sigmoid from reaching exactly 0 or 1        

        loss = -y * torch.log(sig_s) - (1 - y) * torch.log(1 - sig_s)

        return loss.mean()


    def grad(self, X, y): 
        """
        Compute the gradient of the logistic loss.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).
            y (Tensor): Binary labels of shape (n_samples,).

        Returns:
            Tensor: Gradient vector of shape (n_features,).
        """
        y = y.float()
        s = self.score(X)
        sig_s = self.sig(s)
        gradient = (sig_s - y).unsqueeze(1) * X
        return gradient.mean(dim=0)

    def hessian(self, X):
        """
        Compute the Hessian matrix for logistic regression.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).

        Returns:
            Tensor: Hessian matrix of shape (n_features, n_features).
        """
        s = X @ self.w  
        sig = self.sig(s)  
        D = sig * (1 - sig)  
        weighted_X = X * D.unsqueeze(1)
        H = X.T @ weighted_X
        return H + 1e-6 # add a very small term on the end to prevent singular matrices from breaking the hessian

class NewtonOptimizer():
    def __init__(self, model):
        self.model = model 
        self.prev = None

    def step(self, X, y, alpha, Beta):
        """
        Perform one Newton update step for logistic regression.

        Args:
            X (Tensor): Feature matrix of shape (n_samples, n_features).
            y (Tensor): Binary labels of shape (n_samples,).
            alpha (float): Learning rate.
            Beta (float): Momentum term.

        Returns:
            None
        """
        # Initialize the weights if not already done
        if self.model.w is None:
            self.model.w = torch.zeros(X.shape[1])

        # Compute gradient and inverse Hessian
        grad = self.model.grad(X, y)
        H_inv = torch.linalg.pinv(self.model.hessian(X)) 

        # If there was no previous step, start with only the Newton step
        if self.prev is None:
            self.prev = self.model.w
            self.model.w = self.model.w - alpha * (H_inv @ grad)

        # If there was a previous step, apply momentum
        else:
            momentum = Beta * (self.model.w - self.prev)
            self.prev = self.model.w
            self.model.w = self.model.w - alpha * (H_inv @ grad) + momentum
