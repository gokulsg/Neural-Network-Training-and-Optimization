# Parts of the skeleton of this file is adapted from a UdS HLCV Course Assignment SS2021
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        """
        normalize and reshape each image to a flattened array. Note, this won't work with Tensor input now.
        i.e. cannot be used after ToTensor in transform.compose
        
        DO NOTE CHANGE ANYTHING INSIDE THIS FUNCTION!

        :param img: original PIL image obj
        :return: flattened image array in new_size
        """
        image = np.array(img)
        
        # normalize with cifar10 mean and std
        # cifar10 mean and std can be calculated accordingi:
        # (https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data)
        cifar_mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, -1)
        cifar_std = np.array([0.24703233, 0.24348505, 0.26158768]).reshape(1, 1, -1)
        image = (image - cifar_mean) / cifar_std
        
        return image.reshape(self.new_size)


def get_cifar10_dataset(val_size=1000, batch_size=200):
    """
    Load and transform the CIFAR10 dataset. Make Validation set. Create dataloaders for
    train, test, validation sets. Only train_loader uses batch_size of 200, val_loader and
    test_loader have 1 batch (i.e. batch_size == len(val_set) etc.)
    
    DO NOT CHANGE THE CODE IN THIS FUNCTION. YOU MAY CHANGE THE BATCH_SIZE PARAM IF NEEDED.
    
    If you get an error related num_workers, you may change that parameter to a different value.

    :param val_size: size of the validation partition
    :param batch_size: number of samples in a batch
    :return:
    """

    # the datasets.CIFAR getitem actually returns img in PIL format
    # no need to get to Tensor since we're working with our own model and not PyTorch
    transform = transforms.Compose([ReshapeTransform(32 * 32 * 3)])

    # Load the train_set and test_set from PyTorch, transform each sample to a flattened array
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    classes = train_set.classes

    # Split data and define train_loader, test_loader, val_loader
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set),
                                              shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_size,
                                             shuffle=False, num_workers=2)

    
    return train_loader, test_loader, val_loader, classes


"""
Q2 Tasks are inside the class NeuralNetowrkModel below. 
You're not allowed to use PyTorch built-in functions beyond this point.
Only Numpy and standard library built-in operations are allowed.

DO NOT CHANGE THE EXISTING CODE UNLESS SPECIFIED
"""


class NeuralNetworkModel:
    """
    A two-layer fully-connected neural network. The model has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.

    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        # DO NOT CHANGE ANYTHING IN HERE

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """

        Compute the loss and gradients for a two layer fully connected neural
        network.
        
        DO NOT CHANGE THIS FUNCTION
        
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength (lambda).

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = 0.

        # N == number of samples
        # D == number of features or dimensions per sample
        # Eq 3:
        a_1 = X  # shape: (N, D)

        # Eq 4:
        # input a_1 = X; X shape: (num samples N, num dimensions/features D)
        # Hidden 1 = X: (num samples N, input dimensions D) dot W1: (input dimensions D, hidden size H)
        # z = Hidden 1 + b1: (H,)
        z = np.dot(a_1, W1)  # shape: (N, H)
        z_2 = z + b1  # shape: (N, H)

        # Eq 5:
        # a_2 = relu(z_2)
        def elementwise_relu(u):
            return np.where(u >= 0, u, 0.0)

        a_2 = elementwise_relu(z_2)  # shape: (N, H)

        # Eq 6:
        # Hidden 2 = a:(N, H) dot W2:(H, C)
        # z = Hidden 2 + b2:(C,)
        z = np.dot(a_2, W2)  # shape: (N, C)
        z_3 = z + b2  # shape:(N, C)

        # Eq 7:
        # a_3 = softmax(z_3)
        def elementwise_row_softmax(u):
            # each row is a sample, so normalize over each row, hence sum over axis = 1
            # reshape denominator shape (N,) to match u.shape to divide each element in row by its row sum
            u -= np.max(u, axis=1, keepdims=True)
            return np.exp(u) / np.sum(np.exp(u), axis=1).reshape(-1, 1)

        a_3 = elementwise_row_softmax(z_3)  # shape: N, C

        try:
            assert np.all(np.isclose(np.sum(a_3, axis=1), 1.0))  # check that scores for each sample add up to 1
        except AssertionError:
            print(f'scores after softmax: \n{a_3}')
            print(f'sum of scores for all class: {np.sum(a_3, axis=1)}')

        scores = a_3

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = 0.

        # Eq 11
        # j = -log(softmax score corresponding to label y_i)
        # for each row in scores, select the score at the true label index
        j = -np.log(scores[range(len(scores)), y])

        # Eq 12
        j = np.average(j)

        # Eq 13
        loss = j + reg * (np.square(np.linalg.norm(W1)) + np.square(np.linalg.norm(W2)))

        # Backward pass: compute gradients
        grads = {}

        # Backpropagation Eq 16, 18-23
        # Eq 16: gradient wrt to W2 = dj_dz3 * dz_dw
        # dj_dz3 = (1/N) * (a_3 - delta : (N,C)) shape: (N,C)
        # delta is Eq 17: only subtract 1 from where the index of a_3 corresponding to true label
        num_samples = y.shape[0]
        a_3[range(num_samples), y] -= 1.0
        dj_dz3 = a_3 / num_samples  # shape: (N,C)

        # Eq 18, 19
        dz_dw = a_2  # shape: (N,H)
        # np.dot(dj_dz3.T, dz_dw).T == np.dot(dz_dw.T, dj_dz3) (H,N) x (N,C) --> (H,C)
        dj_dw2 = np.dot(dz_dw.T, dj_dz3)  # shape: (H,C)

        # Eq 20
        # gradient wrt W2 = dj_dw2 + 2 * reg * W2
        grads['W2'] = dj_dw2 + 2 * reg * W2  # shape: (H, C)

        # Eq 21: gradients wrt to b2
        # dj_db2 = dj_dz3 * dz3_db2
        # dz3_db2 = 1
        # So, dj_db2 = dj_dz3 + ddb2(regularization term)
        # dj/db2(reg * sqr(L2-norm W1) + sqr(L2-norm W2)) = 0
        dj_db2 = dj_dz3.sum(axis=0)  # shape: (N, C) -- > (C,) same as original b2 shape
        grads['b2'] = dj_db2

        # Eq 22: gradients wrt to W1 = dj_dz3 * dz3_da2 * da2_dz2 * dz2_dW1 + 2 * reg * W1
        # dj_dz3 = (1/N) * (a_3 - delta : (N,C)) shape: (N,C)
        # dz3_da2 = W2
        # da2_dz2 = derivative_relu(z2)
        # dz2_dw1 = a_1
        def derivative_relu(u):
            return np.where(u < 0, 0.0, 1.0)

        dj_da2 = np.dot(dj_dz3, W2.T)  # shape: dj_dz3 shape: (N, C) x W2.T shape: (C,H) --> (N,H)
        dj_dz2 = dj_da2 * derivative_relu(z_2)  # shape: (N,H)
        dz2_dw1 = a_1  # shape: (N, D)
        dj_dw1 = np.dot(dz2_dw1.T, dj_dz2)  # shape: dz2_dw1.T shape: (D,N) x dj_dz2 shape (N,H) --> (D, H)
        grads['W1'] = dj_dw1 + 2 * reg * W1  # shape: (D,H) + (D,H)

        # Eq 23: gradient wrt to b1
        # dj_db1 = dj_dz2 * dz2_db1
        # dz2_db1 = 1
        # So, dj_db1 = dj_dz2 + ddb1(regulrization term)
        # dj/db1(reg * sqr(L2-norm W1) + sqr(L2-norm W2)) = 0
        dj_db1 = dj_dz2.sum(axis=0)  # shape: (N,H) --> (H,) same as original b1 shape
        grads['b1'] = dj_db1

        return loss, grads

    def train(self, train_dataloader, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_epochs=100, verbose=True):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - train_dataloader: PyTorch Dataloader for the training set with pre-determined batch size;
          each batch in the dataloader is of shape (N_training samples/batch, D)
        - X_val: An array of shape (N_val, D) giving validation data.
        - y_val: An array of shape (N_val,) giving validation labels; y_val[i] = c means that
          X_val[i] has label c, where 0 <= c < C.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_epochs: Number of steps to take when optimizing.
        - verbose: boolean; if true print progress during optimization.
        """

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_epochs):
            epoch_train_acc = []
            # TODO: get a random minibatch of training data and labels, storing
            #  them in X_train and y_train respectively.
            #  Compute loss and gradients using the current minibatch
            #  store each batch's loss in loss_history
            
            for batch_idx, (X_train, y_train) in enumerate(train_dataloader):
                
                loss, grads = self.loss(X_train.numpy(), y_train.numpy())
                # TODO: Use the gradients in the grads dictionary to update the parameters of the network
                #  (stored in the dictionary self.params)
                # updating weights from each iteration's gradients * learning rate
                # per Algorithm 1
                
                self.params['W1'] -= grads['W1'] * learning_rate
                self.params['b1'] -= grads['b1'] * learning_rate
                self.params['W2'] -= grads['W2'] * learning_rate
                self.params['b2'] -= grads['b2'] * learning_rate
                
                loss_history.append(loss)

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                if batch_idx % 100 == 0:
                    if verbose:
                        print('batch %d / %d: loss %f' % (batch_idx + 1, len(train_dataloader), loss))
                    epoch_train_acc.append((self.predict(X_train) == y_train.numpy()).mean())

            # Every epoch, check train and val accuracy and decay learning rate.
            val_acc_history.append((self.predict(X_val) == y_val.numpy()).mean())
            # Decay learning rate
            learning_rate *= learning_rate_decay

            # track training, valiation accuracy for each epoch and epoch loss (estimate from last batch in the epoch)
            train_acc_history.append(epoch_train_acc[-1])
            if verbose:
                print(f'Epoch {it + 1} / {num_epochs}:\nTraining Accuracy: {np.mean(epoch_train_acc)}\n'
                      f'Validation Accuracy: {val_acc_history[-1]}\n'
                      f'Loss: {loss_history[-1]}')

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        def elementwise_relu(u):
            return np.where(u >= 0, u, 0.0)
        
        def elementwise_row_softmax(u):
            u -= np.max(u, axis=1, keepdims=True)
            return np.exp(u) / np.sum(np.exp(u), axis=1).reshape(-1, 1)
        
        y_pred = elementwise_relu(np.dot(X, self.params['W1']) + self.params['b1'])
        y_pred = elementwise_row_softmax(np.dot(y_pred, self.params['W2']) + self.params['b2'])
        y_pred = y_pred.argmax(1)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    
def find_best_model():
    # TODO: Implement a function to find a model with most optimal hyperparameters
    
    params = {"lr" : 1e-4, "lr_decay" : 0.95, "reg" : 0.4 , "epoch" : 25}
    return params

