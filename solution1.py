
import matplotlib.pyplot as plt
import numpy as np

def val_acc_plot(results):
    x = [i for i in range(len(results['train_acc_history']))]
    plt.plot(x, results['train_acc_history'], label="Train accuracy")
    plt.plot(x, results['val_acc_history'], label="Validation accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title('Training and validation accuracies vs Iterations(Epochs)')
    plt.legend()
    plt.show()
    
    
def loss_plot(results):
    x = [i for i in range(len(results['loss_history']))]
    plt.plot(x, results['loss_history'], label="loss")

    plt.xlabel("Loss update steps")
    plt.ylabel("Loss")
    plt.title('Loss vs loss update steps')
    plt.legend()
    plt.show()