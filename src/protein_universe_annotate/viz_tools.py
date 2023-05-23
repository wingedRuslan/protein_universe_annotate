import matplotlib.pyplot as plt


def plot_loss_curves(epoch_loss, save_path):
    """
    Plot the training and validation loss curves.

    Args:
        epoch_loss (dict): Dictionary containing training and validation loss values for each epoch.
        save_path (str): Path to save the plot.
    """

    # Create the plot
    plt.plot(epoch_loss['train'], label='Train Loss')
    plt.plot(epoch_loss['val'], label='Validation Loss')

    # Set labels and title
    plt.legend()
    plt.title('Loss over Epochs')
    plt.ylabel('Average Cross Entropy Loss\n(approximated by averaging over minibatches)')
    plt.xlabel('Epoch')

    # Save the plot
    plt.savefig(f'{save_path}/loss_curves.png')
    plt.show()
    plt.close()


def plot_accuracy_curves(epoch_acc, save_path):
    """
    Plot the training and validation accuracy curves.

    Args:
        epoch_acc (dict): Dictionary containing training and validation accuracy values for each epoch.
        save_path (str): Path to save the plot.
    """

    # Create the plot
    plt.plot(epoch_acc['train'], label='Train Accuracy')
    plt.plot(epoch_acc['val'], label='Validation Accuracy')

    # Set labels and title
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    # Save the plot
    plt.savefig(f'{save_path}/accuracy_curves.png')
    plt.show()
    plt.close()


def plot_loss_and_accuracy_curves(epoch_loss, epoch_acc, save_path):
    """
    Plot the training and validation loss curves and accuracy curves on the same figure.
    Args:
        epoch_loss (dict): Dictionary containing training and validation loss values for each epoch.
        epoch_acc (dict): Dictionary containing training and validation accuracy values for each epoch.
        save_path (str): Path to save the plot.
    """
    # Create a new figure and specify the layout
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Plot the loss curves on the first axis
    axs[0].plot(epoch_loss['train'], label='Train Loss')
    axs[0].plot(epoch_loss['val'], label='Validation Loss')
    axs[0].legend()
    axs[0].set_title('Loss over Epochs')
    axs[0].set_ylabel('Avg Cross Entropy Loss\n(approximated by averaging over minibatches)')
    axs[0].set_xlabel('Epoch')

    # Plot the accuracy curves on the second axis
    axs[1].plot(epoch_acc['train'], label='Train Accuracy')
    axs[1].plot(epoch_acc['val'], label='Validation Accuracy')
    axs[1].legend()
    axs[1].set_title('Accuracy over Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{save_path}/loss_and_accuracy_curves.png')
    plt.show()
    plt.close()

