import numpy as np
import torch
import torch.nn.functional as F


class SoftmaxRegression(torch.nn.Module):
    """ Softmax regression model for multi-class classification """

    def __init__(self, num_features, num_classes):
        """
        Args:
            num_features  (int): Number of input features
            num_classes (int): Number of output classes
        """
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the SoftmaxRegression model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            logits (torch.Tensor): Logits before applying softmax.
            probas (torch.Tensor): Probability distribution over the classes.
        """
        # Pass x through a parameterized linear transformation
        logits = self.linear(x)

        # Pass logits through softmax to generate a probability distribution vector over the classes
        probas = F.softmax(logits, dim=1)
        return logits, probas


class ProteinModel:
    """ Wrapper class for training and evaluating deep learning models. """

    def __init__(self, model):
        """
            Initialize the backbone ProteinModel.
        Args:
            model (torch.nn.Module): The protein model.
        """
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def forward(self, x):
        """ Perform forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def train(self, dataloader, loss_fn, optimizer):
        """
        Train the protein model.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader object to use for training.
            loss_fn: Loss function for training.
            optimizer: Optimizer for training.

        Returns:
            loss_avg (float): Average loss value.
            acc_total (float): Accuracy.
        """
        # Number of samples explored
        num_samples = 0

        # Variables to collect overall loss and accuracy
        loss_total = 0.
        acc_total = 0.

        self.model.train()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            num_samples += inputs.size(0)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Compute the model predictions
            logits, probas = self.model(inputs)

            # Compute the loss value for the mini-batch
            # CrossEntropyLoss() works with logits, not probabilities
            loss = loss_fn(logits, labels)

            # Zero out the gradients of all the parameters in the optimizer
            optimizer.zero_grad()

            # Compute the gradient w.r.t. to parameters
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Cummulate loss in loss_total
            loss_total += float(loss.detach().item())

            # Cummulate number of correct classifications in acc_total
            acc_total += float(
                (torch.argmax(probas, dim=1) == labels).sum()
            )

            # Logging - display the current progress
            if batch_idx % 250 == 0:
                train_acc = 100 * acc_total / num_samples
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Accuracy: {train_acc:.2f}%')

        # Divide by total number of visited samples
        loss_avg = loss_total / num_samples
        acc_total = acc_total / num_samples

        return loss_avg, acc_total

    def evaluate(self, dataloader, loss_fn):
        """
        Evaluate the protein model.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader object to use for evaluation.
            loss_fn: Loss function for evaluation.
        Returns:
            loss_avg (float): Average loss value.
            acc_total (float): Accuracy.
        """
        num_samples = 0

        loss_total = 0.
        acc_total = 0.

        self.model.eval()

        with torch.no_grad():
            for inputs, labels in dataloader:
                num_samples += inputs.size(0)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits, probas = self.model(inputs)

                loss_total += float(loss_fn(logits, labels).detach().item())
                acc_total += float(
                    (torch.argmax(probas, dim=1) == labels).sum()
                )

        loss_avg = loss_total / num_samples
        acc_total = acc_total / num_samples

        return loss_avg, acc_total

    def predict(self, dataloader):
        """
        Predicts labels using the provided model and data loader.

        Args:
            model (torch.nn.Module): Model for prediction.
            dataloader (torch.utils.data.DataLoader): Data loader object to use for prediction.

        Returns:
            total_preds (numpy.ndarray): Array containing all predicted labels.
            acc_total (float): Accuracy.
        """

        num_samples = 0

        acc_total = 0.

        self.model.eval()  # Set the model in evaluation mode

        total_preds = []  # Keep the predictions

        with torch.no_grad():  # Disable gradient computation for efficiency
            for inputs, labels in dataloader:
                num_samples += inputs.size(0)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                _, probas = self.model.forward(inputs)
                predictions = torch.argmax(probas, dim=1)
                total_preds.append(predictions.cpu().numpy())

                acc_total += torch.sum(predictions == labels).item()

        acc_total = acc_total / num_samples
        total_preds = np.concatenate(total_preds)  # List of mini-batch predictions -> single continuous array

        return total_preds, acc_total

    def save_model(self, state_dict_model_path='./state_dict_model.pt'):
        """
        Recommended approach - save the state dictionary of the model (save the trained model’s learned parameters).
        Source - https://pytorch.org/tutorials/beginner/saving_loading_models.html
        Args:
            state_dict_model_path (str): Path to save the model (e.g. '../../outputs/state_dict_model.pt)
        """
        torch.save(self.model.state_dict(), state_dict_model_path)

    def load_model(self, state_dict_model_path):
        """
        Load the model state dictionary from a file.
        Note: ProteinModel.model has to be the instance of the appropriate TheModelClass()
              to load the corresponding trained model.
        Args:
            state_dict_model_path (str): Path to the model file.
        """
        self.model.load_state_dict(torch.load(state_dict_model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

