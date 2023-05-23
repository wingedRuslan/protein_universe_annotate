import pandas as pd
import numpy as np
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader

from protein_universe_annotate.data_processing import fit_count_vectorizer
from protein_universe_annotate.ProteinDataset import ProteinDataset
from protein_universe_annotate.ProteinModel import SoftmaxRegression, ProteinModel
from protein_universe_annotate.viz_tools import plot_loss_curves, plot_loss_and_accuracy_curves


def train_softmax_regression(train_data_path, test_data_path, dev_data_path, output_dir,
                             max_features=1000, batch_size=128, learning_rate=0.001, num_epochs=10):
    """
    Train a softmax regression model using the provided data paths.

    Args:
        train_data_path (str): Path to the training data file.
        test_data_path (str): Path to the test data file.
        dev_data_path (str): Path to the development data file.
        output_dir (str): Path to save the trained model and other outputs.
        max_features (int, optional): Maximum number of features for the CountVectorizer. Defaults to 1000.
        batch_size (int, optional): Batch size for training. Defaults to 128.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.001.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.

    Returns:
        None
    """

    # Read the data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    dev_df = pd.read_csv(dev_data_path)

    # Get all input protein sequences to learn BoW feature representation
    all_sequences = train_df['sequence'].tolist() + dev_df['sequence'].tolist() + test_df['sequence'].tolist()

    # Fit CountVectorizer to learn the vocabulary and set indices for amino acids
    ngrams_bow_model = fit_count_vectorizer(ngram_range=(1, 3), max_features=max_features,
                                            input_sequences=all_sequences)

    num_features = len(ngrams_bow_model.vocabulary_)
    num_classes = len(np.unique(np.concatenate((train_df["true_label_encoded"].values,
                                                test_df["true_label_encoded"].values,
                                                dev_df["true_label_encoded"].values))))

    # Define a custom dataset class to be used with data loader object
    datasets = {
        name: ProteinDataset(raw_seqs=data_df['sequence'],
                             seq_encoder=ngrams_bow_model,
                             labels=data_df["true_label_encoded"].values)
        for name, data_df in zip(['train', 'dev', 'test'], [train_df, dev_df, test_df])
    }

    # Create a data loader object for each partition
    # Set shuffle parameter to True only for the training data loader
    dataloaders = {
        name: DataLoader(datasets[name],
                         batch_size=batch_size,
                         shuffle=(name == 'train'))
        for name in datasets
    }

    # Initialize softmax regression model, loss function, and optimizer
    softmax_regression_model = SoftmaxRegression(num_features=num_features,
                                                 num_classes=num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(softmax_regression_model.parameters(), lr=learning_rate)

    # Initialize ProteinModel with softmax regression model as a backbone for training and evaluation
    model = ProteinModel(softmax_regression_model)

    start_time = time.time()
    epoch_loss = {'train': [], 'val': []}
    epoch_acc = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        train_loss_avg, train_acc_total = model.train(dataloaders['train'], loss_fn, optimizer)
        val_loss_avg, val_acc_total = model.evaluate(dataloaders['dev'], loss_fn)

        epoch_loss['train'].append(train_loss_avg)
        epoch_loss['val'].append(val_loss_avg)
        epoch_acc['train'].append(train_acc_total)
        epoch_acc['val'].append(val_acc_total)

        print(f'\n[EPOCH:{epoch + 1:3d}/{num_epochs}]',
              f'train.loss: {train_loss_avg:.4f}',
              f'train.acc: {100 * train_acc_total:3.2f}%',
              f'val.loss: {val_loss_avg:.4f}',
              f'val.acc: {100 * val_acc_total:3.2f}%')
        print(f'Time elapsed: {((time.time() - start_time) / 60):.2f} mins \n')

    # Plot loss and accuracy curves
    plot_loss_and_accuracy_curves(epoch_loss=epoch_loss, epoch_acc=epoch_acc, save_path=output_dir)

    # Evaluate performance on the held-out test dataset
    test_loss_total, test_acc_total = model.evaluate(dataloaders['test'], loss_fn)
    print(f'[Test set performance]',
          f'test.loss: {test_loss_total:.4f}',
          f'test.acc: {100 * test_acc_total:3.2f}%')

    # Predict the labels on the test dataset
    test_predictions, test_acc = model.predict(dataloaders['test'])

    # Save the trained model
    model.save_model(save_path=output_dir)

    # Save predicted labels
    test_df['softmaxreg_preds'] = test_predictions
    test_df.to_csv(f'{output_dir}/softmaxreg_test_preds.csv', index=False)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train SoftMax Regression model on PFam Dataset')
    parser.add_argument("train_data_path", type=str, help="Path to the training data csv file")
    parser.add_argument("test_data_path", type=str, help="Path to the test data csv file")
    parser.add_argument("dev_data_path", type=str, help="Path to the development data csv file")
    parser.add_argument("output_dir", type=str, help="Path to save the outputs of the training")
    parser.add_argument("--max_features", type=int, default=1000, help="Maximum number of features for CountVectorizer "
                                                                       "(ngrams Bag-of-Words feature representation)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")

    args = parser.parse_args()

    train_softmax_regression(args.train_data_path,
                             args.test_data_path,
                             args.dev_data_path,
                             args.output_dir,
                             max_features=args.max_features,
                             batch_size=args.batch_size,
                             learning_rate=args.learning_rate,
                             num_epochs=args.num_epochs)

