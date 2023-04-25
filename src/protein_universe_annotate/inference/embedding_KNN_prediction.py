import argparse
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from protein_universe_annotate.data_processing import read_pfam_dataset


def calc_KNN_preds(train_embeddings_path,
                   data_path,
                   test_embeddings_path,
                   num_neighbors,
                   do_standardize=False,
                   target_dim=None,
                   save_path='KNN_predictions.npy'):
    """
    Calculate KNN predictions using train embeddings and test embeddings.

    Args:
        train_embeddings_path (str): Path to train embeddings file.
        data_path (str): Path to directory containing train dataset files.
        test_embeddings_path (str): Path to test embeddings file.
        num_neighbors (int): Number of neighbors to consider for KNN prediction.
        do_standardize (bool): Do standardization prior to fitting KNN
        target_dim (int, optional): Dimension of embeddings to reduce to using PCA. Defaults to None (=dim of embedding).
        save_path (str): Path to save the computed predictions
    Returns:
        np.ndarray: Predicted labels for test set.
    """

    # Load train embeddings and test embeddings
    train_embeddings = np.load(train_embeddings_path, allow_pickle=True)
    test_embeddings = np.load(test_embeddings_path, allow_pickle=True)

    # Read train dataset
    train_df = read_pfam_dataset('train', data_path)

    # Sort train_df by sequence length so that batches the order of train embeddings
    train_df = train_df.sort_values('sequence', key=lambda col: [len(c) for c in col])

    # Add true label column to train_df
    train_df['true_label'] = train_df.family_accession.apply(lambda s: s.split('.')[0])

    # Get train labels and ensure the length matches the embeddings length
    train_labels = train_df['true_label'].values
    assert len(train_labels) == len(train_embeddings)

    # Group train embeddings by label to calculate the mean(embeddings) for a label
    label_grouped_embeddings = defaultdict(list)
    for sample in zip(train_embeddings, train_labels):
        embed, label = sample[0], sample[1]
        label_grouped_embeddings[label].append(embed)

    # Get label learned representation by taking the mean of the grouped embeddings
    label_learned_representation = dict()
    for label, embeddings in label_grouped_embeddings.items():
        label_learned_representation[label] = np.mean(embeddings, axis=0)

    # Get labels and representations as lists - input to KNN
    labels = list(label_learned_representation.keys())
    representations = list(label_learned_representation.values())

    # Standardization
    if do_standardize:
        sc = StandardScaler()
        sc.fit(representations)
        representations = sc.transform(representations)
        test_embeddings = sc.transform(test_embeddings)

    # Reduce embeddings dimension using PCA (avoid curse of dimensionality)
    if target_dim:
        pca = PCA(n_components=target_dim)
        representations = pca.fit_transform(representations)
        test_embeddings = pca.transform(test_embeddings)

    # Fit KNN classifier and make predictions on test set
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, n_jobs=-1)
    knn.fit(representations, labels)
    test_predictions = knn.predict(test_embeddings)

    # Save the embeddings
    if save_path:
        with open(save_path, 'wb') as f:
            np.save(f, np.array(test_predictions))

    return test_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate KNN predictions based on embeddings of ProtCNN')
    parser.add_argument('--train_embeddings_path', type=str, required=True,
                        help='Path to the training embeddings file (in numpy format)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the directory containing the pfam dataset')
    parser.add_argument('--test_embeddings_path', type=str, required=True,
                        help='Path to the test embeddings file (in numpy format)')
    parser.add_argument('--num_neighbors', type=int, default=1,
                        help='Number of neighbors for KNN classification (default: 1)')
    parser.add_argument('--target_dim', type=int, default=None,
                        help='Number of dimensions for PCA transformation (default: None)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='The output path where to save the computed predictions')
    args = parser.parse_args()

    calc_KNN_preds(train_embeddings_path=args.train_train_embeddings_path,
                   data_path=args.data_path,
                   test_embeddings_path=args.test_embeddings_path,
                   num_neighbors=10,
                   do_standardize=False,
                   target_dim=None,
                   save_path=args.save_path)

