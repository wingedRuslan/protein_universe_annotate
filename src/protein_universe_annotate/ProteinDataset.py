from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """ Protein domain family (Pfam) dataset """

    def __init__(self, raw_seqs, seq_encoder, labels):
        """
        Args:
          raw_seqs (pandas.DataFrame): protein sequences
          seq_encoder (CountVectorizer | OneHotEncoder): encoder object for the sequences
        """
        self.raw_seqs = raw_seqs
        self.seq_encoder = seq_encoder
        self.labels = torch.tensor(labels, dtype=torch.int64)

        # Encode the protein sequences according to the type of seq_encoder
        if isinstance(seq_encoder, CountVectorizer):
            self.encoded_seqs = self._encoding_ngrams_bow()
            self.encoded_seqs = torch.tensor(self.encoded_seqs, dtype=torch.float32)
        else:
            raise ValueError('seq_encoder must be either a OneHotEncoder or a CountVectorizer!')

    def __getitem__(self, index):
        seq = self.encoded_seqs[index]
        label = self.labels[index]
        return seq, label

    def __len__(self):
        return len(self.raw_seqs)

    def _encoding_ngrams_bow(self):
        # Generate n-grams Bag-of-Words feature representation
        return self.seq_encoder.transform(self.raw_seqs).todense()

    def _encoding_one_hot(self):
        # To-Do
        return None
