# Protein Universe Annotate
Using Deep Learning to Annotate the Protein Universe

## Problem description 

**Goal**: Develop a protein classifier that assigns the appropriate Pfam family (i.e. protein domain) to each protein sequence.

**Machine Learning Objective**: Reframe the task of predicting the protein domain as a multiclass classification problem. Given the amino acid sequence of the protein domain, the objective is to predict the class to which it belongs.

## Installation
```
git clone <GITHUB-URL>
cd protein_universe_annotate
pip install -r requirements.txt

# install the package
pip install -e .
```

## Data structure

The Pfam dataset was introduced in the publication - "Using deep learning to annotate the protein universe" [link](https://ai.googleblog.com/2022/03/using-deep-learning-to-annotate-protein.html). PFam Dataset Source - [Kaggle Datasets](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split?resource=download).  

Domains (target label) are functional sub-parts of proteins; 
much like images in ImageNet are pre segmented to contain exactly one object class, the PFam data is presegmented to contain exactly and only one domain.  

### Data Split
The data is partitioned into training/dev/testing folds randomly:

- Training data used to train Deep Learning models.
- Dev (development) data used in a close validation loop (e.g. hyperparameter tuning or model validation).
- Test data reserved for much less frequent evaluations - this helps avoid overfitting on the test data.

### File content
Each fold (train, dev, test) has a number of files in it. Each of those files contains csv on each line, which has the following fields:  

```
sequence: HWLQMRDSMNTYNNMVNRCFATCIRSFQEKKVNAEEMDCTKRCVTKFVGYSQRVALRFAE
family_accession: PF02953.15
sequence_name: C5K6N5_PERM5/28-87
aligned_sequence: ....HWLQMRDSMNTYNNMVNRCFATCI...........RS.F....QEKKVNAEE.....MDCT....KRCVTKFVGYSQRVALRFAE
family_id: zf-Tim10_DDP
```

Description of fields:
- sequence: These are usually the input features to your model. Amino acid sequence for this domain.
  There are 20 very common amino acids (frequency > 1,000,000), and 4 amino acids that are quite uncommon: X, U, B, O, Z.
- family_accession: These are usually the labels for your model. Accession number in form PFxxxxx.y 
  (Pfam), where xxxxx is the family accession, and y is the version number. 
  Some values of y are greater than ten, and so 'y' has two digits.
- family_id: One word name for family. 
- sequence_name: Sequence name, in the form "$uniprot_accession_id/$start_index-$end_index".
- aligned_sequence: Contains a single sequence from the multiple sequence alignment (with the rest of the members of 
  the family in seed, with gaps retained.

Generally, the `family_accession` field is the label, and the sequence (or aligned sequence) is the training feature. This sequence corresponds to a _domain_, not a full protein.

## Methodology 

### Approach 1: ProtCNN

The Pfam dataset was originally introduced in the publication "Using Deep Learning to Annotate the Protein Universe," in which the authors trained deep learning models to predict functional annotations for unaligned amino acid sequences and understand the relationship between amino acid sequence and protein function. The developed model - `ProtCNN` model has achieved significant results, demonstrating higher accuracy and computational efficiency compared to current state-of-the-art techniques such as BLASTp for annotating protein sequences. 

The suggested deep learning model, ProtCNN, is based on dilated convolutional neural networks (CNNs) that can model non-local pairwise amino acid interactions. The authors trained 1-dimensional CNNs (ResNet) to predict the classification of protein sequences and an ensemble of independently trained ProtCNN models (ProtCNN with multiple random seeds). In benchmarking tests, ProtCNN outperformed other state-of-the-art methods such as BLASTp. ProtENN, a simple majority vote across an ensemble of 19 ProtCNN models, further reduced the error rate. The authors also released the trained ProtCNN model, which can be used out of the box for inference since it was trained on exactly the same dataset for the same task.

Source Code for the publication [here](https://github.com/google-research/google-research/tree/master/using_dl_to_annotate_protein_universe)

### Approach 2: Fine-tune pre-trained protein Language Model

Proteins are composed of multiple amino acids, represented by a single letter. There are only a few different amino acids - the 20 most common in the Pfam dataset (standard ones), plus maybe a few rare ones. This means that a protein sequence can be considered just as a text string. For example, a protein with the amino acids Methionine, Alanine, and Histidine in a chain would be represented as the letters "MAH".  

The standard practice for training a transformer-based model on text is to use Transfer Learning by fine-tuning a pre-trained protein language model on the Pfam dataset. A neural network is trained for a long time on a text (=protein) task with abundant training data, and then the whole neural network is copied to a new task (=predict domain), changing only a few neurons that correspond to the network's output.  

The specific pre-trained model used in this approach is ESM-2, which is the state-of-the-art protein language model and publicly available [here](https://github.com/facebookresearch/esm). The pre-training phase has enabled the model to gain a significant understanding of protein structure based on the input of amino acids, similar to how language models comprehend language structure. This makes pre-trained protein models highly suitable for transferring to the task of predicting the domain given sequence of amino acids.  

## Experiments  

**Training Dataset**. In the following experiments, the training dataset size was reduced from the original 1 million to approximately 200,000 instances while maintaining the "long-tail" distribution of protein domain labels. This was done to meet the stringent technical requirements.  

**ProtCNN**. In order to evaluate the performance of ProtCNN and potentially improve upon it, several experiments were carried out. Training a ResNet-based model on encoded input sequences with the goal of reproducing the results of the original publication. To achieve better performance, a larger ResNet models could potentially be used. Additionally, an ensemble of ProtCNNs trained on different random seeds (or different sizes) could be created to boost the performance on the classification task.  

Since the trained ProtCNN model is publicly available, it was used to generate predictions and analyze its performance.  

Another experiment involved using ProtCNN to compute embeddings of input sequences and then using the embeddings to build a KNN algorithm and assign a domain label for test samples (test emebddings). To do this, the average learned representation for each family across its training sequences was computed, yielding a sentinel family representation. Then, each held-out test sequence was classified by finding its nearest sentinel in the space of learned representations using the KNN algorithm. This approach is supposed to yield comparable results as end-to-end predictions using the ProtCNN model, as indicated in the original publication. However, in my experiments this was not achieved.  

**Fine-tuning pre-trained protein Language Model**. To fine-tune the pre-trained protein Language Model, in initial experiment all the model's parameters were frozen except for the classifier layers (last softmax layer). However, the accuracy was below 1%, indicating that all layers must be updated (unfrozen) during fine-tuning.  

## Evaluation Results

TO-DO: To be updated  
