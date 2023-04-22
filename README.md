# Protein Universe Annotate
Using Deep Learning to Annotate the Protein Universe

## Problem description 

**Goal**: predict the function of protein domains, based on the [PFam dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split?resource=download).

Domains are functional sub-parts of proteins; 
much like images in ImageNet are pre segmented to contain exactly one object class, the PFam data is presegmented to contain exactly and only one domain.

**ML task**: multiclass classification machine learning task. 
The objective is to determine the category of a protein domain by using its amino acid sequence as input. 
The training dataset contains roughly 1 million examples, and there are 18,000 possible output categories.

## Data structure
This data is introduced and thoroughly described by the publication "Can Deep Learning Classify the Protein Universe", Bileschi et al, [link](https://ai.googleblog.com/2022/03/using-deep-learning-to-annotate-protein.html).

### Data split and layout
The data is partitioned into training/dev/testing folds randomly:

- Training data used to train Deep Learning models.
- Dev (development) data used in a close validation loop (e.g. hyperparameter tuning or model validation).
- Test data reserved for much less frequent evaluations - this helps avoid overfitting on your test data.

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

Generally, the `family_accession` field is the label, and the sequence (or aligned sequence) is the training feature.

This sequence corresponds to a _domain_, not a full protein.

