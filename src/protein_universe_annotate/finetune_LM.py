import argparse
import numpy as np
from sklearn import preprocessing
from datasets import Dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, HfArgumentParser, Trainer


from protein_universe_annotate.data_processing import read_pfam_dataset, select_long_tail
from protein_universe_annotate.utils import save_label_encoder


def _compute_metrics(eval_pred):
    """
    Computes the accuracy of the predictions.
    Args:
        eval_pred: A tuple containing the predictions and labels.
    Returns:
        The accuracy of the predictions.
    """
    metric = load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def fine_tune_protein_lm(model_checkpoint_name,
                         data_path,
                         freeze_model,
                         save_path,
                         batch_size):
    """
    Fine-tunes a pre-trained protein language model on a Pfam dataset.

    Args:
        model_checkpoint_name (str): The name or path of the pre-trained model checkpoint to use for fine-tuning.
        data_path (str): The path to the directory containing the protein Pfam dataset.
        freeze_model (bool): Whether to freeze the weights of the pre-trained model or not during fine-tuning.
        save_path (str): The path to the directory where the fine-tuned model will be saved.
        batch_size (int): The number of examples per batch during training.

    Returns:
        None
    """
    # Load train and test datasets
    train_df = read_pfam_dataset('train', data_path)
    test_df = read_pfam_dataset('test', data_path)

    # Convert true labels from PF00001.21 to PF00001
    train_df['true_label'] = train_df.family_accession.apply(lambda s: s.split('.')[0])
    test_df['true_label'] = test_df.family_accession.apply(lambda s: s.split('.')[0])

    # Reduce the training dataset size by keeping on long-tail classes (due to computation limits)
    train_selected = select_long_tail(train_df, col_name='true_label')
    print(f'The size of the training dataset: {len(train_selected)}')

    # Get sequences and labels
    train_sequences = train_selected["sequence"].tolist()
    train_labels = train_selected["true_label"].tolist()
    test_sequences = test_df["sequence"].tolist()
    test_labels = test_df["true_label"].tolist()

    # Encode labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(train_labels)
    num_classes = len(list(label_encoder.classes_))
    train_labels = label_encoder.transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    # Save the label-encoding mapping
    save_label_encoder(label_encoder, save_path)

    # Tokenize sequences
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_name)
    train_tokenized = tokenizer(train_sequences)
    test_tokenized = tokenizer(test_sequences)

    # Create train and test datasets
    train_dataset = Dataset.from_dict(train_tokenized)
    test_dataset = Dataset.from_dict(test_tokenized)
    train_dataset = train_dataset.add_column("labels", train_labels)
    test_dataset = test_dataset.add_column("labels", test_labels)

    # Load the pre-trained model (init the last layer classifier to predict protein domains)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_name, num_labels=num_classes)

    # Freeze model if specified
    if freeze_model:
        for param in model.esm.parameters():
            param.requires_grad = False

    # Print number of parameters
    print(f'Total number of parameters in a model: {model.num_parameters()}')
    print(f'The number of trainable parameters: {model.num_parameters(only_trainable=True)}')

    # Define training arguments
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_json_file(json_file='lm_training_args.json')
    
    # Train the model
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(output_dir=save_path)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained protein language model on PFam Dataset')
    parser.add_argument('--model_checkpoint_name', type=str, required=True,
                        help='Name of pre-trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to directory containing data')
    parser.add_argument('--freeze_model', action='store_true', help='Whether to freeze the pre-trained model')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to directory for saving fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training/evaluation')
    args = parser.parse_args()

    parser = argparse.ArgumentParser(description='Fine-Tune the Transformer LM pre-trained on protein sequences')

    fine_tune_protein_lm(model_checkpoint_name=args.model_checkpoint_name,
                         data_path=args.data_path,
                         freeze_model=args.freeze_model,
                         save_path=args.save_path,
                         batch_size=args.batch_size)

