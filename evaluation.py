import torch
import torch.nn as nn
import numpy as np
from SignalPeptidePredictor import SignalPeptidePredictorCNN2


def encode_with_key(sequences, encoding_key):
    aa_to_index = {amino_acid: index for index, amino_acid in enumerate(encoding_key)}

    max_length = max(len(seq) for seq in sequences)

    encoded_sequences = []

    for sequence in sequences:
        one_hot_matrix = np.zeros((max_length, len(encoding_key)))

        for i, aa in enumerate(sequence):
            if aa in aa_to_index:
                one_hot_matrix[i, aa_to_index[aa]] = 1
            else:
                raise ValueError(f"Unknown amino acid: {aa}")

        encoded_sequences.append(one_hot_matrix)
    return encoded_sequences


def simple_encoding(sequences):
    encoded_sequences = []
    for sequence in sequences:
        if 'S' in sequence or 'T' in sequence or 'L' in sequences or 'P' in sequence:
            encoded_sequences.append([1, 0, 0])
        elif 'M' in sequence:
            encoded_sequences.append([0, 1, 0])
        else:
            encoded_sequences.append([0, 0, 1])
    return encoded_sequences


def encode_organisms(all_organisms):
    output = []
    for o in all_organisms:
        if o == 'EUKARYA':
            output.append([1, 0, 0])
        elif o == 'POSITIVE':
            output.append([0, 1, 0])
        elif o == 'NEGATIVE':
            output.append([0, 0, 1])
    return output


def encode_sequences(sequences_in):
    sequences = [s[0] for s in sequences_in]
    outputs = [s[1] for s in sequences_in]
    organisms = [s[2] for s in sequences_in]

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    encoded_sequences = encode_with_key(sequences, amino_acids)
    encoded_peptides = simple_encoding(outputs)
    encided_organisms = encode_organisms(organisms)

    result = [[s, p, o] for s, p, o in zip(encoded_sequences, encoded_peptides, encided_organisms)]

    return result


def decode_with_key(sequence, encoding_key):
    index_to_aa = {index: amino_acid for index, amino_acid in enumerate(encoding_key)}
    output = []
    for vector in sequence:
        output.append(index_to_aa[np.argmax(vector)])
    return output


def binarize_output(input_sequence):
    input_sequence = input_sequence.reshape((70, 7))
    output = np.zeros(input_sequence.shape)
    output[(np.arange(output.shape[0]), np.argmax(input_sequence, axis=1))] = 1
    return output


def decode_sequences(sequences_in):
    decoded_sequences = []
    for s in sequences_in:
        binarized_output = binarize_output(s)
        decoded_sequences.append(decode_with_key(binarized_output, 'STLPIMO'))
    return decoded_sequences


def labeling(dataset):
    return [[el[1], el[2], el[0].split('|')[1]] for el in dataset]


def f1_loss(y_true, y_pred, beta=1) -> np.float32:

    tp = (y_true * y_pred).sum(dim=-1)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=-1)
    fp = ((1 - y_true) * y_pred).sum(dim=-1)
    fn = (y_true * (1 - y_pred)).sum(dim=-1)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + epsilon)

    return f1.mean(), recall.mean(), precision.mean()


path_to_saved_model = r'C:\Users\ivana\PycharmProjects\PeptideSignal\trained_model\epoch_988_0.35008442401885986.pth'
loaded_model = torch.load(path_to_saved_model, map_location=torch.device('cpu'))
fasta_file = r"D:/Downloads/train_set.fasta"

f = open(fasta_file, "r")
all_lines = f.readlines()
lines = [l.strip('\n') for l in all_lines]
groups = [lines[i:i+3] for i in range(0, len(lines), 3)]

train_ds = []
val_ds = []
test_ds = []
for group in groups:
    if group[0][-1] == '0':
        train_ds.append(group)
    elif group[0][-1] == '1':
        val_ds.append(group)
    elif group[0][-1] == '2':
        test_ds.append(group)

labeled_train_set = labeling(train_ds)
labeled_val_set = labeling(val_ds)
labeled_test_ds = labeling(test_ds)

encoded_train_ds = encode_sequences(labeled_train_set)
encoded_val_ds = encode_sequences(labeled_val_set)
encoded_test_ds = encode_sequences(labeled_test_ds)

X_train = [el[0] for el in encoded_train_ds]
X_train_2 = [el[2] for el in encoded_train_ds]
y_train = [el[1] for el in encoded_train_ds]

X_val = [el[0] for el in encoded_val_ds]
X_val_2 = [el[2] for el in encoded_val_ds]
y_val = [el[1] for el in encoded_val_ds]

X_test = [el[0] for el in encoded_test_ds]
X_test_2 = [el[2] for el in encoded_test_ds]
y_test = [el[1] for el in encoded_test_ds]

X_train_tensor = torch.tensor(X_train).float()
X_train_2 = torch.tensor(X_train_2).float()
y_train_tensor = torch.tensor(y_train).float()
X_val_tensor = torch.tensor(X_val).float()
X_val_2 = torch.tensor(X_val_2).float()
y_val_tensor = torch.tensor(y_val).float()
X_test_tensor = torch.tensor(X_test).float()
X_test_2 = torch.tensor(X_test_2).float()
y_test_tensor = torch.tensor(y_test).float()


loaded_model.eval()
with torch.no_grad():
    test_predictions = loaded_model(X_test_tensor, X_test_2).squeeze()
    train_predictions = loaded_model(X_train_tensor, X_train_2).squeeze()
    validation_predictions = loaded_model(X_val_tensor, X_val_2).squeeze()

    test_predictions_sm = nn.Softmax(dim=-1)(test_predictions)
    train_predictions_sm = nn.Softmax(dim=1)(train_predictions)
    validation_predictions_sm = nn.Softmax(dim=1)(validation_predictions)

    test_predictions_labeled = torch.argmax(test_predictions_sm, dim=-1)
    train_predictions_labeled = torch.argmax(train_predictions_sm, dim=-1)
    validation_predictions_labeled = torch.argmax(validation_predictions_sm, dim=-1)

    y_test_tensor_labeled = torch.argmax(y_test_tensor, dim=-1)
    y_train_tensor_labeled = torch.argmax(y_train_tensor, dim=-1)
    y_val_tensor_labeled = torch.argmax(y_val_tensor, dim=-1)

    accuracy_test = (test_predictions_labeled == y_test_tensor_labeled).float().mean()
    accuracy_train = (train_predictions_labeled == y_train_tensor_labeled).float().mean()
    accuracy_val = (validation_predictions_labeled == y_val_tensor_labeled).float().mean()

    f1_test, recall_test, precision_test = f1_loss(y_test_tensor, test_predictions_sm)
    f1_train, recall_train, precision_train = f1_loss(y_train_tensor, train_predictions_sm)
    f1_val, recall_val, precision_val = f1_loss(y_val_tensor, validation_predictions_sm)

    print(f'Accuracy Test: {accuracy_test.item():.4f}, F1 Test: {f1_test:.4f}, Recall Test: {recall_test:.4f}, Precision Test: {precision_test:.4f}')
    print(f'Accuracy Train: {accuracy_train.item():.4f}, F1 Train: {f1_train:.4f}, Recall Train: {recall_train:.4f}, Precision Train: {precision_train:.4f}')
    print(f'Accuracy Val: {accuracy_val.item():.4f}, F1 Val: {f1_val:.4f}, Recall Val: {recall_val:.4f}, Precision Val: {precision_val:.4f}')

print('done')
