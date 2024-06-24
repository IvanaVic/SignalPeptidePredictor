import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")


class SignalPeptidePredictorCNN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SignalPeptidePredictorCNN2, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4*hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim[0] * output_dim[1])
        self.fc_org = nn.Linear(3, hidden_dim)
        self.fc_seq = nn.Linear(hidden_dim*4*73, hidden_dim*3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, y):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc_seq(x))
        y = self.relu(self.fc_org(y))
        xy = torch.concat([x, y], dim=-1)
        xy = self.relu(self.fc1(xy))
        xy = self.dropout(xy)
        out = self.fc2(xy)
        return out


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
    peptides = 'STLPIMO'
    encoded_sequences = encode_with_key(sequences, amino_acids)
    encoded_peptides = simple_encoding(outputs)
    encided_organisms = encode_organisms(organisms)

    result = [[s, p, o] for s, p, o in zip(encoded_sequences, encoded_peptides, encided_organisms)]

    return result


def balance_dataset(input_dataset):
    counter_data = 0
    for _, labels in labeled_train_set:
        if len(np.unique([*labels])) == 1 and np.unique([*labels])[0] == 'I':
            counter_data += 1

    percent = 0.1
    data_samples = counter_data - int((percent*len(input_dataset) - counter_data) / (percent - 1))
    if data_samples < 0:
        raise ValueError

    output = []
    counter = 0
    for data, labels in input_dataset:
        if len(np.unique([*labels])) == 1 and np.unique([*labels])[0] == 'I' and counter < data_samples:
            output.append([data, labels])
            counter += 1
        elif len(np.unique([*labels])) == 1 and np.unique([*labels])[0] == 'I' and counter >= data_samples:
            continue
        else:
            output.append([data, labels])
    return output


def modified_cross_entropy(y_predicted, y_true):
    entropy = -torch.sum(y_true.reshape((y_true.shape[0], 70, 7)) * torch.log(y_predicted), dim=-1)
    return torch.mean(torch.tensor(entropy))


def f1_loss(y_true, y_pred, beta=1) -> np.float32:

    tp = (y_true * y_pred).sum(dim=-1)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=-1)
    fp = ((1 - y_true) * y_pred).sum(dim=-1)
    fn = (y_true * (1 - y_pred)).sum(dim=-1)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + epsilon)

    return f1.mean()


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_predicted, y_true):
        entropy = -torch.sum(y_true.reshape((y_true.shape[0], 70, 7)) * torch.log(y_predicted), dim=-1)
        return torch.tensor(entropy).mean()


def train_model(model, saving_path, optimizer, loss_function, num_epochs, X_train, X_train_2, y_train, X_val, X_val_2, y_val):
    # Training loop
    min_val_loss = np.inf
    metrics_train = {'loss': [], 'accuracy': [], 'F1': []}
    metrics_val = {'loss': [], 'accuracy': [], 'F1': []}
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train, X_train_2).squeeze()
        loss = loss_function(outputs, y_train)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val, X_val_2).squeeze()
            loss_val = loss_function(outputs_val, y_val)

            outputs_sm = nn.Softmax(dim=-1)(outputs)
            outputs_val_sm = nn.Softmax(dim=-1)(outputs_val)

            output_labeled = torch.argmax(outputs_sm, dim=-1)
            output_val_labeled = torch.argmax(outputs_val_sm, dim=-1)
            y_train_labeled = torch.argmax(y_train, dim=-1)
            y_val_labeled = torch.argmax(y_val, dim=-1)

            accuracy_metric = (output_labeled == y_train_labeled).float().mean()
            accuracy_metric_val = (output_val_labeled == y_val_labeled).float().mean()

            f1 = f1_loss(y_train, outputs_sm)
            f1_val = f1_loss(y_val, outputs_val_sm)

            if loss_val.item() < min_val_loss:
                min_val_loss = loss_val.item()
                torch.save(model, os.path.join(saving_path, f'epoch_{epoch + 1}_{loss_val.item()}.pth'))

            metrics_train['loss'].append(loss.item())
            metrics_train['accuracy'].append(accuracy_metric.item())
            metrics_train['F1'].append(f1.item())

            metrics_val['loss'].append(loss_val.item())
            metrics_val['accuracy'].append(accuracy_metric_val.item())
            metrics_val['F1'].append(f1_val.item())

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy_metric.item():.4f}, F1: {f1:.4f} '
                  f'Validation loss: {loss_val.item():.4f}, Validation accuracy: {accuracy_metric_val.item():.4f}, Validation F1: {f1_val:.4f}')
    model.train(False)
    return metrics_train, metrics_val

def labeling(dataset):
    return [[el[1], el[2], el[0].split('|')[1]] for el in dataset]


fasta_file = r"./drive/MyDrive/dataset/train_set.fasta"

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

indices = torch.randperm(X_train_tensor.size()[0])
X_train_tensor = X_train_tensor[indices]
X_train_2 = X_train_2[indices]
y_train_tensor = y_train_tensor[indices]

X_train_tensor = X_train_tensor.cuda()
X_train_2 = X_train_2.cuda()
y_train_tensor = y_train_tensor.cuda()

X_val_tensor = X_val_tensor.cuda()
X_val_2 = X_val_2.cuda()
y_val_tensor = y_val_tensor.cuda()

X_test_tensor = X_test_tensor.cuda()
X_test_2 = X_test_2.cuda()
y_test_tensor = y_test_tensor.cuda()

input_dim = 20
hidden_dim = 8
output_dim = (3, 1)
learning_rate = 0.0001
num_epochs = 1000
saving_path = os.path.join(os.path.abspath(os.getcwd()), r'saved_models')
plot_saving_path = os.path.join(os.path.abspath(os.getcwd()), r'saved_plots')
if os.path.exists(saving_path):
  shutil.rmtree(saving_path)
  os.mkdir(saving_path)
else:
  os.mkdir(saving_path)

signal_peptide_model = SignalPeptidePredictorCNN2(input_dim, hidden_dim, output_dim)
signal_peptide_model.to('cuda')

criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimization_algorithm = optim.Adam(signal_peptide_model.parameters(), lr=learning_rate)

train_history, validation_history = train_model(signal_peptide_model, saving_path, optimization_algorithm, criterion, num_epochs, X_train_tensor, X_train_2, y_train_tensor, X_val_tensor, X_val_2,  y_val_tensor)

signal_peptide_model.eval()
with torch.no_grad():
    predictions = signal_peptide_model(X_test_tensor, X_test_2).squeeze()

    predictions_sm = nn.Softmax(dim=-1)(predictions)
    predictions_labeled = torch.argmax(predictions_sm, dim=-1)
    y_test_tensor_labeled = torch.argmax(y_test_tensor, dim=-1)
    accuracy = (predictions_labeled == y_test_tensor_labeled).float().mean()
    f1_test = f1_loss(y_test_tensor, predictions_sm)
    print(f'Accuracy: {accuracy.item():.4f}, F1: {f1_test:.4f}')

# Save plots
plt.plot(train_history['loss'], label='Train loss')
plt.plot(validation_history['loss'], label='Validation loss')
plt.legend()
plt.savefig(os.path.join(plot_saving_path, 'loss_plot.png'), dpi=300)
plt.close()

plt.plot(train_history['accuracy'], label='Train accuracy')
plt.plot(validation_history['accuracy'], label='Validation accuracy')
plt.legend()
plt.savefig(os.path.join(plot_saving_path, 'accuracy_plot.png'), dpi=300)
plt.close()

plt.plot(train_history['F1'], label='Train F1 score')
plt.plot(validation_history['F1'], label='Validation F1 score')
plt.legend()
plt.savefig(os.path.join(plot_saving_path, 'f1_plot.png'), dpi=300)
plt.close()
