import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.optim as optim


class RNN(nn.Module):
    # TODO: IMPLEMENT THIS FUNCTION
    # Initialize the three layers in the RNN, self.embedding, self.rnn, and self.fc
    # Each one has a corresponding function in nn
    # embedding maps from input_dim->embedding_dim
    # rnn maps from embedding_dim->hidden_dim
    # fc maps from hidden_dim->output_dim
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        ## CHANGE THESE DEFINITIONS
        self.embedding = []
        self.rnn = []
        self.fc = []
    
    # TODO: IMPLEMENT THIS FUNCTION
    # x has dimensions [sentence length, batch size]
    # embedded has dimensions [sentence length, batch size, embedding_dim]
    # output has dimensions [sentence length, batch size, hidden_dim]
    # hidden has dimensions [1, batch size, hidden_dim]
    def forward(self, x):

        ## CHANGE THESE DEFINITIONS
        embedded = []
        output, hidden = []
        
        return self.fc(hidden.squeeze(0))

# Calculates Batch Accuracy (just as another metric)
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum()/len(correct)
    return acc

# TODO: IMPLEMENT THIS FUNCTION
# Get a set of predictions in the batch, then calculate the batch loss and accuracy, which
# will be used in the optimization procedure
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        ## CHANGE THESE THREE VARIABLES
        batch_predictions = []
        batch_loss = 0
        batch_acc = 0
        ## CHANGE THESE THREE VARIABLES

        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
        epoch_acc += batch_acc.item()
        
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

# TODO: IMPLEMENT THIS FUNCTION
# Evaluates Performance of Model across an epoch
# You need to get a set of predictions within a batch and use that to calculate the batch loss and accuracy
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            ## CHANGE THESE THREE VARIABLES
            batch_predictions = []
            batch_loss = 0
            batch_acc = 0
            ## CHANGE THESE THREE VARIABLES

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
        
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

# Do not modify main
def main():

    SEED = 477
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # Load and Split Dataset
    TEXT = data.Field(tokenize='moses')
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    TEXT.build_vocab(train_data, max_size=25000)
    LABEL.build_vocab(train_data)

    # Load Dataset Iterators and Batch Size
    BATCH_SIZE = 64

    device = torch.device('cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size=BATCH_SIZE,
        device=device)

    # Fix Dimensions of the Network
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    # Fix model, optimizer, and loss function
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # Run for 5 epochs and evaluate performance
    N_EPOCHS = 5

    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

if __name__ == "__main__": main()