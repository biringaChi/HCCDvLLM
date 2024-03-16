import time
import torch
import argparse
import numpy as np
from repr import Repr
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(60)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, metavar = "", required = True, help = "Model type")
parser.add_argument('--epochs', type = int, metavar = "", required = True, help = "Number of training cycles")
parser.add_argument('--lr', type = float, metavar = "", required = True, help = "Adam optimizer learning rate")
args = parser.parse_args()


class HCCD(nn.Module):
    def __init__(self):
        super(HCCD, self).__init__()
        self.nclasses = 8
        self.dprob = 0.5
            
        self.model = nn.Sequential(
            nn.Linear(768, 512),
            nn.LeakyReLU(), 
            nn.Dropout(self.dprob),
            nn.Linear(512, 256), 
            nn.LeakyReLU(),
            nn.Dropout(self.dprob),
            nn.Linear(256, self.nclasses)
        )
        
    def forward(self, x):
        return self.model(x)

class Detect(Repr):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.HCCD = HCCD().to(self.device)
        self.training_loss = []
        self.validation_loss = []
        self.optimizer = optim.Adam(self.HCCD.parameters(), lr =  args.lr) # M1 (10 epochs, 1e-3), M2 (4 epochs, lr=1e-4)

    def feature_loader(self):
        if args.model == "M1":
            x = self._stream_features("bert", "bert-base-uncased")
        elif args.model == "M2":
            x = self._stream_features("gpt2", "gpt2")
        X_train, Xs, y_train, ys = train_test_split(np.array(x), np.array(self._get_labels()), train_size = 0.8, random_state = 1)
        X_val, X_test, y_val, y_test = train_test_split(Xs, ys, test_size = 0.5, random_state = 1)
        train_features = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(np.array(y_train))), shuffle = True, batch_size = 32)
        val_features = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(np.array(y_val))), shuffle = True, batch_size = 32)
        test_features = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(np.array(y_test))), shuffle = True, batch_size = 32)
        return train_features, val_features, test_features

    def detect(self):
        train_features, val_features, test_features = self.feature_loader()
        loss_criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        print(">_ training")
        for epoch in range(args.epochs):
            self.HCCD.train()
            for idx, batch in tqdm(enumerate(train_features, 0)):
                embeddings, labels = batch
                self.optimizer.zero_grad()
                outputs = self.HCCD(embeddings.to(self.device))
                train_loss = loss_criterion(outputs, labels.to(self.device))
                train_loss.backward()
                self.optimizer.step()
                self.training_loss.append(train_loss.item())
                print(train_loss.item())
                
            self.HCCD.eval()
            for idx, batch in tqdm(enumerate(val_features, 0)):
                embeddings, labels = batch
                outputs = self.HCCD(embeddings.to(self.device))
                valid_loss = loss_criterion(outputs, labels.to(self.device))
                self.validation_loss.append(valid_loss.item())
                print(valid_loss.item())
                
            print(f"Epoch: {epoch}")
        print(f"Finshed training. \nTime to train: {time.time() - start_time} seconds")

        start_time = time.time()
        labels_metric = []
        prediction_metric = []
        self.HCCD.eval()
        start_time = time.time()
        with torch.no_grad():
            for embeddings, labels in test_features:
                outputs = self.HCCD(embeddings.float().to(self.device))
                labels = labels.to(self.device)
                _, predictions = torch.max(outputs.data, 1)            
                prediction_metric.append(predictions.tolist())
                labels_metric.append(labels.tolist())
        print(f"Inference Time: {time.time() - start_time} sec")

        actual  = [i for sublist in labels_metric for i in sublist]
        preds  = [i for sublist in prediction_metric for i in sublist]

        accuracy = metrics.accuracy_score(actual, preds)
        f1 = metrics.f1_score(actual, preds, average = None)
        precision = metrics.precision_score(actual, preds, average = None)
        recall = metrics.recall_score(actual, preds, average = None)
        math_corr = metrics.matthews_corrcoef(actual, preds)

        return {
            "accuracy" : np.mean(accuracy),
            "f1" : np.mean(f1), 
            "precision" : np.mean(precision), 
            "recall" : np.mean(recall), 
            "mcc" : np.mean(math_corr)
        }

if __name__ == "__main__":
   print(Detect().detect())