import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn import metrics
from matplotlib import pyplot as plt


torch.manual_seed(60)

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

class Train:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.HCCD = HCCD().to(self.device)
        self.training_loss = []
        self.validation_loss = []
        # self.optimizer = optim.Adam(self.HCCD.parameters(), lr =  1e-3, betas = (0.9, 0.999)) # bert
        self.optimizer = optim.Adam(self.HCCD.parameters(), lr =  1e-5, betas = (0.9, 0.999)) # gpt

    def train_(self, epochs):
        loss_criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        print(">_ training")
        for epoch in range(epochs):
            self.HCCD.train()
            for idx, batch in tqdm(enumerate(loader_train, 0)):
                embeddings, labels = batch
                self.optimizer.zero_grad()
                outputs = self.HCCD(embeddings.to(self.device))
                train_loss = loss_criterion(outputs, labels.to(self.device))
                train_loss.backward()
                self.optimizer.step()
                # if idx % 8 == 0: 
                self.training_loss.append(train_loss.item())
                print(train_loss.item())
                
            self.HCCD.eval()
            for idx, batch in tqdm(enumerate(loader_val, 0)):
                embeddings, labels = batch
                outputs = self.HCCD(embeddings.to(self.device))
                valid_loss = loss_criterion(outputs, labels.to(self.device))
                self.validation_loss.append(valid_loss.item())
                # print(valid_loss.item())
                
            print(f"Epoch: {epoch}")
        print(f"Finshed training. \nTime to train: {time.time() - start_time} seconds")
        plt.plot(self.training_loss, label = "Training Loss")
        plt.plot(self.validation_loss,label = "Validation Loss")
        plt.legend()
        plt.show

    def test_(self):
        start_time = time.time()
        labels_metric = []
        prediction_metric = []
        self.HCCD.eval()
        start_time = time.time()
        with torch.no_grad():
            for embeddings, labels in loader_test:
                outputs = self.HCCD(embeddings.float().to(self.device))
                labels = labels.to(self.device)
                _, predictions = torch.max(outputs.data, 1)            
                prediction_metric.append(predictions.tolist())
                labels_metric.append(labels.tolist())
        print(f"Inference Time: {time.time() - start_time} sec")
        labels_metric_flat  = [i for sublist in labels_metric for i in sublist]
        prediction_metric_flat  = [i for sublist in prediction_metric for i in sublist]
        f1 = metrics.f1_score(labels_metric_flat, prediction_metric_flat, average = None)
        precision = metrics.precision_score(labels_metric_flat, prediction_metric_flat, average = None)
        recall = metrics.recall_score(labels_metric_flat, prediction_metric_flat, average = None)
        math_corr = metrics.matthews_corrcoef(labels_metric_flat, prediction_metric_flat)
        
        return np.mean(f1), np.mean(precision), np.mean(recall), np.mean(math_corr)
        
train_hccd = Train()
train_hccd.train_(10)
train_hccd.test_()