import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from tqdm import tqdm


class Manager():

    def __init__(self, device, net, criterion, train_dataloader, val_dataloader, test_dataloader):
        self.device = device
        self.net = net
        self.best_model = self.net
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def do_batch(self, optimizer, batch, labels):
        """Runs model for one batch."""
        batch = batch.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()  # Zero-ing the gradients
        outputs = self.net(batch)

        loss = self.criterion(outputs, labels)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)
        running_corrects = torch.sum(
            preds == labels.data).data.item()  # number corrects

        loss.backward()  # backward pass: computes gradients
        optimizer.step()  # update weights based on accumulated gradients

        return (loss, running_corrects)

    def do_epoch(self, optimizer, scheduler, current_epoch):
        """Trains model for one epoch."""

        self.net.train()  # Set network in training mode

        running_train_loss = 0
        running_corrects = 0
        total = 0
        batch_idx = 0
        for images, labels in tqdm(self.train_dataloader, desc='Epoch: %d ' % (current_epoch)):

            loss, corrects = self.do_batch(optimizer, images, labels)

            running_train_loss += loss.item()
            running_corrects += corrects
            total += labels.size(0)
            batch_idx += 1

        scheduler.step()

        # Average Scores
        train_loss = running_train_loss / batch_idx  # average over all batches
        train_accuracy = running_corrects / \
            float(total)  # average over all samples

        print('\nTrain Loss {}, Train Accuracy {}'\
              .format(train_loss, train_accuracy))

        return (train_loss, train_accuracy)

    def validate(self):

        self.net.train(False)

        running_val_loss = 0
        running_corrects = 0
        total = 0
        batch_idx = 0
        for images, labels in self.val_dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)
            total += labels.size(0)

            # Forward Pass
            outputs = self.net(images)
            loss = self.criterion(outputs, labels)
            running_val_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

            batch_idx += 1

        # Calcuate Scores
        val_loss = running_val_loss / batch_idx
        val_accuracy = running_corrects / float(total)

        return (val_loss, val_accuracy)

    def train(self, optimizer, scheduler, num_epochs, validation = True):

        self.net.to(self.device)
        cudnn.benchmark  # Calling this optimizes runtime

        train_loss_history = {}
        train_accuracy_history = {}
        val_loss_history = {}
        val_accuracy_history = {}

        for epoch in range(num_epochs):

            train_loss, train_accuracy = self.do_epoch(
                optimizer, scheduler, epoch+1)  # Epochs start counting form 1
            # Validate at each epoch
            if validation:
                
                self.best_loss = 1e-2
                self.best_epoch = 0
                
                val_loss, val_accuracy = self.validate()

                train_loss_history[epoch+1] = train_loss
                train_accuracy_history[epoch+1] = train_accuracy
                val_loss_history[epoch+1] = val_loss
                val_accuracy_history[epoch+1] = val_accuracy
            
                # Best validation model
                if val_accuracy < self.best_loss:
                    self.best_loss = val_accuracy
                    self.best_model = self.net
                    self.best_epoch = epoch
                    print('Best model updated\n')
                
            else:
                train_loss_history[epoch+1] = train_loss
                train_accuracy_history[epoch+1] = train_accuracy

        if validation:
            return (train_loss_history, train_accuracy_history,
                val_loss_history, val_accuracy_history)
        
        return (train_loss_history, train_accuracy_history)

    def test(self):

        self.net.train(False)  # Set Network to evaluation mode

        running_corrects = 0
        total = 0
        for images, labels in tqdm(self.test_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            total += labels.size(0)

            # Forward Pass
            outputs = self.net(images)
            loss = self.criterion(outputs, labels)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy
        accuracy = running_corrects / \
            float(total)  

        print('\nTest Accuracy: {}'.format(accuracy))
