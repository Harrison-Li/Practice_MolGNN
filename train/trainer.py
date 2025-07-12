import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import numpy as np
from model.gnn import MolGNN
from .dataset import MoleculeDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, max_epochs, batch_size, learning_rate, criterion, optimizer, ckpt_path, device, early_stopping_threshold=None):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.optimizer = optimizer
        self.ckpt_path = ckpt_path
        self.device = device
        self.early_stopping_threshold = early_stopping_threshold

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)
            #self.model = torch.nn.DataParallel(self.model).cuda()

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
        
    def train(self, wandb):
        model, config = self.model, self.config
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        best_loss = float('inf')

        def run_epoch(is_train, epoch, data_loader):
            model.train() if is_train else model.eval()
            
            losses = []
            pbar = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
            
            context = torch.enable_grad() if is_train else torch.no_grad()
            with context:
                for it, batch in pbar:
                    x_fea, x_adj, y = batch
                    x_fea, x_adj, y = x_fea.to(self.device), x_adj.to(self.device), y.to(self.device)
                    # print(x_fea.size(), x_adj.size(), y.size())
                    if is_train:
                        optimizer.zero_grad()

                    output = model(x_fea, x_adj)
                    # Ensure output and target have the same shape for the loss function
                    loss = config.criterion(output.squeeze(), y)
                    
                    if is_train:
                        loss.backward()
                        optimizer.step()
                    
                    losses.append(loss.item())
                    pbar.set_description(f"Epoch {epoch + 1} ({'Train' if is_train else 'Valid'}), Loss: {np.mean(losses):.4f}")
            
            return np.mean(losses)

        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False)

        for epoch in range(config.max_epochs):
            train_loss = run_epoch(True, epoch, train_loader)
            test_loss = run_epoch(False, epoch, test_loader)
            
            wandb.log({'epoch_valid_loss': test_loss, 'epoch_train_loss': train_loss, 'epoch': epoch + 1})
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Valid Loss = {test_loss:.4f}")
            
            # Early stopping condition
            if test_loss < best_loss:
                best_loss = test_loss
                self.save_checkpoint()
                logger.info(f"Best model saved at epoch {epoch + 1} with loss {test_loss:.4f}")

            if config.early_stopping_threshold and test_loss > config.early_stopping_threshold:
                logger.info(f"Validation loss {test_loss:.4f} exceeded threshold. Stopping training.")
                break

