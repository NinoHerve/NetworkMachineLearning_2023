import sys
sys.path.append('../')

import unittest
from pathlib import Path
from utils.models import CNN 
import torch
import torch.nn as nn


class TestCNN(unittest.TestCase):

    def test_training(self):

        X = torch.rand((32,128,625))
        y = torch.randint(0,2,(32,)).float()

        model  = CNN(n_channels=128, n_kernels=8)
        before = []
        for p in model.parameters():
            before.append(p.clone())
        
        loss_fn   = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train one step
        model.train()
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()   
        optimizer.step()

        after = []
        for p in model.parameters():
            after.append(p.clone())

        # Check if all weights have changed
        changed = []
        for b, a in zip(before, after):
            changed.append((b != a).all())

        self.assertTrue(changed)

        return
    
    def test_loss(self):

        X = torch.rand((32,128,625))
        y = torch.randint(0,2,(32,)).float()

        model  = CNN(n_channels=128, n_kernels=8)
        before = []
        for p in model.parameters():
            before.append(p.clone())
        
        loss_fn   = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train one step
        model.train()
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        self.assertNotEqual(loss, 0)

        return


if __name__ == '__main__':
    unittest.main()
