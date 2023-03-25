import os
import torch
from torch.utils.data import DataLoader
from src.models import MyModel
from data_collection.data_loader import MapDataLoader

def train(data_dir, batch_size, num_epochs):
    # Load training and validation sets
    data_loader = MapDataLoader(data_dir, batch_size)
    train_set = data_loader.prepare_data_loader()
    val_set = data_loader.prepare_data_loader()

    # Create model and optimizer
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(num_epochs):
        for i, (images, masks) in enumerate(train_set):
            # Forward pass
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 100 batches
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(train_set), loss.item()))

        # Evaluate on validation set after each epoch
        with torch.no_grad():
            total = 0
            correct = 0
            for images, masks in val_set:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += masks.size(0) * masks.size(1) * masks.size(2)
                correct += (predicted == masks).sum().item()

            accuracy = correct / total
            print('Epoch [{}/{}], Validation Accuracy: {:.4f}'
                  .format(epoch+1, num_epochs, accuracy))

    # Save the trained model
    model_path = os.path.join(data_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
