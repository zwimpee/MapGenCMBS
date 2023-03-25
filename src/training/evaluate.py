import torch
import argparse
from src.models import MyModel
from data_collection.data_loader import MapData

def evaluate(model_path, data_dir):
    # Load the model
    model = torch.load(model_path)
    model.eval()

    # Load the test set
    test_data = MapData(os.path.join(data_dir, "processed", "test", "images"),
                        os.path.join(data_dir, "processed", "test", "masks"))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Evaluate the model
    total_correct = 0
    total_pixels = 0
    with torch.no_grad():
        for images, masks in test_loader:
            # Forward pass
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            # Compute accuracy
            correct = (predictions == masks).sum().item()
            total_correct += correct
            total_pixels += images.numel()

    accuracy = total_correct / total_pixels
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a segmentation model.")
    parser.add_argument("model_path", type=str, help="Path to the saved model.")
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory.")
    args = parser.parse_args()

    evaluate(args.model_path, args.data_dir)
