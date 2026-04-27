from model.dataset import ColorizationDataset
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    # Initialize the dataset
    dataset = ColorizationDataset('data/raw', img_size=256, augment=True)
    
    # Use a batch size of 4 to check for size consistency issues (stacking)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # num_workers=0 for Windows compatibility

    try:
        # Get one batch
        L, ab = next(iter(loader))
        
        print("Batch loaded successfully!")
        print("L  shape:", L.shape)    # Expected: [4, 1, 256, 266] -> wait, [4, 1, 256, 256]
        print("ab shape:", ab.shape)   # Expected: [4, 2, 256, 256]
        print("L  range:", L.min().item(), "->", L.max().item())
        print("ab range:", ab.min().item(), "->", ab.max().item())
        
        if L.shape[2:] == (256, 256) and ab.shape[2:] == (256, 256):
            print("Dataset OK: All images are correctly sized.")
        else:
            print(f"Warning: Unexpected shape {L.shape}")
            
    except Exception as e:
        print(f"Error during dataset testing: {e}")