import os
import random


def format_image_path(series_id, img_id, series_padding=6, img_padding=2):
    """
    Format image path with zero-padded series_id and image_id, separated by dash.
    
    Args:
        series_id: Series ID (will be zero-padded)
        img_id: Image ID (will be zero-padded)
        series_padding: Number of digits for series_id padding (default 6)
        img_padding: Number of digits for image_id padding (default 2)
    
    Returns:
        Formatted path like '000261-07.jpg'
    """
    padded_series = str(series_id).zfill(series_padding)
    padded_img = str(img_id).zfill(img_padding)
    return f"{padded_series}-{padded_img}.JPG"


def make_shuffle_path(seed=None):
    """
    Reads train and validation pairlist files and returns paths for image pairs.

    Args:
        seed: Optional random seed for reproducible shuffling (default: None)

    Pairlist format: series_id img1_id img2_id score winner loser
    Returns: train_pathA, train_pathB, train_result, val_pathA, val_pathB, val_result
    """
    if seed is not None:
        random.seed(seed)

    data_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(data_dir, 'train_pairlist.txt')
    val_file = os.path.join(data_dir, 'val_pairlist.txt')

    train_pathA = []
    train_pathB = []
    train_result = []

    val_pathA = []
    val_pathB = []
    val_result = []

    # Read training pairs
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                series_id = parts[0]
                img1_id = parts[1]
                img2_id = parts[2]
                # score = parts[3]
                winner = parts[4]
                # loser = parts[5]

                # Create image paths: padded_series_id-padded_img_id.jpg (e.g., 000261-07.jpg)
                pathA = format_image_path(series_id, img1_id)
                pathB = format_image_path(series_id, img2_id)

                # Result: 0 if img1 wins, 1 if img2 wins
                if winner == img1_id:
                    result = 0
                else:
                    result = 1

                train_pathA.append(pathA)
                train_pathB.append(pathB)
                train_result.append(result)

    # Read validation pairs
    with open(val_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                series_id = parts[0]
                img1_id = parts[1]
                img2_id = parts[2]
                # score = parts[3]
                winner = parts[4]
                # loser = parts[5]

                # Create image paths: padded_series_id-padded_img_id.jpg (e.g., 000261-07.jpg)
                pathA = format_image_path(series_id, img1_id)
                pathB = format_image_path(series_id, img2_id)

                if winner == img1_id:
                    result = 0
                else:
                    result = 1

                val_pathA.append(pathA)
                val_pathB.append(pathB)
                val_result.append(result)

    # Shuffle training data
    combined = list(zip(train_pathA, train_pathB, train_result))
    random.shuffle(combined)
    train_pathA, train_pathB, train_result = zip(*combined)

    print(f"Loaded {len(train_pathA)} training pairs and {len(val_pathA)} validation pairs")

    return list(train_pathA), list(train_pathB), list(train_result), list(val_pathA), list(val_pathB), list(val_result)


if __name__ == "__main__":
    # Test the function
    train_A, train_B, train_res, val_A, val_B, val_res = make_shuffle_path()
    print(f"\nFirst 3 training examples:")
    for i in range(min(3, len(train_A))):
        print(f"  {train_A[i]} vs {train_B[i]} -> winner: {'A' if train_res[i] == 0 else 'B'}")
