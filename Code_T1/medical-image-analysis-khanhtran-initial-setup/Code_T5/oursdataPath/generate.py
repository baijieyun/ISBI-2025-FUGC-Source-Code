import os

def generate_file_list(data_dir, output_file):
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.relpath(os.path.join(root, file), start=os.path.dirname(data_dir))
                f.write(f"{file_path}\n")

if __name__ == "__main__":
    base_dir = "./oursdataPath"
    
    labeled_data_dir = os.path.join(base_dir, "train_data")
    unlabeled_data_dir = os.path.join(base_dir, "unlabeled_data")
    val_data_dir = os.path.join(base_dir, "val_data")
    
    generate_file_list(labeled_data_dir, os.path.join(base_dir, "train.txt"))
    generate_file_list(unlabeled_data_dir, os.path.join(base_dir, "unlabeled.txt"))
    generate_file_list(val_data_dir, os.path.join(base_dir, "val.txt"))