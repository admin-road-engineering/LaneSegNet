#\!/usr/bin/env python3
import json
import hashlib
from pathlib import Path

def load_json_dataset(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def save_json_dataset(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def hash_based_split(sample_id, train_ratio=0.8, val_ratio=0.1):
    hash_obj = hashlib.md5(str(sample_id).encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    percentage = (hash_int % 100) / 100.0
    
    if percentage < train_ratio:
        return 'train'
    elif percentage < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'

def main():
    print("DATASET SPLIT FIXER")
    print("="*60)
    
    train_data = load_json_dataset('data/train_data.json')
    val_data = load_json_dataset('data/val_data.json') if Path('data/val_data.json').exists() else []
    test_data = load_json_dataset('data/test_data.json')
    
    print(f"Current - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    if len(val_data) > 100:
        print("Validation set already exists - no changes needed")
        return True
    
    all_samples = train_data + val_data + test_data
    new_train, new_val, new_test = [], [], []
    
    for sample in all_samples:
        sample_id = sample.get('id', sample.get('image_path', ''))
        split = hash_based_split(sample_id)
        
        if split == 'train':
            new_train.append(sample)
        elif split == 'val':
            new_val.append(sample)
        else:
            new_test.append(sample)
    
    print(f"New split - Train: {len(new_train)}, Val: {len(new_val)}, Test: {len(new_test)}")
    
    backup_dir = Path("data/backup_splits")
    backup_dir.mkdir(exist_ok=True)
    
    import shutil, datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for file in ['train_data.json', 'val_data.json', 'test_data.json']:
        src = Path(f"data/{file}")
        if src.exists():
            shutil.copy2(src, backup_dir / f"{file}.backup_{timestamp}")
    
    save_json_dataset(new_train, 'data/train_data.json')
    save_json_dataset(new_val, 'data/val_data.json') 
    save_json_dataset(new_test, 'data/test_data.json')
    
    print("SUCCESS: Dataset split fixed with proper validation set!")
    return True

if __name__ == "__main__":
    main()
