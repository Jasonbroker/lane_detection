import json
import os
from PIL import Image
from torch.utils.data import Dataset

class TUSimpleQwenVLDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Load all label files
        label_files = [
            # 'label_data_0313.json',
            # 'label_data_0601.json',
            'label_data_0531.json'
        ] if split == 'train' else ['test_tasks_0627.json']
        
        for label_file in label_files:
            with open(os.path.join(root_dir, label_file), 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    img_path = os.path.join(root_dir, data['raw_file'])
                    lanes = data['lanes']
                    h_samples = data['h_samples']
                    
                    # Convert to list of (x, y) dicts
                    lane_coords = []
                    for lane in lanes:
                        coords = []
                        for x, y in zip(lane, h_samples):
                            if x >= 0:  # -2 or -1 means invalid
                                coords.append({"x": int(x), "y": int(y)})
                        if coords:
                            lane_coords.append(coords)
                    
                    self.samples.append({
                        'image_path': img_path,
                        'lanes': lane_coords
                    })
                    print(f"Loaded {img_path} with lanes: {lane_coords}")
                    if len(self.samples) >= 3 and split == 'train':  # Limit for quick testing
                        break
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        prompt = "Detect all lane lines in the image. Output each lane as a list of (x, y) coordinates in JSON format.\n<image>"
        output_json = json.dumps({"lanes": sample['lanes']}, indent=2)
        
        return {
            'image': image,
            'prompt': prompt,
            'output': output_json
        }