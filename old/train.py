

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from torch.utils.data import DataLoader
from data_set import TUSimpleQwenVLDataset
import torch
print(torch.__version__)

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/media/zzcc/file/aigc/llm/models/Qwen/Qwen2___5-VL-7B-Instruct")
processor = Qwen2_5_VLProcessor.from_pretrained("/media/zzcc/file/aigc/llm/models/Qwen/Qwen2___5-VL-7B-Instruct")

from torchvision import transforms
# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0,1]
    # Add other transforms as needed (Resize, Normalize, etc.)
])
dataset = TUSimpleQwenVLDataset(root_dir="/media/zzcc/file/aigc/llm/datasets/TUSimple/train_set/", split='train', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop (simplified)
for batch in dataloader:
    inputs = processor(
        text=batch['prompt'],
        images=batch['image'],
        return_tensors="pt",
        padding=True
    )

    print(processor.tokenizer.special_tokens_map)
    
    labels = processor.tokenizer(
        batch['output'],
        return_tensors="pt",
        padding=True
    ).input_ids
    
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()