import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip,
    RandomResizedCrop, Resize, ToTensor,
)
from datasets import load_dataset
from transformers import ViTForImageClassification, AutoImageProcessor
import time
from transformers import ViTForImageClassification, ViTFeatureExtractor

import torch
import torch.nn as nn

class LayerwiseFGFourierFT(nn.Module):
    def __init__(
        self,
        n: int = 100,
        alpha: float = 300.0,
        d1: int = 4096,
        d2: int = 4096,
        base_layer: nn.Module = None,
        delta_W_init: torch.Tensor = None,
        top_k_ratio: float = 0.05
    ):
        super(LayerwiseFGFourierFT, self).__init__()

        # Definitions
        self.d1 = d1
        self.d2 = d2
        self.alpha = alpha
        self.base_layer = base_layer

        # delta_W_init (초기 가중치 변화량)을 FFT로 변환
        W_freq = torch.fft.fft2(delta_W_init)
        magnitude = torch.abs(W_freq)

        # 중요 주파수 위치 선정 (top-k 방식)
        total_freq = d1 * d2
        k = int(total_freq * top_k_ratio)
        topk_values, topk_indices = torch.topk(magnitude.view(-1), k)

        # Frequency mask 생성 (중요 주파수 위치는 1, 나머지는 0)
        self.mask = torch.zeros((d1, d2), dtype=torch.bool)
        self.mask.view(-1)[topk_indices] = True

        # 중요 주파수 위치의 초기값만 학습 파라미터로 설정
        self.c = nn.Parameter(W_freq[self.mask].clone().detach(), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # 스펙트럼 행렬 생성 및 중요 위치만 채워넣기
        F = torch.zeros(self.d1, self.d2, dtype=torch.complex64, device=x.device)
        F[self.mask] = self.c


        Delta_W = torch.fft.ifft2(F).real * self.alpha


        h = self.base_layer(x)
        h += torch.einsum('ijk,kl->ijl', x, Delta_W)

        return h

# Load model and preprocessing
model_checkpoint = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_checkpoint)
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose([
    RandomResizedCrop(image_processor.size["height"]),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])
val_transforms = Compose([
    Resize(image_processor.size["height"]),
    CenterCrop(image_processor.size["height"]),
    ToTensor(),
    normalize,
])



feature_extractor = ViTFeatureExtractor.from_pretrained(model_checkpoint)

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


train_ds = load_dataset("Donghyun99/FGVC-Aircraft", split="train")
val_ds = load_dataset("Donghyun99/FGVC-Aircraft", split="test")
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.tensor([example["label"] for example in batch])
    return {"pixel_values": pixel_values, "labels": labels}

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Replace query layers with Fourier layer
for i, layer in enumerate(model.vit.encoder.layer):
    original_linear = layer.attention.attention.query
    delta_W_init = original_linear.weight.data.clone() * 0.01

    layer.attention.attention.query = LayerwiseFGFourierFT(
        n=100,
        alpha=12,
        d1=original_linear.weight.size(0),
        d2=original_linear.weight.size(1),
        base_layer=original_linear,
        delta_W_init=delta_W_init,
        top_k_ratio=0.5
    )

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

epochs = 1000
epoch_train_acc, epoch_train_loss = [], []
epoch_val_acc, epoch_val_loss = [], []

start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

    # === Train ===
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    

    # 예시:
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Total parameters: {count_total_parameters(model):,}")
    #  90,106,600 
    for batch in train_loader:
        inputs = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.logits.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    epoch_train_loss.append(train_loss / len(train_loader))
    epoch_train_acc.append(100. * train_correct / train_total)

    # === Validation ===
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)

            val_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    epoch_val_loss.append(val_loss / len(val_loader))
    epoch_val_acc.append(100. * val_correct / val_total)

    print(f"[Epoch {epoch+1}/{epochs}] "
          f"Train Loss: {epoch_train_loss[-1]:.4f}, Train Acc: {epoch_train_acc[-1]:.2f}% | "
          f"Val Loss: {epoch_val_loss[-1]:.4f}, Val Acc: {epoch_val_acc[-1]:.2f}% | ")
    
    script=f"[Epoch {epoch+1}/{epochs}] Train Loss: {epoch_train_loss[-1]:.4f}, Train Acc: {epoch_train_acc[-1]:.2f}% | \
        Val Loss: {epoch_val_loss[-1]:.4f}, Val Acc: {epoch_val_acc[-1]:.2f}% "
    
    torch.save(model.state_dict(),"fgvc_new_model.pth")
    with open("fgvc_new_model.txt","a") as logger:
        logger.writelines(script)
        logger.writelines("\n")
        logger.writelines("\n")
         
