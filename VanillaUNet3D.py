# %%
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    Lambdad,
    Activations,
    ScaleIntensityRange,
    Lambda
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, SSIMMetric
from monai.losses import DiceLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import nrrd

print_config()

# %%
# Convert train and validation images into lists with locations
print("Model initialization")
train_dir = "./data/Train/Train"
val_dir = "./data/Val/Nasal25"

train_nrrd_files = sorted([os.path.join(train_dir, f) for f in os.listdir(
    train_dir) if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")])
train_seg_nrrd_files = sorted([os.path.join(train_dir, f)
                              for f in os.listdir(train_dir) if f.endswith(".seg.nrrd")])

val_nrrd_files = sorted([os.path.join(val_dir, f) for f in os.listdir(
    val_dir) if f.endswith(".nrrd") and not f.endswith(".seg.nrrd")])
val_seg_nrrd_files = sorted([os.path.join(val_dir, f)
                            for f in os.listdir(val_dir) if f.endswith(".seg.nrrd")])

train_datalist = [{"image": img, "label": lbl}
                  for img, lbl in zip(train_nrrd_files, train_seg_nrrd_files)]
validation_datalist = [{"image": img, "label": lbl}
                       for img, lbl in zip(val_nrrd_files, val_seg_nrrd_files)]
print(f" Trian datalist setup {train_datalist[0]}")

# %%

from monai.transforms import Transposed
import numpy as np


def binarize(label, threshold=0.1):
    binary_mask = (label > threshold)
    binary_mask[binary_mask > 0] = 1  # Set all non-zero pixels to 1
    return binary_mask


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader="NrrdReader", image_only=True),
        #Transposed(keys=['image', 'label'], indices=[2, 1, 0]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(("label"), binarize),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 32),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"],
                   reader="NrrdReader", image_only=True),
        #Transposed(keys=['image', 'label'], indices=[2, 1, 0]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(("label"), binarize),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
)


# %%
train_ds = CacheDataset(
    data=train_datalist, transform=train_transforms, cache_rate=1.0, num_workers=4)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

val_ds = CacheDataset(
    data=validation_datalist, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %%
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# %%
pre_trained_state_dict = torch.load(
    "./models/VanillaUNet3D/Unet3D_1500EP_25Samples_32patch.pth")

# %%
model_state_dict = model.state_dict()

#update only if the shape matches
# Update the state dict to load only the compatible keys
for name, param in pre_trained_state_dict.items():
    if name in model_state_dict and param.shape == model_state_dict[name].shape:
        model_state_dict[name] = param

# %%
# Load the modified state dict into the UNet model
model.load_state_dict(model_state_dict)

# Set the last 5 layers to be trainable
total_layers = len(list(model.parameters()))
print(f"total_layers : {total_layers}")
for idx, (name, param) in enumerate(model.named_parameters()):
    if idx >= total_layers - 92:
        #print(f"Number of trainable layers {idx}")
        param.requires_grad = True
    else:
        print(f"Number of frozen layers {idx}")
        param.requires_grad = False

# %%
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# %%
out_root = "./logs/VanillaUNet3D"
max_epochs = 1500
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        print("######### input shape", inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (128, 128, 32)
                sw_batch_size = 2
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            #metric = 1-loss.item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    out_root, "Unet3D_1500EP_25Samples_32patch" + ".pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

# %%



