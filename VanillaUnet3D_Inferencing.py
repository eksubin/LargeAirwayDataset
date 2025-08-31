# %%
# inference.py
# %%
import torch
import os
import nrrd
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, AsDiscrete
from monai.networks.nets import UNet

# %%
# Define the paths
model_path = "./models/VanillaUNet3D/Unet3D_1500EP_25Samples_32patch.pth"
input_dir = "./data/Inference"  # Directory containing the input NRRD images
output_dir = "./data/Results"    # Directory to save the results


# %%
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# %%
# Load the model
from monai.networks.layers import Norm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# Load the trained model weights
model.load_state_dict(torch.load(model_path))
model.eval()


# %%
inference_transforms = Compose(
    [
        LoadImaged(keys=["image"], reader="NrrdReader", image_only=True),
        EnsureChannelFirstd(keys=["image"]),
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
# Perform inference
for filename in os.listdir(input_dir):
    if filename.endswith(".nrrd"):
        input_path = os.path.join(input_dir, filename)
        data_dict = {"image": input_path}
        
        # Apply transforms
        transformed = inference_transforms(data_dict)
        input_tensor = transformed["image"].unsqueeze(0).to(device)  # Adding batch dimension
        
        with torch.no_grad():
            output = model(input_tensor)
            output_probs = torch.softmax(output, dim=1)
            output_segmentation = torch.argmax(output_probs, dim=1).cpu().numpy()  # Get predicted segmentation
        
        # Save the segmentation result
        output_filename = os.path.join(output_dir, filename)
        #nrrd.write(output_filename.replace('.nrrd', '_seg.nrrd'), output_segmentation)

        print(f"Processed {filename}, saved segmentation as {output_filename.replace('.nrrd', '_seg.nrrd')}")


# %%



