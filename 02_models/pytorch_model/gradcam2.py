import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model_scn2     import ExtendedSimpleCNN2D
import cv2

# Load a pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ExtendedSimpleCNN2D(3,8)
# print(model)
state_dictonary = torch.load("./03_runs/class_langage-3/best_checkpoint.pth", map_location = device)
model.load_state_dict(state_dictonary['model_state'])
model = model.to(device)

model.eval()
# Define a hook to capture gradients
gradients = []

def save_gradient(grad):
    gradients.append(grad)

# Select the target layer
target_layer = model.body[3].block[0].block_1[0]
# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((177, 177)),  # Redimensionner les images
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),         # Convertir en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess an image
image_path = "./01_data/test/R/30.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)

hook = target_layer.register_backward_hook(lambda module, grad_input, grad_output: save_gradient(grad_output[0]))

# Forward pass
output = model(input_tensor)
class_idx = output.argmax(dim=1).item()  # Predicted class

# Backward pass for the target class
model.zero_grad()
score = output[0, class_idx]
score.backward()

# # Hook gradients to the target layer
# for name, module in model.named_modules():
#     if module == target_layer:
#         module.register_backward_hook(lambda module, grad_input, grad_output: save_gradient(grad_output[0]))

# Access gradients and feature maps
gradient = gradients[0].cpu().data.numpy()
feature_maps = target_layer.weight.cpu().data.numpy()

print(f"Dimensions des gradients : {gradient.shape}")
print(f"Dimensions des cartes de caractéristiques : {feature_maps.shape}")


# Global Average Pooling of gradients
weights = np.mean(gradient, axis=(2, 3))  # Moyenne sur les dimensions height et width (pour chaque carte)

# Vérifier que les dimensions sont compatibles
assert weights.shape[1] == feature_maps.shape[0], "Le nombre de poids ne correspond pas au nombre de cartes de caractéristiques"

# Generate Grad-CAM heatmap
cam = np.zeros((feature_maps.shape[2], feature_maps.shape[3]), dtype=np.float32)  # Prendre la taille des cartes de caractéristiques (3x3)

# Appliquer les poids sur chaque carte de caractéristiques
for i in range(weights.shape[1]):  # Parcourir chaque carte de caractéristiques
    for j in range(feature_maps.shape[1]):  # Parcourir chaque canal de la carte de caractéristiques
        cam += weights[0, i] * feature_maps[i, j, :, :]  # Multiplier le poids par chaque canal de la carte de caractéristiques

cam = np.maximum(cam, 0)  # ReLU
cam = cam / cam.max()  # Normalize

# Resize heatmap to match the image size
heatmap = cv2.resize(cam, (image.size[0], image.size[1]))
heatmap = (heatmap * 255).astype(np.uint8)

# Overlay heatmap on the image
heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap_img, 0.4, 0)

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(superimposed_img)
plt.axis("off")
plt.show()
