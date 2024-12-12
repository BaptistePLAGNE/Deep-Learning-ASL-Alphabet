import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model_scn2 import ExtendedSimpleCNN2D
import cv2
from IPython.display import display

def forward_hook(module,input,output):
    activation.append(output)
    
def backward_hook(module,grad_in,grad_out):
    grad.append(grad_out[0])

activation=[]
grad=[]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ExtendedSimpleCNN2D(3,8)

state_dictonary = torch.load("./03_runs/class_langage-4/best_checkpoint.pth", map_location = device)
model.load_state_dict(state_dictonary['model_state'])
model = model.to(device)

model.eval()

preprocess = transforms.Compose([
    transforms.Resize((135, 135)),  # Redimensionner les images
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),         # Convertir en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess an image
image_path = "./01_data/test/A/10.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)

model.body[3].block[0].block_1[0].register_backward_hook(backward_hook)
model.body[3].block[0].block_1[0].register_forward_hook(forward_hook)

# Forward pass
output = model(input_tensor)
loss = output[0,7]
model.zero_grad()
loss.backward()

grads = grad[0].cpu().data.numpy().squeeze()
fmap = activation[0].cpu().data.numpy().squeeze()

tmp = grads.reshape([grads.shape[0],-1])
weights=np.mean(tmp,axis=1)


cam = np.zeros(grads.shape[1:])
for i,w in enumerate(weights):
    cam += w*fmap[i,:]

cam=(cam>0)*cam
cam=cam/cam.max()*255

npic=np.array(image)
cam = cv2.resize(cam,(npic.shape[0],npic.shape[1]))

heatmap=cv2.applyColorMap(np.uint8(cam),cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(npic, 0.6, heatmap, 0.4, 0)

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
