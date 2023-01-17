# %%
import torch
from torchvision.models import AlexNet
# from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.append('../src')
from training import preprocess

# %%
modelfile = input("Enter the model file: ")
imagefile = input("Enter the image file: ")

# %%
weight = torch.load(modelfile, map_location=torch.device('cpu'))
model = AlexNet(num_classes=2)
model.load_state_dict(weight)

# %%
print(modelfile, imagefile)

# %%
# plt.imshow(plt.imread(imagefile))

# %%
print(model)

# %%
image_tensor = preprocess(Image.open(imagefile))
print(image_tensor.shape)
image_tensor = image_tensor.unsqueeze(0)
print(image_tensor.shape)

# %%
# predict
model.eval()
with torch.no_grad():
    output = model(image_tensor)
    print(output)

# %%



