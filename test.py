from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch.nn.functional as nnF
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


img1_pil = Image.open('test1.jpg').convert('RGB')
img2_pil = Image.open('test2.jpg').convert('RGB')


orig_height, orig_width = img1_pil.size[1], img1_pil.size[0]  # PIL尺寸是 (width, height)
print(f"Original image size: {orig_height} x {orig_width}")


to_tensor = T.ToTensor()
img1 = to_tensor(img1_pil)
img2 = to_tensor(img2_pil)

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()


def preprocess(img1, img2):

    target_height, target_width = 520, 960


    img1_resized = F.resize(img1, size=[target_height, target_width], antialias=False)
    img2_resized = F.resize(img2, size=[target_height, target_width], antialias=False)


    img1_transformed, img2_transformed = transforms(img1_resized, img2_resized)

    img1_batch = img1_transformed.unsqueeze(0)
    img2_batch = img2_transformed.unsqueeze(0)

    return img1_batch, img2_batch, (target_height, target_width)



img1_batch, img2_batch, processed_size = preprocess(img1, img2)
processed_height, processed_width = processed_size

model = raft_large(weights=weights, progress=False).to(device)
model = model.eval()

img1_batch = img1_batch.to(device)
img2_batch = img2_batch.to(device)

with torch.no_grad():
    flow_predictions = model(img1_batch, img2_batch)

flow = flow_predictions[-1]

print(f"Processed flow dtype = {flow.dtype}")
print(f"Processed flow shape = {flow.shape} = (N, 2, H_processed, W_processed)")
print(f"Processed flow min = {flow.min():.4f}, max = {flow.max():.4f}")



def resize_flow(flow_tensor, target_size, mode='bilinear', align_corners=False):

    # 获取原始尺寸
    n, c, h, w = flow_tensor.shape
    target_h, target_w = target_size


    scale_h = target_h / h
    scale_w = target_w / w


    resized_flow = nnF.interpolate(
        flow_tensor,
        size=target_size,
        mode=mode,
        align_corners=align_corners
    )


    resized_flow[:, 0, :, :] *= scale_w  # u 分量 (水平)
    resized_flow[:, 1, :, :] *= scale_h  # v 分量 (垂直)

    return resized_flow


# 将光流调整回原始尺寸
flow_resized = resize_flow(flow, (orig_height, orig_width))

print(f"\nResized flow dtype = {flow_resized.dtype}")
print(f"Resized flow shape = {flow_resized.shape} = (N, 2, H_orig, W_orig)")
print(f"Resized flow min = {flow_resized.min():.4f}, max = {flow_resized.max():.4f}")


flow_np = flow_resized[0].cpu().numpy()
print(f"\nFlow numpy array shape = {flow_np.shape}")


u = flow_np[0]
v = flow_np[1]

print(f"\nHorizontal flow (u) shape: {u.shape}")
print(f"Vertical flow (v) shape: {v.shape}")
print(f"Horizontal flow range: [{u.min():.2f}, {u.max():.2f}]")
print(f"Vertical flow range: [{v.min():.2f}, {v.max():.2f}]")

# 可视化光流（示例）
import matplotlib.pyplot as plt


magnitude = np.sqrt(u ** 2 + v ** 2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img1_pil)
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(img2_pil)
plt.title('Image 2')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(u, cmap='coolwarm')
plt.title('Horizontal Flow (u)')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(magnitude, cmap='hot')
plt.title('Flow Magnitude')
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()




