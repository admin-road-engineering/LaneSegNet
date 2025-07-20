import torch
from mmcv import ops

print("PyTorch CUDA available:", torch.cuda.is_available())
try:
    boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32).cuda()
    scores = torch.tensor([0.9], dtype=torch.float32).cuda()
    result = ops.nms(boxes, scores, 0.5)
    print("MMCV CUDA ops are working.")
except Exception as e:
    print(f"MMCV CUDA ops failed: {e}") 