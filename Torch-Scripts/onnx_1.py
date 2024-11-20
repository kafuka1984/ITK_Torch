import torch
import torch.nn as nn
import onnx

# 加载模型
model = torch.load("/database/home/tangchi/Deployments/medical.ai/model/model.pt")
# 将模型转换为torchScript格式
script_model = torch.jit.script(model)
# 保存torchScript模型
torch.jit.save(script_model, "model.ts")
