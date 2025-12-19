import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Concat, Detect
from torch.nn import Upsample

# 加载模型（从YAML构建，确保与配置文件一致）
model = YOLO("./yolov8n.pt")

# 存储层信息：按执行顺序记录每个目标模块的输入/输出尺寸
layer_info = []


def hook_fn(module, inputs, outputs):
    """ 捕获目标模块的输入输出尺寸 """
    # 定义模块类型到名称的映射
    module_type_name = {
        Conv: "Conv",
        C2f: "C2f",
        SPPF: "SPPF",
        Upsample: "Upsample",
        Concat: "Concat",
        Detect: "Detect"
    }
    module_name = module_type_name.get(type(module), "Unknown")

    # 解析输入输出形状（支持嵌套Tensor/列表/元组）
    def parse_shape(data):
        if isinstance(data, torch.Tensor):
            return list(data.shape[1:])  # 忽略批次维度
        elif isinstance(data, (list, tuple)):
            return [parse_shape(x) for x in data]
        else:
            return str(type(data))  # 其他类型转为字符串

    # 记录信息
    layer_info.append({
        "name": module_name,
        "input": [parse_shape(inp) for inp in inputs],
        "output": parse_shape(outputs)
    })


# 注册钩子：仅针对YAML中出现的模块类型
target_modules = (Conv, C2f, SPPF, Upsample, Concat, Detect)
for module in model.modules():
    if isinstance(module, target_modules):
        module.register_forward_hook(hook_fn)

# 运行前向传播（输入尺寸需与YAML配置匹配）
example_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    model(example_input)

# 打印结果（按执行顺序，对应YAML中的层顺序）
print("根据YAML配置的各层输出尺寸：")
for idx, info in enumerate(layer_info):
    print(f"层 {idx} ({info['name']}):")
    print(f"  输入形状: {info['input']}")
    print(f"  输出形状: {info['output']}\n")