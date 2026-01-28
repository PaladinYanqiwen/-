# H2RF Prompt Fusion 模块

## 功能概述

`H2RFPromptFusion` 是一个集成视觉令牌和文本推理的融合模块，用于后续的情感分析任务。它将HRF生成的R2（意见推理）与视觉信息结合，进行句子级或方面级的融合。

## 输入接口规范

### 核心输入参数

| 参数 | 类型 | 形状 | 说明 |
|-----|------|------|------|
| **O** | `torch.Tensor` 或 `numpy.ndarray` | `(B, M, H)` 或 `(M, H)` | 视觉令牌嵌入 |
| **S_list** | `List[str]` | 长度 B | 句子列表（来自HRF输入） |
| **R2_list** | `List[str]` | 长度 B | 意见推理列表（来自HRF输出） |
| **mode** | `str` | - | 融合模式：`"sentence"` 或 `"aspect"` |
| **template_id** | `str` | - | 模板ID：`Ps1/Ps2/Ps3` 或 `Pa1/Pa2/Pa3` |
| **A_list** | `List[str]` | 长度 B | 方面列表（仅当 mode="aspect" 时需要） |

### O 的格式说明

视觉令牌 O 可以有以下几种格式：

#### 格式 1: 批处理输入 (推荐)
```python
O.shape = (B, M, H)
# B: batch_size (与 S_list, R2_list 长度一致)
# M: num_visual_tokens (视觉令牌数，如 10, 20, 50 等)
# H: hidden_size (必须 == 768，用于BERT的隐藏维度)

例：torch.randn(2, 10, 768)  # 2个样本，每个10个视觉令牌
```

#### 格式 2: 单样本输入 (自动扩展)
```python
O.shape = (M, H)
# 会自动扩展为 (1, M, H)

例：torch.randn(10, 768)  # 自动变为 (1, 10, 768)
```

#### 格式 3: 不同数据类型
- **torch.Tensor**: 直接使用
- **numpy.ndarray**: 自动转换为 torch.Tensor

### 使用示例

#### 方式 1: 从 torch.Tensor 直接输入
```python
from Hr2F.fusion import H2RFPromptFusion
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion = H2RFPromptFusion(model_name="bert-base-uncased").to(device)

# 视觉令牌（例如来自CLIP编码器）
O = torch.randn(2, 10, 768, device=device)

# HRF生成的数据
S_list = ["Amazing product", "Poor service"]
R2_list = ["positive indicators", "mixed sentiment"]

# 句子级融合
mask_logits, mask_pos = fusion(
    O=O,
    S_list=S_list,
    R2_list=R2_list,
    mode="sentence",
    template_id="Ps1"
)
```

#### 方式 2: 从 numpy 数组加载
```python
import numpy as np
from Hr2F.fusion import VisualTokenLoader

# 从文件加载视觉令牌
O_np = np.load("visual_tokens.npy")  # 形状: (2, 10, 768)

mask_logits, mask_pos = fusion(
    O=O_np,  # 自动转换为 torch.Tensor
    S_list=S_list,
    R2_list=R2_list,
    mode="sentence",
    template_id="Ps1"
)
```

#### 方式 3: 使用 VisualTokenLoader 辅助类
```python
from Hr2F.fusion import VisualTokenLoader

# 创建虚拟视觉令牌（用于测试）
O_dummy = VisualTokenLoader.create_dummy_visual_tokens(
    batch_size=2,
    num_visual=10,
    hidden_size=768,
    device=device
)

# 验证视觉令牌格式
is_valid = VisualTokenLoader.validate_visual_tokens(O_dummy, expected_hidden_size=768)

# 从文件加载
O_from_file = VisualTokenLoader.load_from_numpy("path/to/visual_tokens.npy")
```

#### 方式 4: 从CLIP编码器获取
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 初始化CLIP
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# 加载图像
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt").to(device)

# 获取视觉嵌入
with torch.no_grad():
    vision_outputs = model.vision_model(**inputs)
    O = vision_outputs.last_hidden_state  # (1, num_patches, 768)
```

## 融合模式和模板

### 句子级融合 (mode="sentence")

| 模板ID | 文本格式 |
|-------|--------|
| **Ps1** | `The sentence "{S}" combined with the "{R2}" has [MASK] emotion` |
| **Ps2** | `The sentence "{S}" combined with the "{R2}" presents a [MASK] sentiment` |
| **Ps3** | `The details in the sentence "{S}" and the analysis "{R2}" reveal a [MASK] sentiment` |

### 方面级融合 (mode="aspect")

| 模板ID | 文本格式 |
|-------|--------|
| **Pa1** | `For the aspect "{A}", "{S}" combined with the "{R2}" has [MASK] emotion` |
| **Pa2** | `For the aspect "{A}", "{S}" combined with the "{R2}" presents a [MASK] sentiment` |
| **Pa3** | `For the aspect "{A}", the details in the sentence "{S}" and the analysis "{R2}" reveal a [MASK] sentiment` |

## 输出格式

### 返回值

```python
mask_logits, mask_pos = fusion(...)

# mask_logits: (B, vocab_size)
#   - 张量，包含 [MASK] 位置的词表预测分数
#   - 可使用 torch.argmax(mask_logits, dim=1) 获取预测词ID
#   - 可使用 torch.softmax(mask_logits, dim=1) 获取概率分布

# mask_pos: List[int]，长度 B
#   - 每个样本中 [MASK] 在融合序列中的位置
#   - 融合序列结构: [CLS] + O (M个令牌) + [SEP] + prompt_text + [SEP]
```

### 获取预测结果

```python
# 获取预测的词ID
pred_token_ids = torch.argmax(mask_logits, dim=1)

# 获取概率分布
pred_probs = torch.softmax(mask_logits, dim=1)

# 转换为词汇
tokenizer = fusion.tokenizer
pred_tokens = [tokenizer.decode([tid]) for tid in pred_token_ids]
```

## 数据流整合示例

整个HRF到融合的完整流程：

```python
from hrf.hrf_r2_opinion_reasoning import LLMConfig, build_client, run_hrf
from Hr2F.fusion import H2RFPromptFusion, VisualTokenLoader
import torch

# 1. 初始化HRF
cfg = LLMConfig.from_env()
client = build_client(cfg)

# 2. 生成R2
S = "This product is amazing."
I_CLIP = "a person holding a smartphone"
out = run_hrf(client, cfg, mode="sentence", step="step2", 
              S=S, I_CLIP=I_CLIP)
R2 = out["R2"]

# 3. 获取视觉令牌 O（由你后续提供）
O = torch.randn(1, 10, 768, device=device)  # 临时示例

# 4. 融合
fusion = H2RFPromptFusion().to(device)
mask_logits, mask_pos = fusion(
    O=O,
    S_list=[S],
    R2_list=[R2],
    mode="sentence",
    template_id="Ps1"
)

# 5. 获取预测
pred_emotion = fusion.tokenizer.decode([torch.argmax(mask_logits[0])])
print(f"R2: {R2}")
print(f"预测情感: {pred_emotion}")
```

## 常见错误和解决

| 错误 | 原因 | 解决方案 |
|-----|-----|--------|
| `hidden_size not match` | O的隐藏维度不是768 | 使用线性层将O投影到768维 |
| `A_list is None` | aspect模式未提供A_list | 提供方面列表：`A_list=["quality", "price"]` |
| `[MASK] not found` | 模板中没有[MASK]标记 | 确保使用了正确的模板ID |
| 维度不匹配 | 批次大小不一致 | 确保len(S_list) == len(R2_list) == O.shape[0] |

## 后续集成清单

- [ ] 提供视觉令牌提取模块（CLIP或其他编码器）
- [ ] 验证O的维度和数据类型
- [ ] 测试不同的模板组合
- [ ] 评估融合效果
- [ ] 微调BERT权重（可选）
