# 图像/视频分割模型学习笔记

> 基于 maskformer_student.ipynb、mask2former_student.ipynb、sam1_student.ipynb、sam2_student.ipynb 整理

---

## 一、MaskFormer（maskformer_student.ipynb）

### 1.1 模块功能

MaskFormer 将语义分割统一为 **mask classification** 范式：不再对每个像素做逐像素分类，而是预测一组 (mask, class) 对，再合成最终分割图。核心三大模块：

| 模块 | 作用 |
|------|------|
| **Backbone (ResNet-50)** | 提取多尺度特征 C0–C3，通道分别为 256/512/1024/2048 |
| **Pixel Decoder (FPN)** | 自顶向下融合多尺度特征（类似 FPN），输出 pixel embedding (B,256,H/4,W/4) 和 transformer_feature (B,256,H/32,W/32) |
| **Transformer Decoder** | 100 个可学习 Object Query 通过 6 层 Decoder（含 self-attention + cross-attention）与图像特征交互，输出 query hidden states |
| **Predictor** | class_head (1层MLP) 预测类别；mask_head (3层MLP) 预测 mask embedding；最终通过 `einsum(mask_embed, pixel_embedding)` 得到 mask |

### 1.2 Pipeline

```
Image → Backbone → multi-scale features [C0,C1,C2,C3]
                         ↓
                  Pixel Decoder (FPN)
                   ↓              ↓
         transformer_feature    pixel_embedding
                   ↓
         Transformer Decoder (6层, 100 queries)
                   ↓
              Predictor
           ↓            ↓
      class_logits   mask_embeds
                         ↓
          einsum(mask_embeds, pixel_embedding) → pred_masks
```

### 1.3 Pixel Decoder 详细结构

FPN 自顶向下融合过程：

```
C3 → 1x1 conv(2048→256) → c3_feature (B,256,H/32,W/32)
  ↓ upsample + C2 → 1x1 conv(1024→256) → 相加 → 3x3 conv → c2_feature (B,256,H/16,W/16)
  ↓ upsample + C1 → 1x1 conv(512→256)  → 相加 → 3x3 conv → c1_feature (B,256,H/8,W/8)
  ↓ upsample + C0 → 1x1 conv(256→256)  → 相加 → 3x3 conv → c0_feature (B,256,H/4,W/4)
```

- `transformer_feature = c3_feature`（最高层，送入 Transformer Decoder 做 cross-attention 的 KV）
- `pixel_embedding = c0_feature`（最低层/最高分辨率，用于与 mask embedding 做点积生成 mask）

### 1.4 思考题解答

**Q1: 对照 Pixel Decoder 的多尺度处理 pipeline，理解代码。**

> 代码中 `forward` 从 `i = len(features)-1`（即 C3）到 `i = 0`（即 C0）逐级遍历。对 C3 仅做 lateral conv（1×1）得到初始 `x`；对 C2、C1、C0 先将 `x` 上采样到当前层的空间尺寸，再与 lateral conv 的结果相加，最后过 3×3 conv。`fpn_features[0]` 是 C3 层特征（送给 Transformer），`fpn_features[-1]` 是 C0 层特征（做 pixel embedding）。

**Q2: image_features 有几个 tokens？object queries 有几个 tokens？**

> - `image_features`：shape 为 (B, C, H, W)，flatten 后变为 (B, H×W, C)。对于 H/32 × W/32 的特征图，token 数 = H/32 × W/32。例如输入 512×512 的图，则 token 数 = 16×16 = 256。
> - `object queries`：固定 **100** 个（由 `num_queries=100` 决定）。

**Q3: Transformer Decoder 的每一层由什么构成？顺序是怎样的？**

> `nn.TransformerDecoderLayer` 由以下三部分按顺序组成：
> 1. **Self-Attention**：query tokens 之间互相关注，建模 query 间关系
> 2. **Cross-Attention**：query 作为 Q，image features 作为 K 和 V，让 query 从图像中提取信息
> 3. **FFN (Feed-Forward Network)**：两层全连接 + ReLU 激活
>
> 每个子层之后都有残差连接和 LayerNorm。

**Q4: 找出 MaskFormer 流程图中的一处小错误/瑕疵。**

> 在流程图中，cross-attention 的 KV 来源标注为 pixel decoder 的输出（pixel embedding），但实际代码中 **Transformer Decoder 的 KV 是 `transformer_feature`（即 C3 层/最高层特征）**，而非最终的 pixel embedding。pixel embedding（C0 层高分辨率特征）只用于最后与 mask embedding 做点积，不参与 Transformer Decoder 的计算。图中对此的标注有误或容易产生误导。

---

## 二、Mask2Former（mask2former_student.ipynb）

### 2.1 模块功能

Mask2Former 是 MaskFormer 的升级版，在统一的 mask classification 框架下引入了多项改进，使其能够同时胜任语义分割、实例分割和全景分割（"Universal Segmentation"）。

| 改进点 | 说明 |
|--------|------|
| **Masked Attention** | cross-attention 时，每个 query 仅关注其预测 mask 区域内的像素，而非全局，大幅提升效率和精度 |
| **多尺度 Deformable Attention** | Pixel Decoder 改用 deformable attention 处理多尺度特征 |
| **更深的 Decoder** | Transformer Decoder 从 6 层增加到 **9 层**（配置中 decoder_layers=10 但实际 layers 为 9） |
| **Backbone 升级** | 使用 Swin-B 替代 ResNet-50，特征提取能力更强 |
| **Query 分离** | queries 拆分为 `queries_features`（内容 embedding）和 `queries_embedder`（位置 embedding），分别初始化 |
| **Level Embedding** | 为不同尺度的特征添加可学习的 level embedding，帮助模型区分特征来自哪个尺度 |
| **逐层 Mask Refinement** | 每层 Decoder 都输出中间 mask，mask 在各层之间不断 refine |

### 2.2 Decoder 内部结构

每个 `Mask2FormerMaskedAttentionDecoderLayer` 包含：
1. **Self-Attention** (`self_attn`)：query 之间交互
2. **Cross-Attention** (`cross_attn`)：query 对图像特征做 masked attention（仅关注预测 mask 内部区域）
3. **FFN** (`fc1` → ReLU → `fc2`)
4. 各子层都有 LayerNorm

### 2.3 Decoder 的 Q / K&V

- **Q**：`queries_features`（内容）+ `queries_embedder`（位置）
- **K & V**：pixel decoder 处理后的图像特征 + `position_embedder`（正弦位置编码）+ `level_embed`（尺度标识）

### 2.4 思考题解答

**Q1: 打开 decoder 详细输出，理解每个 module 的含义。**

> - `position_embedder (Mask2FormerSinePositionEmbedding)`：正弦位置编码，为图像特征（K&V）提供空间位置信息
> - `queries_embedder (Embedding(100,256))`：100 个 query 的位置 embedding
> - `queries_features (Embedding(100,256))`：100 个 query 的内容 embedding（初始化学习到的内容先验）
> - `decoder.layers`：9 层 Decoder Layer，每层含 self-attn、cross-attn (masked)、FFN
> - `decoder.mask_predictor`：每层 Decoder 输出后，mask_predictor 使用 3 层 MLP 将 query hidden state 映射为 mask embedding，再与 pixel embedding 点积生成 mask → 用于下一层的 masked attention
> - `level_embed (Embedding(3,256))`：3 个尺度（stride 8/16/32）的可学习 embedding，拼接到对应尺度的图像特征上

**Q2: Mask2Former 与 MaskFormer 输出精度对比，优劣势？**

> **优势：**
> - 分割精度显著更高，边界更加精细。Masked Attention 让每个 query 聚焦于自身负责的区域，避免了全局注意力中的干扰
> - 多尺度特征融合更有效（deformable attention + level embedding）
> - 逐层 mask refinement，mask 质量随 Decoder 深度逐步提升
> - 统一架构可处理语义/实例/全景三种分割任务
>
> **劣势：**
> - 模型更复杂，参数量更大（Swin-B backbone + 更深 Decoder）
> - 推理速度相对较慢
> - Masked Attention 需要每层都预测中间 mask，增加了计算开销

---

## 三、SAM1 — Segment Anything Model（sam1_student.ipynb）

### 3.1 模块功能

SAM 是 Meta 提出的 **open-vocabulary（开放词汇）** 分割模型。与 MaskFormer/Mask2Former 不同，SAM 不做类别预测，而是通过 prompt（point/box/mask）指定要分割的目标，输出分割 mask。

| 模块 | 作用 |
|------|------|
| **Image Encoder (ViT-Huge)** | 32 个 ViT Block + Neck，将 1024×1024 图像编码为 (1,256,64,64) 特征 |
| **Prompt Encoder** | 将 point/box/mask 等 prompt 编码为 sparse embedding 和 dense embedding |
| **Mask Decoder (TwoWayTransformer)** | 2 层双向 Transformer Block + upscaling + HyperNetwork MLP，输出多个 mask 及 IoU 预测 |

### 3.2 架构细节

**Image Encoder：**
- 输入：1024×1024 RGB 图像
- Patch Embed：16×16 patch → 1280 维
- 64×64 = 4096 个 tokens，每个 1280 维
- 经过 32 个 ViT Block 后，通过 Neck（1×1 conv + 3×3 conv）降维到 256 通道
- 输出：(1, 256, 64, 64)

**Prompt Encoder：**
- Point prompt → sparse embedding（point 坐标 + 正/负标签 embedding）
- Box prompt → sparse embedding（两个角点）
- Mask prompt → dense embedding（通过 CNN 下采样到 256×64×64）
- 无 mask 时使用 `no_mask_embed` 可学习 embedding

**Mask Decoder：**
- 可学习 tokens：`iou_token`（1个）+ `mask_tokens`（4个），与 prompt 的 sparse embedding 拼接，共 7 个 query tokens（1 iou + 4 mask + 2 prompt points）
- TwoWayTransformer：2 层双向注意力 Block
- 输出 3 个 mask（multi-mask 模式）+ 对应 IoU 预测分数
- 通过 ConvTranspose2d 上采样到 256×256，再与 HyperNetwork MLP 输出做点积生成最终 mask

### 3.3 思考题解答

**Q1: 根据 (1,64,64,1280) 的 ViT tensor shape，一张图被 patchify 为了多少个 tokens？**

> 64 × 64 = **4096 个 tokens**。原图 1024×1024，patch size 为 16×16，所以 1024/16 = 64，共 64×64 = 4096 个 patch/token。

**Q2: mask_decoder 有几个 block？**

> **2 个** TwoWayAttentionBlock（`layers: (0-1): 2 x TwoWayAttentionBlock`），加上最后一个 `final_attn_token_to_image`。

**Q3: 每个 block 有 self-attn 吗？**

> **有。** 每个 TwoWayAttentionBlock 包含 `self_attn`，用于 query tokens 之间的自注意力。

**Q4: 每个 block 有 cross-attn 吗？有几个？**

> **有，2 个。** 分别是：
> - `cross_attn_token_to_image`：token（query）→ 图像特征的 cross-attention
> - `cross_attn_image_to_token`：图像特征 → token 的 cross-attention
>
> 这就是 **TwoWay（双向）** 的含义：不仅 query 从图像提取信息，图像特征也从 query/prompt 获取信息。

**Q5: 每个 block 中这些 attn 模块的顺序？**

> 1. `self_attn`（query tokens 间 self-attention）→ norm1
> 2. `cross_attn_token_to_image`（query attend to image）→ norm2
> 3. `mlp`（前馈网络）→ norm3
> 4. `cross_attn_image_to_token`（image attend to query）→ norm4

**Q6: 调整 points_per_side 观察全景分割结果。**

> `points_per_side` 控制 SamAutomaticMaskGenerator 在图像上均匀采样 prompt point 的密度。值越大（如 64），生成的 mask 越多越细粒度；值越小（如 8），mask 更粗粒度、数量更少。默认 32 是一个较好的平衡点。

---

## 四、SAM2 — Segment Anything Model 2（sam2_student.ipynb）

### 4.1 模块功能

SAM2 将 SAM 从 **单帧图像分割** 扩展到 **视频目标分割与跟踪**。核心创新是引入 **Memory 机制**，使模型能够跨帧传播分割结果。

| 模块 | 作用 |
|------|------|
| **Image Encoder (Hiera)** | 使用 Hiera 架构（层级化 ViT）替代原始 ViT，更高效 |
| **Prompt Encoder** | 同 SAM1，支持 point/box/mask prompt |
| **Mask Decoder** | 同 SAM1，生成 mask 和 IoU 预测 |
| **Memory Encoder** | 将当前帧的 mask 预测和图像特征编码为 memory feature |
| **Memory Attention** | 当前帧特征通过 cross-attention 关注历史帧的 memory features |
| **Memory Bank (FIFO)** | 存储最近 **7 帧** 的 memory features 和 object pointers，先进先出 |

### 4.2 视频分割 Pipeline

```
第0帧: Prompt (point/mask) → Prompt Encoder → Mask Decoder → 初始 mask
         ↓
       Memory Encoder → Memory Bank (存入 maskmem_features + obj_ptr)
         
第N帧: Image Encoder → 当前帧特征
         ↓
       Memory Attention (当前帧特征 × Memory Bank 中历史帧) → 增强特征
         ↓
       Mask Decoder → 当前帧 mask
         ↓
       Memory Encoder → 更新 Memory Bank (FIFO, 保留最近7帧)
```

### 4.3 Memory Bank 结构

从输出可以观察到：
- `maskmem_features`：每帧 64 维特征，存储最近 7 帧（FIFO），shape 从 (1,64,64,64) 逐步增长到 (1,448,64,64) 后保持不变
- `obj_ptr`：每帧一个 256 维 object pointer，同样保留最近 7 帧，shape 从 (1,256) 增长到 (7,256)
- 超过 7 帧后，最旧的帧特征被丢弃（先进先出）

### 4.4 使用方式

- **单目标跟踪**：首帧给 point 或 mask prompt，`propagate_in_video` 自动传播到后续帧
- **多目标跟踪**：首帧对不同 obj_id 分别给 mask prompt，模型同时跟踪多个目标
- **失败案例**：复杂遮挡和快速运动场景（如 duck 序列）下，SAM2 可能丢失目标

### 4.5 与 SAM1 的关键区别

| 维度 | SAM1 | SAM2 |
|------|------|------|
| 输入 | 单帧图像 | 视频序列 |
| Backbone | ViT-Huge (32层) | Hiera-B+ (更高效) |
| Memory | 无 | FIFO Memory Bank (7帧) |
| 跨帧 | 不支持 | Memory Attention 实现跨帧关联 |
| 核心函数 | `SamPredictor.predict()` | `add_new_points_or_box` / `add_new_mask` + `propagate_in_video` |

---

## 五、四个模型的横向对比

| 特性 | MaskFormer | Mask2Former | SAM1 | SAM2 |
|------|-----------|------------|------|------|
| 任务 | 语义分割 | 语义/实例/全景 | Prompt 分割 | 视频 Prompt 分割 |
| 是否分类 | 是 (150类) | 是 (150类) | 否 (open-vocab) | 否 (open-vocab) |
| Backbone | ResNet-50 | Swin-B | ViT-H | Hiera-B+ |
| Decoder 层数 | 6 | 9 | 2 (TwoWay) | 2 (TwoWay) |
| Query 数量 | 100 | 100 | 5-7 (iou+mask+prompt) | 同 SAM1 |
| 核心创新 | mask classification | Masked Attention | Prompt 驱动分割 | Memory 跨帧传播 |
| 输入交互 | 无 | 无 | point/box/mask | point/box/mask |
