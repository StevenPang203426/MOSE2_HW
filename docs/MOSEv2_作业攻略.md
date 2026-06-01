# MOSEv2 作业完全攻略

## 一、任务概述

### 1.1 任务目标

使用视频目标分割（VOS）模型，对 MOSEv2 验证集（Validation Set）中的视频进行推理。已提供 418 个视频的推理结果（基于 SAM2.1-B+），需要对**剩余 15 个视频**进行推理，然后将所有 433 个视频的结果合并提交到 codabench。

### 1.2 评分机制

| 情况 | 预计分数（J&F'） |
|------|----------------|
| 只提交已有的 418 个视频（15 个视频 0 分） | ~41.05 |
| 15 个视频全部使用 SAM2.1-B+ 推理 | ~43.25 |
| **超过 44 分** | **极大 bonus** |

评分指标为 J&F'（Jaccard & F-measure），越高越好，满分 100。

### 1.3 提交要求

- **提交链接：** https://www.codabench.org/competitions/10062/#/participate-tab
- **账号命名：** `cv_{学号}`，如 `cv_25213050108`
- **提交形式：** zip 包，包含全部 433 个视频的分割结果
- **每天提交上限：** 20 次，自动取最高分
- **截止日期：** 6 月 2 日前

### 1.4 提交文件结构

```
submission.zip
├── video_name_1/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── video_name_2/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── video_name_.../
```

### 1.5 任务包结构

```
mosev2_作业.zip
├── JPEGImages/      # 待测 15 个视频的 frames（JPEG 图片序列）
├── Annotations/     # 待测 15 个视频的首帧 mask（PNG）
└── Output/          # 已有 418 个视频的输出结果
```

---

## 二、背景知识

### 2.1 什么是 MOSEv2

MOSEv2（coMplex video Object SEgmentation v2）是目前全球最复杂的视频目标分割数据集，由复旦大学等团队构建。它包含 5,024 个视频、701,976+ 高质量 mask、10,074 个目标对象，涵盖 200 个类别。

相比前代 MOSEv1，MOSEv2 新增了以下高难度场景：

- 恶劣天气（雨、雪、雾）
- 低光照（夜间、水下）
- 多镜头切换（Multi-shot）
- 伪装目标（Camouflage）
- 非物理目标（阴影、倒影）
- 知识依赖场景（需要外部知识才能区分目标）
- 以及继承自 v1 的：目标遮挡、消失重现、密集同类干扰、不规则形变等

SAM2 在 MOSEv1 上能达到 76.4%，但在 MOSEv2 上骤降至 50.9%，这表明当前方法在真实复杂场景下仍有巨大提升空间。

### 2.2 什么是半监督 VOS

半监督视频目标分割（Semi-supervised VOS）的设定：
- **输入：** 完整视频帧序列 + 第一帧的目标分割 mask
- **输出：** 后续每一帧的目标分割 mask
- **核心挑战：** 在遮挡、消失、外观变化等情况下持续跟踪和分割目标

---

## 三、模型选型分析

### 3.1 SAM2.1（基线方案）

**简介：** Meta 于 2024 年发布的统一图像和视频分割基础模型，通过 memory 模块实现视频帧间的目标一致性。

**架构：** 图像编码器（Hiera） + Memory Attention + Mask Decoder + Memory Bank

**模型尺寸：**

| 模型 | 参数量 | 显存需求 | 推理速度 |
|------|--------|---------|---------|
| SAM2.1-B+（Base Plus） | ~80M | ~8 GB | ~30 FPS |
| SAM2.1-L+（Large） | ~220M | ~20-24 GB | ~15 FPS |

**优点：**
- 代码成熟，官方维护，开箱即用
- 社区文档丰富，debug 容易
- 是本作业提供的 418 个结果所使用的模型

**缺点：**
- 使用贪心策略（greedy selection）选择每帧最高 IoU 的 mask
- 存在严重的**误差累积**问题：一旦某帧分割错误，错误会级联传播到后续所有帧
- 在 MOSEv2 的遮挡、消失重现场景中表现不佳

**适用场景：** 跑通流程、验证格式、作为对照基线

**GitHub：** https://github.com/facebookresearch/sam2

---

### 3.2 SAM2Long（⭐ 强烈推荐）

**简介：** 上海 AI Lab 和 CUHK 提出的 training-free 改进策略（ICCV 2025），通过 Memory Tree 解决 SAM2 的误差累积问题。

**核心创新：**
- 不再每帧只保留一个最佳 mask，而是维护**多条候选路径**（Memory Pathways）
- 每帧为每条路径生成多个候选 mask，形成树状搜索结构
- 使用累积得分选择固定数量的最优路径
- 最终选择全局最优路径作为结果

**通俗理解：** SAM2 像是走迷宫只看一步，SAM2Long 像是同时探索多条路线，最后选最优的那条。

**性能提升：**
- 在 9 个 VOS + 3 个 VOT 共 12 个 benchmark 上平均提升 3.7 个 J&F 点
- 在长视频数据集（SA-V、LVOS）上提升高达 5.3 个 J&F 点
- 在 MOSEv2 验证集上，SAM2Long-L 达到 42.9% J&F'，是所有零训练方法中最优的

**算力需求：**

| 配置 | 显存需求 | 推理时间（15 个视频） |
|------|---------|---------------------|
| SAM2Long + B+ | ~10-12 GB | ~20-40 分钟 |
| SAM2Long + L+ | ~24-30 GB | ~40-90 分钟 |

> 注意：SAM2Long 维护 3 条路径，推理时间约为 SAM2 的 2-3 倍，但显存增加不多。

**关键超参数：**
- `num_pathways`：路径数量，**推荐 3**（论文实验证明 3 是最优平衡点，4 无额外增益）
- `δ_iou`：IoU 阈值，用于筛选可靠帧，0.1-0.7 均有竞争力
- `[w_low, w_high]`：memory attention 权重调制范围

**GitHub：** https://github.com/Mark12Ding/SAM2Long

---

### 3.3 SAM3（进阶方案）

**简介：** Meta 于 2025 年 11 月发布的第三代 Segment Anything 模型，核心创新是引入了 Promptable Concept Segmentation（PCS），可以用文本或图像示例作为 prompt 来分割所有匹配的对象实例。

**与本任务的关系：**
- SAM3 的核心创新在于**概念级分割**（如输入"红色棒球帽"找到所有实例），这与本作业的半监督 VOS 设定（给定首帧 mask）不完全一致
- 但 SAM3 的视频 tracker 部分复用了 SAM2 的架构并做了改进，理论上在 mask propagation 上也更强
- 在 ICCV 2025 PVUW 挑战赛 MOSEv2 赛道上，基于 SAM3 的 Re-Prompting 方案达到了 51.17% J&F'（第 3 名），但该方案使用了额外的 anchor re-prompting 策略

**优点：**
- 模型能力更强，尤其在区分相似对象时（presence token 机制）
- 兼容 SAM2 的 mask prompt 接口

**缺点：**
- 需要自行适配半监督 VOS 推理流程
- 模型更大，显存需求可能达到 24-48 GB
- 代码相对较新，community support 较少

**GitHub：** https://github.com/facebookresearch/sam3

**结论：** 如果你有充足的 GPU 资源（A100/A6000）且有时间调试，可以尝试。否则 SAM2Long 更稳妥。

---

### 3.4 SeC（竞赛级方案）

**简介：** Segment with Concepts，利用大型视觉语言模型（LVLM）建立对目标对象的深层语义理解，在外观剧变、镜头切换等极端场景下表现出色。

**MOSEv2 成绩：** 测试集 39.7 J&F'（零样本，第 2 名）；与 SAM2Long 做 cascaded ensemble 后达到 86.16 J&F（MOSE v1 测试集）。

**缺点：** 需要 LVLM（如 InternVL），显存需求极大（40 GB+），推理非常慢。

**结论：** 计算资源不足的同学不建议使用。如果有多卡 A100，可以考虑与 SAM2Long 做 ensemble。

---

### 3.5 Cutie（轻量替代方案）

**简介：** Adobe Research 提出的 VOS 模型（CVPR 2024），使用 object-level memory reading 和 query-based object transformer。

**性能：** 在 MOSE v1 上达到 68.3 J&F（+8.7 over XMem），运行速度 ~36 FPS。

**算力：** ~8-12 GB 显存

**缺点：** 在 MOSEv2（更难）上可能不如 SAM2 系列

**结论：** 如果 GPU 显存只有 8 GB，可以作为备选方案。

**GitHub：** https://github.com/hkchengrex/Cutie

---

### 3.6 模型选型总结

| 模型 | 难度 | 显存 | 预期提升（vs 基线 43.25） | 推荐度 |
|------|------|------|--------------------------|--------|
| SAM2.1-B+ | ⭐ | 8 GB | 基线 | 保底 |
| SAM2.1-L+ | ⭐ | 20-24 GB | +0.5~1.0 | ★★★ |
| SAM2Long + B+ | ⭐⭐ | 10-12 GB | +0.5~1.5 | ★★★★ |
| SAM2Long + L+ | ⭐⭐ | 24-30 GB | +1.0~2.5 | ★★★★★ |
| SAM3 | ⭐⭐⭐ | 24-48 GB | +1.5~3.0 | ★★★ |
| SAM3 + Re-Prompt | ⭐⭐⭐⭐ | 48 GB+ | +3.0+ | ★★ |
| Cutie | ⭐⭐ | 8-12 GB | 不确定 | ★★ |

---

## 四、推荐执行方案

### 方案 A：稳妥高分方案（适合 24 GB+ GPU）

1. 用 SAM2.1-B+ 跑通 15 个视频，验证格式，提交拿 43.25 保底分
2. 用 **SAM2Long + SAM2.1-L+** 重跑 15 个视频
3. 合并 + 提交，目标 44+

### 方案 B：资源有限方案（适合 12 GB GPU）

1. 用 SAM2.1-B+ 跑通流程
2. 用 **SAM2Long + SAM2.1-B+** 重跑（显存约 10-12 GB）
3. 调节 SAM2Long 参数（pathway 数、IoU 阈值等）

### 方案 C：冲击高分方案（适合 A100/多卡）

1. SAM2Long-L+ 跑出基线
2. SAM3 做推理，与 SAM2Long 结果逐视频对比
3. 对每个视频取更好的结果
4. 可选：SeC 作为第三路结果做 cascaded ensemble

---

## 五、详细实操步骤

### Step 0：前置准备

#### 硬件要求

- **最低配置：** NVIDIA GPU 8 GB 显存（如 RTX 3060/3070）
- **推荐配置：** NVIDIA GPU 24 GB 显存（如 RTX 3090/4090/A5000）
- **高端配置：** NVIDIA A100 40/80 GB

#### 软件环境

```bash
# 确认 CUDA 和 PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 建议 Python 3.10+, PyTorch 2.1+, CUDA 12.x
```

---

### Step 1：解压作业包

```bash
# 创建工作目录
mkdir -p ~/mosev2_hw && cd ~/mosev2_hw

# 解压作业包
unzip mosev2_作业.zip -d ./data

# 确认目录结构
tree ./data -L 2

# 预期输出：
# data/
# ├── JPEGImages/
# │   ├── video_01/
# │   ├── video_02/
# │   └── ...（共 15 个视频）
# ├── Annotations/
# │   ├── video_01/
# │   ├── video_02/
# │   └── ...（共 15 个视频，每个只有首帧 mask）
# └── Output/
#     ├── existing_video_01/
#     └── ...（共 418 个视频的结果）
```

---

### Step 2：安装 SAM2

```bash
cd ~/mosev2_hw

# 克隆 SAM2 仓库
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# 安装依赖
pip install -e .

# 下载模型权重
cd checkpoints

# 下载 B+ 模型（~300MB，用于跑通流程）
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# 下载 L+ 模型（~900MB，用于提分）
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

cd ~/mosev2_hw
```

---

### Step 3：用 SAM2.1-B+ 跑通基线（保底分）

创建推理脚本 `inference_sam2.py`：

```python
import os
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# ===================== 配置 =====================
SAM2_CHECKPOINT = "./sam2/checkpoints/sam2.1_hiera_base_plus.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
JPEG_DIR = "./data/JPEGImages"
ANNO_DIR = "./data/Annotations"
OUTPUT_DIR = "./output_15videos"
DEVICE = "cuda"
# =================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 构建视频预测器
predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)

# 获取待推理的 15 个视频
video_names = sorted(os.listdir(JPEG_DIR))
print(f"共 {len(video_names)} 个视频待推理")

for vid_idx, video_name in enumerate(video_names):
    print(f"\n[{vid_idx+1}/{len(video_names)}] 处理视频: {video_name}")

    video_dir = os.path.join(JPEG_DIR, video_name)
    anno_dir = os.path.join(ANNO_DIR, video_name)
    out_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(out_dir, exist_ok=True)

    # 读取所有帧文件名
    frame_names = sorted([
        f for f in os.listdir(video_dir)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])

    # 读取首帧 mask（可能包含多个目标，用不同像素值区分）
    first_mask_path = os.path.join(anno_dir, "00000.png")
    first_mask = np.array(Image.open(first_mask_path))

    # 提取所有 object ID（排除背景 0）
    obj_ids = np.unique(first_mask)
    obj_ids = obj_ids[obj_ids != 0].tolist()
    print(f"  检测到 {len(obj_ids)} 个目标: {obj_ids}")

    # 初始化视频状态
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=video_dir)

        # 为每个目标添加首帧 mask
        for obj_id in obj_ids:
            mask = (first_mask == obj_id).astype(np.uint8)
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask
            )

        # 向前传播（propagate）
        for frame_idx, obj_ids_out, masks_out in predictor.propagate_in_video(state):
            # 合成多目标 mask 到一张图
            h, w = masks_out[0].shape[-2:]
            out_mask = np.zeros((h, w), dtype=np.uint8)

            for oid, m in zip(obj_ids_out, masks_out):
                binary = (m[0] > 0.0).cpu().numpy().astype(np.uint8)
                out_mask[binary == 1] = oid

            # 保存为 PNG（palette 模式与标注一致）
            out_img = Image.fromarray(out_mask).convert("P")
            # 使用与首帧 mask 相同的调色板（如果有的话）
            ref_img = Image.open(first_mask_path)
            if ref_img.mode == "P":
                out_img.putpalette(ref_img.getpalette())

            frame_name = frame_names[frame_idx].replace('.jpg', '.png').replace('.jpeg', '.png')
            out_img.save(os.path.join(out_dir, frame_name))

        # 释放显存
        predictor.reset_state(state)

    print(f"  完成，共输出 {len(os.listdir(out_dir))} 帧")

print("\n全部推理完成！")
```

运行：

```bash
cd ~/mosev2_hw
python inference_sam2.py
```

> **注意：** 以上脚本为示例框架。SAM2 的 API 在不同版本可能略有差异，请以官方 README 和 `notebooks/video_predictor_example.ipynb` 为准。运行前务必阅读官方示例。

---

### Step 4：安装并使用 SAM2Long（提分）

```bash
cd ~/mosev2_hw

# 克隆 SAM2Long
git clone https://github.com/Mark12Ding/SAM2Long.git
cd SAM2Long

# 按照 README 安装（通常基于 SAM2 环境即可）
pip install -e .
```

SAM2Long 的推理与 SAM2 类似，但需要指定 `num_pathways` 参数。请参照其 README 中的 VOS 推理示例，关键修改点：

```python
# SAM2Long 的核心区别：设置 memory pathway 数量
# 在配置或推理代码中指定
num_pathways = 3  # 推荐值

# 其余推理流程与 SAM2 一致
# 读取首帧 mask → 初始化 → propagate → 保存结果
```

如果使用 L+ 模型，修改 checkpoint 和 config 路径：

```python
SAM2_CHECKPOINT = "./sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
```

---

### Step 5：合并结果

```bash
cd ~/mosev2_hw

# 创建最终提交目录
mkdir -p submission

# 复制 418 个已有结果
cp -r data/Output/* submission/

# 复制 15 个新推理结果（覆盖任何已有的同名目录）
cp -r output_15videos/* submission/

# 确认总共 433 个视频
echo "视频总数: $(ls submission/ | wc -l)"
# 应输出：视频总数: 433
```

---

### Step 6：检查与打包

```bash
cd ~/mosev2_hw

# 检查每个视频的帧数是否合理
for dir in submission/*/; do
    video=$(basename "$dir")
    count=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    echo "$video: $count frames"
done

# 检查 mask 尺寸是否与原图一致（抽样检查）
python3 -c "
from PIL import Image
import os

# 随机抽查一个新推理的视频
video = os.listdir('output_15videos')[0]
jpeg_dir = f'data/JPEGImages/{video}'
mask_dir = f'output_15videos/{video}'

# 取第一帧对比
jpeg = Image.open(os.path.join(jpeg_dir, sorted(os.listdir(jpeg_dir))[0]))
mask = Image.open(os.path.join(mask_dir, sorted(os.listdir(mask_dir))[0]))

print(f'JPEG size: {jpeg.size}')
print(f'Mask size: {mask.size}')
assert jpeg.size == mask.size, 'ERROR: 尺寸不一致！'
print('OK: 尺寸一致')
"

# 打包提交
cd submission
zip -r ../submission.zip .
cd ..

echo "提交文件大小: $(du -h submission.zip | cut -f1)"
```

---

### Step 7：提交到 Codabench

1. 打开 https://www.codabench.org/competitions/10062/#/participate-tab
2. 用 `cv_{学号}` 账号登录
3. 在 "My Submissions" 页面上传 `submission.zip`
4. 等待评测完成（通常几分钟到十几分钟）
5. 在 "Results" 页面查看分数

---

## 六、提分技巧与注意事项

### 6.1 关键提分手段

1. **换大模型（B+ → L+）：** 最简单的提升，通常提升 0.5-1.0 分
2. **使用 SAM2Long：** 不需要训练，只改推理策略，提升 1-3 分
3. **组合使用：** SAM2Long + L+ 是性价比最高的方案
4. **逐视频检查：** 15 个视频不多，可以可视化查看哪些分割有问题，针对性调参
5. **多次提交：** 每天 20 次上限，每改进一步就提交看效果

### 6.2 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 提交后分数为 0 或很低 | 文件格式错误 | 检查 PNG 是否为 palette 模式，object ID 是否正确 |
| 显存不足 (OOM) | 模型太大或视频太长 | 换 B+ 模型；减小 batch size；使用 `torch.autocast` 开启 bf16/fp16 |
| mask 尺寸不对 | 输出分辨率与原图不一致 | 确保输出 mask 的 H×W 与输入帧完全一致 |
| 某些帧缺失 | propagation 漏帧 | 检查输出帧数是否与 JPEGImages 中的帧数一致 |
| 分数没有提升 | 模型对这 15 个视频的提升有限 | 尝试更好的模型（SAM2Long、SAM3）或调参 |

### 6.3 Mask 格式说明

MOSEv2 的 mask 格式要求：
- **PNG 格式**，8-bit palette mode
- **像素值 = object ID**：背景为 0，第一个目标为 1，第二个为 2，以此类推
- 分辨率必须与对应视频帧完全一致
- 文件名与帧编号一致（`00000.png`, `00001.png`, ...）

### 6.4 可视化检查（推荐）

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_result(jpeg_path, mask_path):
    img = np.array(Image.open(jpeg_path))
    mask = np.array(Image.open(mask_path))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)
    axes[0].set_title("原图")

    axes[1].imshow(mask, cmap="tab10")
    axes[1].set_title("预测 Mask")

    axes[2].imshow(img)
    axes[2].imshow(mask, alpha=0.5, cmap="tab10")
    axes[2].set_title("叠加效果")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# 用法示例
# visualize_result("data/JPEGImages/video_01/00050.jpg",
#                  "output_15videos/video_01/00050.png")
```

---

## 七、进阶方案：使用 SAM3

如果你决定尝试 SAM3，以下是基本步骤：

```bash
# 克隆 SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3

# 安装依赖（注意 CLIP 包的兼容性）
pip install -e .
pip uninstall clip -y
pip install git+https://github.com/ultralytics/CLIP.git

# 下载权重（查看 README 获取最新链接）
```

SAM3 的半监督 VOS 推理需要注意：
- SAM3 支持 mask prompt，可以像 SAM2 一样传入首帧 mask
- 视频 tracker 部分的 API 与 SAM2 类似
- 但可能需要额外配置 concept detection 相关参数
- 建议参考 ICCV 2025 MOSEv2 赛道第 3 名的技术报告了解 Re-Prompting 策略

---

## 八、时间规划建议

| 阶段 | 时间 | 任务 |
|------|------|------|
| Day 1 | 2-3 小时 | 搭建环境 + SAM2.1-B+ 跑通流程 + 提交保底分 |
| Day 2 | 2-3 小时 | 安装 SAM2Long + L+ 重跑 + 提交观察提分 |
| Day 3-4 | 3-4 小时 | 调参优化 / 尝试其他模型 / 多次提交取最高分 |

总计约 8-10 小时即可完成，建议尽早开始，留出 debug 的时间。

---

## 九、参考资料

| 资源 | 链接 |
|------|------|
| MOSEv2 官网 | https://mose.video/ |
| MOSEv2 论文 | https://arxiv.org/abs/2508.05630 |
| SAM2 GitHub | https://github.com/facebookresearch/sam2 |
| SAM2Long GitHub | https://github.com/Mark12Ding/SAM2Long |
| SAM2Long 论文 | https://arxiv.org/abs/2410.16268 |
| SAM3 GitHub | https://github.com/facebookresearch/sam3 |
| SAM3 论文 | https://arxiv.org/abs/2511.16719 |
| Cutie GitHub | https://github.com/hkchengrex/Cutie |
| Codabench 提交 | https://www.codabench.org/competitions/10062/#/participate-tab |
