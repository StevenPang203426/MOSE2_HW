"""
MOSEv2 作业统一推理脚本
======================
支持 SAM2 基线和 SAM2Long 两种推理策略，通过 --strategy 切换。

============ 环境安装（uv + 清华源） ============

# 0. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 1. 创建虚拟环境
uv venv mosev2_env --python 3.10
source mosev2_env/bin/activate

# 2. 安装 PyTorch（根据 CUDA 版本）
#    CUDA 12.x（推荐）：
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    CUDA 11.8：
#    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装依赖（清华源）
uv pip install numpy>=1.24.4 tqdm>=4.66.1 hydra-core>=1.3.2 \
    iopath>=0.1.10 pillow>=9.4.0 matplotlib opencv-python \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 克隆并安装模型库
#    SAM2 基线：
#      git clone https://github.com/facebookresearch/sam2.git && cd sam2
#    SAM2Long（推荐）：
#      git clone https://github.com/Mark12Ding/SAM2Long.git && cd SAM2Long
SAM2_BUILD_CUDA=0 uv pip install -e . --no-build-isolation \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 下载模型权重
mkdir -p checkpoints
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# 6. 验证
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -c "from sam2.build_sam import build_sam2_video_predictor; print('OK')"

========================================

用法:
    # SAM2 基线（保底分 ~43.25）
    python run_inference.py --strategy sam2

    # SAM2Long + B+（需安装 SAM2Long）
    python run_inference.py --strategy sam2long

    # SAM2Long + L+（推荐，需 24GB+ 显存）
    python run_inference.py --strategy sam2long --model large

    # 调参
    python run_inference.py --strategy sam2long --model large \
        --num_pathway 3 --iou_thre 0.1 --uncertainty 2

    # 仅合并打包（推理已完成）
    python run_inference.py --pack_only

    # 自定义路径
    python run_inference.py --strategy sam2long \
        --jpeg_dir /path/to/JPEGImages \
        --anno_dir /path/to/Annotations \
        --existing_output_dir /path/to/output
"""

import argparse
import os
import shutil
import zipfile
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ===================== 默认路径 =====================
DEFAULTS = {
    "jpeg_dir": "mosev2_作业/homework/JPEGImages",
    "anno_dir": "mosev2_作业/homework/Annotations",
    "existing_output_dir": "mosev2_作业/homework/output",
    "output_dir": "output_15videos",
    "submission_dir": "submission",
    "submission_zip": "submission.zip",
}

# 模型配置
MODEL_CONFIGS = {
    "base": {
        "checkpoint": "checkpoints/sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "name": "SAM2.1-B+",
    },
    "large": {
        "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "name": "SAM2.1-L+",
    },
}

# DAVIS 标准调色板（VOS 评测通用）
DAVIS_PALETTE = [0] * 768
_DAVIS_RAW = (
    b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80"
    b"\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00"
    b"@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80"
)
DAVIS_PALETTE[:len(_DAVIS_RAW)] = list(_DAVIS_RAW)


# ===================== Mask I/O =====================

def load_mask(path):
    """加载 PNG mask，返回 (ndarray uint8, palette list)。
    palette 优先取文件自带的；若无则用 DAVIS 标准调色板。"""
    img = Image.open(path)
    palette = img.getpalette() if img.mode == "P" else None
    mask = np.array(img).astype(np.uint8)
    return mask, palette or DAVIS_PALETTE[:]


def save_mask(path, mask, palette):
    """保存 (H,W) uint8 mask 为 palette PNG。"""
    out = Image.fromarray(mask.astype(np.uint8), mode="P")
    out.putpalette(palette)
    out.save(path)


def combine_per_obj_masks(per_obj_mask, height, width):
    """将 {obj_id: bool_mask} 合成为单张 (H,W) uint8 mask。
    小 ID 优先（覆盖大 ID）。"""
    combined = np.zeros((height, width), dtype=np.uint8)
    for oid in sorted(per_obj_mask.keys(), reverse=True):
        m = per_obj_mask[oid].reshape(height, width)
        combined[m] = oid
    return combined


# ===================== 推理核心 =====================

def build_predictor(args):
    """构建 SAM2 video predictor，返回 (predictor, model_name)。"""
    from sam2.build_sam import build_sam2_video_predictor

    cfg = MODEL_CONFIGS[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hydra_overrides = ["++model.non_overlap_masks=true"]
    predictor = build_sam2_video_predictor(
        config_file=cfg["config"],
        ckpt_path=cfg["checkpoint"],
        apply_postprocessing=False,
        hydra_overrides_extra=hydra_overrides,
        device=device,
    )
    return predictor, cfg["name"], device


def infer_single_video(predictor, video_dir, anno_dir, out_dir, args, device):
    """对单个视频执行推理，保存结果到 out_dir。"""
    os.makedirs(out_dir, exist_ok=True)

    # 帧文件名（无扩展名，排序）
    frame_names = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[-1].lower() in ['.jpg', '.jpeg', '.png']
    ])

    # 读取首帧 mask
    first_mask_path = os.path.join(anno_dir, f"{frame_names[0]}.png")
    first_mask, palette = load_mask(first_mask_path)

    # 提取 object IDs
    obj_ids = np.unique(first_mask)
    obj_ids = obj_ids[obj_ids > 0].tolist()
    print(f"  帧数: {len(frame_names)}, 目标: {obj_ids}")

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=video_dir,
            async_loading_frames=False,
        )

        # SAM2Long: 设置 memory tree 参数
        if args.strategy == "sam2long":
            state['num_pathway'] = args.num_pathway
            state['iou_thre'] = args.iou_thre
            state['uncertainty'] = args.uncertainty

        # 添加首帧 mask prompt
        for obj_id in obj_ids:
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                mask=(first_mask == obj_id),
            )

        # 传播
        height = state["video_height"]
        width = state["video_width"]

        for frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            per_obj = {
                int(oid): (out_mask_logits[i][0] > 0.0).cpu().numpy()
                for i, oid in enumerate(out_obj_ids)
            }
            combined = combine_per_obj_masks(per_obj, height, width)
            save_mask(
                os.path.join(out_dir, f"{frame_names[frame_idx]}.png"),
                combined,
                palette,
            )

        predictor.reset_state(state)

    output_count = len(os.listdir(out_dir))
    if output_count != len(frame_names):
        print(f"  ⚠ 帧数不匹配: 输出 {output_count}, 输入 {len(frame_names)}")
    return output_count


def run_inference(args):
    """对所有待推理视频执行推理。"""
    predictor, model_name, device = build_predictor(args)

    strategy_info = f"策略: {args.strategy}"
    if args.strategy == "sam2long":
        strategy_info += f" (pathway={args.num_pathway}, iou={args.iou_thre}, unc={args.uncertainty})"
    print(f"模型: {model_name} | {strategy_info} | 设备: {device}\n")

    video_names = sorted([
        p for p in os.listdir(args.jpeg_dir)
        if os.path.isdir(os.path.join(args.jpeg_dir, p))
    ])
    print(f"共 {len(video_names)} 个视频\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, vname in enumerate(video_names):
        print(f"[{i+1}/{len(video_names)}] {vname}")
        count = infer_single_video(
            predictor,
            video_dir=os.path.join(args.jpeg_dir, vname),
            anno_dir=os.path.join(args.anno_dir, vname),
            out_dir=os.path.join(args.output_dir, vname),
            args=args,
            device=device,
        )
        print(f"  ✓ {count} 帧\n")

    print(f"推理完成，结果: {args.output_dir}/\n")


# ===================== 验证 =====================

def verify_outputs(args):
    """验证输出 mask 的格式（帧数、尺寸、palette 模式）。"""
    print("验证输出格式...\n")
    errors = []

    for vname in sorted(os.listdir(args.output_dir)):
        out_dir = os.path.join(args.output_dir, vname)
        jpeg_dir = os.path.join(args.jpeg_dir, vname)
        if not os.path.isdir(out_dir):
            continue

        out_frames = sorted(f for f in os.listdir(out_dir) if f.endswith('.png'))
        in_frames = sorted(
            f for f in os.listdir(jpeg_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )

        if len(out_frames) != len(in_frames):
            errors.append(f"{vname}: 帧数 {len(out_frames)} vs {len(in_frames)}")

        if out_frames:
            out_img = Image.open(os.path.join(out_dir, out_frames[0]))
            in_img = Image.open(os.path.join(jpeg_dir, in_frames[0]))
            if out_img.size != in_img.size:
                errors.append(f"{vname}: 尺寸 {out_img.size} vs {in_img.size}")
            if out_img.mode != "P":
                errors.append(f"{vname}: 非 palette 模式 ({out_img.mode})")

        print(f"  ✓ {vname}: {len(out_frames)} 帧")

    print()
    if errors:
        print("问题:")
        for e in errors:
            print(f"  ✗ {e}")
        return False
    print("全部通过!\n")
    return True


# ===================== 合并打包 =====================

def merge_and_pack(args):
    """合并已有结果 + 新结果，打包 submission.zip。"""
    print("合并结果...\n")

    if os.path.exists(args.submission_dir):
        shutil.rmtree(args.submission_dir)
    os.makedirs(args.submission_dir)

    # 复制已有结果
    existing = [d for d in os.listdir(args.existing_output_dir)
                if os.path.isdir(os.path.join(args.existing_output_dir, d))]
    print(f"复制 {len(existing)} 个已有结果...")
    for vname in tqdm(existing, desc="已有结果"):
        shutil.copytree(
            os.path.join(args.existing_output_dir, vname),
            os.path.join(args.submission_dir, vname),
        )

    # 复制新结果（覆盖同名）
    new_videos = [d for d in os.listdir(args.output_dir)
                  if os.path.isdir(os.path.join(args.output_dir, d))]
    print(f"复制 {len(new_videos)} 个新结果...")
    for vname in tqdm(new_videos, desc="新结果"):
        dst = os.path.join(args.submission_dir, vname)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(args.output_dir, vname), dst)

    total = len([d for d in os.listdir(args.submission_dir)
                 if os.path.isdir(os.path.join(args.submission_dir, d))])
    print(f"\n合并: {total} 个视频" + (f" ⚠ (预期 433)" if total != 433 else ""))

    # 打包
    zip_path = args.submission_zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"打包 {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(args.submission_dir):
            for f in sorted(files):
                fp = os.path.join(root, f)
                zf.write(fp, os.path.relpath(fp, args.submission_dir))

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"完成: {zip_path} ({size_mb:.1f} MB)")
    print(f"\n上传: https://www.codabench.org/competitions/10062/#/participate-tab")


# ===================== CLI =====================

def parse_args():
    p = argparse.ArgumentParser(
        description="MOSEv2 作业推理 — 支持 SAM2 / SAM2Long",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 策略
    p.add_argument("--strategy", choices=["sam2", "sam2long"], default="sam2long",
                   help="推理策略 (default: sam2long)")
    p.add_argument("--model", choices=["base", "large"], default="base",
                   help="模型大小: base=B+(8GB), large=L+(24GB+) (default: base)")

    # SAM2Long 参数
    g = p.add_argument_group("SAM2Long 参数 (仅 --strategy sam2long 时生效)")
    g.add_argument("--num_pathway", type=int, default=3, help="路径数 (default: 3)")
    g.add_argument("--iou_thre", type=float, default=0.1, help="IoU 阈值 (default: 0.1)")
    g.add_argument("--uncertainty", type=float, default=2, help="不确定性 (default: 2)")

    # 路径
    g2 = p.add_argument_group("路径配置")
    g2.add_argument("--jpeg_dir", default=DEFAULTS["jpeg_dir"])
    g2.add_argument("--anno_dir", default=DEFAULTS["anno_dir"])
    g2.add_argument("--existing_output_dir", default=DEFAULTS["existing_output_dir"])
    g2.add_argument("--output_dir", default=DEFAULTS["output_dir"])
    g2.add_argument("--submission_dir", default=DEFAULTS["submission_dir"])
    g2.add_argument("--submission_zip", default=DEFAULTS["submission_zip"])

    # 流程控制
    p.add_argument("--pack_only", action="store_true", help="跳过推理，仅合并打包")
    p.add_argument("--no_pack", action="store_true", help="只推理不打包")

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("MOSEv2 作业推理")
    print("=" * 50 + "\n")

    if not args.pack_only:
        run_inference(args)
        verify_outputs(args)

    if not args.no_pack:
        merge_and_pack(args)

    print("\n全部完成!")


if __name__ == "__main__":
    main()
