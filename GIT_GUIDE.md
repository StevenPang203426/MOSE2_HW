# Git 操作指南：本地 → GitHub → 云服务器

## 1. 本地：初始化并推送到 GitHub

```bash
cd D:\AI\MOSE2

# 初始化（如果还没有 .git）
git init

# 添加文件（.gitignore 会自动排除大文件和数据）
git add .
git commit -m "MOSEv2 homework: unified inference script"

# 在 GitHub 创建一个 **私有仓库**（不要勾选 README/gitignore）
# 然后关联远程仓库：
git remote add origin https://github.com/<你的用户名>/MOSE2.git
# 或 SSH：
# git remote add origin git@github.com:<你的用户名>/MOSE2.git

# 推送
git branch -M main
git push -u origin main
```

## 2. 云服务器：拉取代码 + 准备环境

```bash
# 克隆你的仓库
git clone https://github.com/<你的用户名>/MOSE2.git
cd MOSE2

# ===== 环境安装 =====
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv mosev2_env --python 3.10
source mosev2_env/bin/activate

# PyTorch（根据 CUDA 版本）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 其他依赖
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ===== 安装 SAM2Long =====
git clone https://github.com/Mark12Ding/SAM2Long.git
cd SAM2Long
SAM2_BUILD_CUDA=0 uv pip install -e . --no-build-isolation \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..

# ===== 下载模型权重 =====
mkdir -p SAM2Long/checkpoints
# B+ 模型
wget -P SAM2Long/checkpoints/ \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
# L+ 模型（推荐）
wget -P SAM2Long/checkpoints/ \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# ===== 上传作业数据 =====
# 方法1: scp 从本地上传（在本地执行）
# scp -r D:\AI\MOSE2\mosev2_作业 user@server:/path/to/MOSE2/

# 方法2: 如果数据在网盘，直接在服务器下载

# ===== 验证 =====
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2Long OK')"
```

## 3. 云服务器：运行推理

```bash
cd MOSE2

# 把脚本复制到 SAM2Long 目录下运行（因为需要读取 sam2 的 configs/）
cp run_inference.py SAM2Long/
cp -r mosev2_作业 SAM2Long/
cd SAM2Long

# SAM2Long + L+（推荐）
python run_inference.py --strategy sam2long --model large

# 或 SAM2 基线（保底）
python run_inference.py --strategy sam2

# 推理完成后，submission.zip 在当前目录
ls -lh submission.zip
```

## 4. 下载结果并提交

```bash
# 从服务器下载 submission.zip（在本地执行）
scp user@server:/path/to/SAM2Long/submission.zip ./

# 上传到 codabench
# https://www.codabench.org/competitions/10062/#/participate-tab
```

## 常用 Git 命令

```bash
# 查看哪些文件会被追踪
git status

# 修改后提交
git add -A && git commit -m "update inference script" && git push

# 云服务器拉取更新
git pull
```
