# Comfyui-HAIGC-PSD

中文 | [English](#english)

一个用于 ComfyUI 的 PSD 工作流插件：支持 PSD 读取（分层输出）、PSD 组装保存（带图层/遮罩/混合模式/排版）、图层过滤与迭代、以及提示词批量队列等辅助节点。

## 中文

### 功能概览

- PSD 加载：从 PSD 文件/目录读取合成图与图层图像，并输出图层名称与元信息（JSON）。
- PSD 保存：把 ComfyUI 的图像/遮罩/图层数据组装为 PSD 并保存到输出目录，同时提供下载按钮。
- PSD 图层：把图像（含 batch/列表）转换为 PSD 图层数据，支持不透明度、混合模式、排版方向与间距。
- 多图串联输入：把多张图以“串联”的方式累积为图像序列并携带图层名，便于后续 PSD 图层/PSD 保存直接使用。
- 图层迭代：把“分层输出”的图层图像按张输出（列表输出），可配合后续逐层处理节点。
- 图层过滤：按索引与尺寸条件筛选图层，支持反选逻辑。
- 合并 PSD 单元：把多个“PSD 单元/批次”按方向重新排版（向上/下/左/右）并设置间距。
- 提示词批量队列：把多段提示词分割为列表，让工作流按提示词数量自动运行多次。

### 安装

1. 将本项目放入：
   - `ComfyUI/custom_nodes/Comfyui-HAIGC-PSD`
2. 安装依赖（在 ComfyUI 的 Python 环境中执行）：

```bash
pip install -r requirements.txt
```

依赖列表见 [requirements.txt](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/requirements.txt)：
- `psd-tools`：读取 PSD
- `pytoshop`：写入 PSD

3. 重启 ComfyUI。

### 节点列表（按分类）

#### HAIGC/PSD

- PSD保存 (HAIGC)
  - 作用：将图像/遮罩/图层数据保存为 PSD。
  - 输入：
    - 文件名前缀（必填）
    - 图像（可选）：支持单图、batch、或来自“多图串联输入”的图像序列
    - 遮罩（可选）：MASK（batch 可选）
    - 图层数据（可选）：来自“PSD图层 (HAIGC)”等节点的 PSD_LAYERS
    - 背景配置（可选）：来自“PSD背景 (HAIGC)”
  - 输出：无（输出节点），保存到 ComfyUI 输出目录，并在节点上提供 PSD 下载按钮
  - 代码： [HAIGC_SavePSD](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L1378-L1550)

- PSD图层 (HAIGC)
  - 作用：把图像转换为 PSD 图层数据（PSD_LAYERS），用于后续合成与保存。
  - 常用参数：
    - 每批合成方式：所有单元合成 / 批次展开为图层 / 单个单元合成
    - 不透明度、混合模式
    - 排版方式：叠加/向右/向左/向下/向上
    - 间距、间距颜色、匹配上一层大小
    - 遮罩（可选）
  - 输出：图层数据（PSD_LAYERS）
  - 代码： [HAIGC_Layer](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L548-L1376)

- 多图串联输入 (HAIGC)
  - 作用：把多张图累积为图像序列，并可附带图层名，用于 PSD 图层/保存节点。
  - 输入：
    - 图像：支持单图、batch、或图像列表
    - 图层名：本次追加的图层名（对追加的每张图都会应用）
    - 上一层（可选）：用于串联累积
  - 输出：
    - 图像序列（IMAGE）：带 names 的列表（内部为 NamedImageList）
    - 图层连接（HAIGC_IMAGE_STACK）：继续串联用的句柄
  - 代码： [HAIGC_ImageSequence](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L2342-L2408)

- PSD背景 (HAIGC)
  - 作用：为 PSD 保存提供背景配置（纯色或图像），可选择锁定与适应模式。
  - 输入：颜色、不透明度、锁定、适应模式、图像（可选）
  - 输出：背景配置
  - 代码： [HAIGC_PSD_Background](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L2410-L2437)

- PSD加载 (HAIGC)
  - 作用：读取 PSD，输出合成图与分层图像列表，并输出图层名称与元信息 JSON。
  - 输入：
    - PSD文件路径 或 目录路径（二选一）
    - 图层输出模式：
      - 按PSD原位置：会把图层按 PSD 位置贴回到画布（可能带 padding）
      - 完整原图层：输出每个图层的原始裁剪尺寸
    - 图层索引：-1 表示输出所有图层；>=0 表示只输出指定图层
  - 输出：
    - 合成图像（IMAGE）
    - 图层图像（IMAGE 列表输出）
    - 图层名称（STRING，按文件分组）
    - 图层信息（STRING，JSON）
  - 交互：节点上提供“选择PSD文件”按钮，支持分片上传超大 PSD
  - 代码： [HAIGC_LoadPSD](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L2440-L2807) / [前端扩展](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/js/haigc_psd_ui.js#L58-L172)

- 图层迭代 (HAIGC)
  - 作用：把分层输出（IMAGE 列表或 batch）按张迭代输出，便于逐层处理。
  - 输入：图层图像（IMAGE），图层信息（可选，JSON）
  - 输出：图像（IMAGE 列表输出）
  - 代码： [HAIGC_LayerIterator](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L2809-L2889)

- 合并PSD单元 (HAIGC)
  - 作用：把多个 PSD 单元（如多个批次/多套图层）重新排版合并为一个图层数据，便于一次保存为 PSD。
  - 输入：图层数据、排版方式（向上/下/左/右）、间距、间距颜色
  - 输出：图层数据（PSD_LAYERS）
  - 代码： [HAIGC_CombineUnits](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L3179-L3456)

- 图层过滤 (HAIGC)
  - 作用：按索引、尺寸范围筛选图层（适合配合 PSD加载 的分层输出）。
  - 输入：
    - 选择索引：如 `1,2,5-7`（为空表示“全选”）
    - 最小/最大宽高：0 表示不限制
    - 反选：
      - 选择索引有填写时：反选索引集合
      - 选择索引为空时：反选“尺寸过滤结果”
  - 输出：过滤图像（IMAGE 列表输出）、过滤信息（JSON）
  - 代码： [HAIGC_LayerFilter](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_layer_filter.py#L35-L224)

#### HAIGC/高级功能

- 提示词批量队列
  - 作用：将多段提示词分割成列表，让工作流根据数量自动运行多次。
  - 分割方式：按行 / 按段落 / 按分隔符
  - 可选：跳过空白、去除首尾空格、随机顺序与随机种子、前后缀、最少字符数
  - 输出：提示词列表（STRING 列表输出）、总数量、队列信息
  - 代码： [PromptBatchQueue](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/advanced_features.py#L10-L115)

### 推荐工作流示例

#### 1) 生成图像并保存 PSD（最常用）

1. 生成 IMAGE（例如 KSampler -> VAE Decode）
2. 使用 “PSD图层 (HAIGC)” 生成图层数据（可设置排版方式/混合模式/遮罩等）
3. 可选：用 “PSD背景 (HAIGC)” 设置背景色/背景图
4. 使用 “PSD保存 (HAIGC)” 输出 PSD 文件

#### 2) 读取 PSD 并筛选/迭代图层

1. “PSD加载 (HAIGC)” 输出 “图层图像 + 图层信息”
2. “图层过滤 (HAIGC)” 按索引/尺寸过滤
3. “图层迭代 (HAIGC)” 将过滤结果逐层输出给后续处理节点

#### 3) 多图串联输入（便于快速堆栈）

1. 多次使用 “多图串联输入 (HAIGC)” 串联图像与图层名
2. 将“图像序列”直接接到 “PSD图层 (HAIGC)” 或 “PSD保存 (HAIGC)” 的 “图像”端口

### 常见问题（FAQ）

- 提示“缺少依赖：psd-tools”
  - 请确认已在 ComfyUI 的 Python 环境执行 `pip install -r requirements.txt`，并重启 ComfyUI。

- 预览图像为空/不显示
  - ComfyUI 预览通常需要 RGB 3 通道输出；若你在其他节点里自行处理 PIL/张量，请确保输出为 `[B,H,W,3]`。

- 分层输出尺寸不一致
  - “按PSD原位置”模式可能带 padding（贴回画布）；“完整原图层”会输出裁剪后的原始图层尺寸。

### 联系方式

- 微信：HAIGC1994

---

## English

### Overview

- Load PSD: read a PSD file/folder, output composite image + per-layer images, layer names and JSON metadata.
- Save PSD: assemble images/masks/layer data into a PSD saved under ComfyUI output, with a download button.
- PSD Layer builder: convert ComfyUI images (batch/list) into PSD layer data with opacity, blend mode and layout.
- Image Sequence (chaining): accumulate multiple images into a named sequence for layer/save nodes.
- Layer Iterator: iterate layer outputs as a list for per-layer processing.
- Layer Filter: select layers by indices and size constraints, with invert logic.
- Combine PSD Units: layout multiple “units/batches” into a single PSD layer set.
- Prompt Batch Queue: split multi-line prompts into a list and let the workflow run multiple times.

### Installation

1. Put this repo into:
   - `ComfyUI/custom_nodes/Comfyui-HAIGC-PSD`
2. Install dependencies (in ComfyUI’s Python environment):

```bash
pip install -r requirements.txt
```

Dependencies in [requirements.txt](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/requirements.txt):
- `psd-tools` for reading PSD
- `pytoshop` for writing PSD

3. Restart ComfyUI.

### Nodes (by category)

#### HAIGC/PSD

- PSD Save (HAIGC)
  - Assemble and save PSD from IMAGE/MASK/PSD_LAYERS (+ optional background config). Includes a “Download PSD” button.
  - Source: [HAIGC_SavePSD](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L1378-L1550)

- PSD Layer (HAIGC)
  - Build PSD layer data from ComfyUI images. Supports opacity, blend mode, layout direction and spacing.
  - Source: [HAIGC_Layer](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L548-L1376)

- Image Sequence (HAIGC)
  - Chain/accumulate images into a named sequence. Accepts single image, batch tensor, or image list.
  - Source: [HAIGC_ImageSequence](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L2342-L2408)

- PSD Background (HAIGC)
  - Provide background config (color/image, opacity, lock, adapt mode) for PSD saving.
  - Source: [HAIGC_PSD_Background](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L2410-L2437)

- PSD Load (HAIGC)
  - Load PSD from a file or folder. Outputs composite image, per-layer images, names and JSON metadata.
  - Includes an “Upload PSD” button with chunked upload for large PSD files.
  - Source: [HAIGC_LoadPSD](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L2440-L2807) / [UI extension](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/js/haigc_psd_ui.js#L58-L172)

- Layer Iterator (HAIGC)
  - Iterate layer outputs (list/batch) as a list of images, optionally cropping based on metadata.
  - Source: [HAIGC_LayerIterator](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L2809-L2889)

- Combine PSD Units (HAIGC)
  - Layout multiple units/batches into one PSD_LAYERS with direction and spacing.
  - Source: [HAIGC_CombineUnits](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_psd.py#L3179-L3456)

- Layer Filter (HAIGC)
  - Filter layers by indices and size range.
  - Invert behavior:
    - When indices are provided: invert selected indices
    - When indices are empty: invert the size-filter result
  - Source: [HAIGC_LayerFilter](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/haigc_layer_filter.py#L35-L224)

#### HAIGC/高级功能

- Prompt Batch Queue
  - Split prompt text into a list and let the workflow run multiple times.
  - Source: [PromptBatchQueue](file:///e:/AHAIGC_Comfyui/ComfyUI/custom_nodes/Comfyui-HAIGC-PSD/advanced_features.py#L10-L115)

### Contact

- WeChat: HAIGC1994

