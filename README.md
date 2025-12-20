# ComfyUI-HAIGC-PSD
**[English](#english) | [中文](#chinese)**

<a name="english"></a>
## English

A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows you to save image batches as layered **PSD (Photoshop)** files. It preserves transparency (RGBA), supports custom backgrounds, and provides a direct download button in the UI.

### Features

- **Layered Export**: Saves an image batch as a single `.psd` file with multiple layers.
- **RGBA Support**: Automatically detects and preserves transparency in input images.
- **Mask Support**: Optional mask input for non-RGBA images.
- **Background Handling**: 
  - Optionally input a `background_image` to set the canvas size and background layer.
  - Auto-alignment of layers.
- **UI Preview & Download**: 
  - Displays a flattened preview of the PSD in the node.
  - **"Download PSD" button** directly in the node interface for one-click access.

### Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/HAIGC/HAIGC-PSD.git
   ```
2. Install the required dependency:
   ```bash
   pip install pytoshop
   ```
   *(Note: If you use ComfyUI Manager, it should handle dependencies automatically via `requirements.txt`)*

3. Restart ComfyUI.

### Usage

The node is located under the category `HAIGC/PSD`.

#### Inputs

| Input Name | Type | Description |
|---|---|---|
| **images** | `IMAGE` | The batch of images to be saved as layers. (Top-most image in batch becomes the top layer). |
| **filename_prefix** | `STRING` | Prefix for the saved filename. (Default: `ComfyUI_PSD`) |
| **masks** | `MASK` (Optional) | Optional batch of masks to apply transparency to `images`. Ignored if `images` are already RGBA. |
| **background_image** | `IMAGE` (Optional) | If connected, this image becomes the **Background** layer and defines the **Canvas Size** of the PSD. |

#### Logic

- **Canvas Size**: 
  - If `background_image` is provided: The PSD canvas size equals the background image size.
  - If not: The PSD canvas size equals the size of the first image in the `images` batch.
- **Transparency**:
  - The node prioritizes the Alpha channel of the input `images`.
  - If no Alpha channel exists, it looks for the `masks` input.
  - If neither exists, the layer is opaque.
- **Layer Order**: 
  - `background_image` is always locked as the bottom "Background" layer.
  - `images` batch is stacked on top (Index 0 is bottom-most of the batch, Index N is top-most).

---

<a name="chinese"></a>
## 中文说明

这是一个 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 的自定义节点，允许您将图片批次（Batch）保存为分层的 **PSD (Photoshop)** 文件。它支持保留透明度（RGBA），支持自定义背景，并在节点界面上提供直接下载按钮。

### 功能特点

- **分层导出**: 将一批图片保存为包含多个图层的单个 `.psd` 文件。
- **RGBA 支持**: 自动检测并保留输入图片的透明通道。
- **遮罩支持**: 为非透明图片提供可选的遮罩（Mask）输入。
- **背景处理**: 
  - 可选输入 `background_image` 来设定画布尺寸并作为背景图层。
  - 图层自动对齐。
- **UI 预览与下载**: 
  - 在节点中显示 PSD 的合成预览图。
  - **"Download PSD" 按钮**: 直接在节点界面一键下载生成的 PSD 文件。

### 安装方法

1. 将本仓库克隆到您的 `ComfyUI/custom_nodes/` 目录：
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/HAIGC/HAIGC-PSD.git
   ```
2. 安装必要的依赖库：
   ```bash
   pip install pytoshop
   ```
   *(注：如果您使用 ComfyUI Manager，它应该会通过 `requirements.txt` 自动处理依赖)*

3. 重启 ComfyUI。

### 使用方法

该节点位于分类 `HAIGC/PSD` 下。

#### 输入说明

| 输入名称 | 类型 | 描述 |
|---|---|---|
| **images** | `IMAGE` | 作为图层保存的图片批次。（批次中的第一张图位于最底层，最后一张位于最顶层）。 |
| **filename_prefix** | `STRING` | 保存文件的前缀名。（默认：`ComfyUI_PSD`） |
| **masks** | `MASK` (可选) | 可选的遮罩批次，用于为 `images` 应用透明度。如果 `images` 已经是 RGBA 格式，则忽略此输入。 |
| **background_image** | `IMAGE` (可选) | 如果连接，该图片将作为 **背景 (Background)** 图层，并决定 PSD 的 **画布尺寸**。 |

#### 逻辑说明

- **画布尺寸**: 
  - 如果提供了 `background_image`: PSD 画布尺寸等于背景图片尺寸。
  - 如果未提供: PSD 画布尺寸等于 `images` 批次中第一张图片的尺寸。
- **透明度**:
  - 节点优先读取输入 `images` 的 Alpha 通道。
  - 如果没有 Alpha 通道，则检查 `masks` 输入。
  - 如果两者都没有，图层将是不透明的。
- **图层顺序**: 
  - `background_image` 始终固定为底部的 "Background" 图层。
  - `images` 批次叠加在背景之上（Index 0 在最下方，Index N 在最上方）。

## Requirements / 依赖

- `pytoshop`
