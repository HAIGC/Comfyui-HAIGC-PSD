import torch
import json
import re

def image_list_to_batch(images):
    if not images:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    normalized = []
    for img in images:
        if not isinstance(img, torch.Tensor):
            continue
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.dim() != 4:
            continue
        normalized.append(img)
    if not normalized:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    
    max_h = max(int(t.shape[1]) for t in normalized)
    max_w = max(int(t.shape[2]) for t in normalized)
    max_c = max(int(t.shape[3]) for t in normalized)
    dtype = normalized[0].dtype
    device = normalized[0].device
    total_b = sum(int(t.shape[0]) for t in normalized)
    
    out = torch.zeros((total_b, max_h, max_w, max_c), dtype=dtype, device=device)
    cursor = 0
    for t in normalized:
        b, h, w, c = (int(t.shape[0]), int(t.shape[1]), int(t.shape[2]), int(t.shape[3]))
        out[cursor:cursor + b, 0:h, 0:w, 0:c] = t
        cursor += b
    return out

class HAIGC_LayerFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图层图像": ("IMAGE", ),
                "选择索引": ("STRING", {"default": "", "multiline": False, "placeholder": "e.g. 1, 2, 5-7 (为空则全选)"}),
                "最小宽度": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "最小高度": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "最大宽度": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "最大高度": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "反选": ("BOOLEAN", {"default": False, "label": "反选"}),
            },
            "optional": {
                "图层信息": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("过滤图像", "过滤信息")
    OUTPUT_IS_LIST = (True, False)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "HAIGC/PSD"

    def process(self, 图层图像, 选择索引, 最小宽度, 最小高度, 最大宽度, 最大高度, 反选, 图层信息=None):
        # Unwrap parameters (since INPUT_IS_LIST=True, all inputs are lists)
        images_in = 图层图像
        
        def get_val(val, default):
            if isinstance(val, list):
                if len(val) > 0:
                    return val[0]
                return default
            return val

        selected_indices = get_val(选择索引, "")
        min_width = get_val(最小宽度, 0)
        min_height = get_val(最小高度, 0)
        max_width = get_val(最大宽度, 0)
        max_height = get_val(最大高度, 0)
        invert_selection = get_val(反选, False)
        layer_info = get_val(图层信息, None)

        # Handle List or Tensor input logic
        # If input is [BatchTensor] (standard batch), treat as tensor mode
        # If input is [T1, T2, ...] (list of layers), treat as list mode
        if len(images_in) == 1 and isinstance(images_in[0], torch.Tensor) and images_in[0].dim() == 4 and images_in[0].shape[0] > 1:
            images = images_in[0]
            batch_size = images.shape[0]
            is_list = False
        else:
            images = images_in
            batch_size = len(images)
            is_list = True
        
        # Parse Layer Info
        layer_data_list = []
        if layer_info:
            try:
                layer_data_list = json.loads(layer_info)
            except:
                pass
        
        # Parse Indices
        selected_set = set()
        has_explicit_indices = False
        if selected_indices.strip():
            has_explicit_indices = True
            # Replace Chinese comma with English comma
            selected_indices = selected_indices.replace("，", ",")
            # Split by comma or space
            parts = re.split(r'[,\s]+', selected_indices)
            for part in parts:
                if not part:
                    continue
                if '-' in part:
                    try:
                        start, end = map(int, part.split('-'))
                        # Assume 1-based input from user, convert to 0-based
                        for i in range(start, end + 1):
                            selected_set.add(i - 1)
                    except ValueError:
                        pass
                else:
                    try:
                        idx = int(part)
                        selected_set.add(idx - 1)
                    except ValueError:
                        pass
        else:
            # If empty, assume all selected initially
            selected_set = set(range(batch_size))
        
        # Filter Logic
        final_indices = []
        
        for i in range(batch_size):
            # 1. Index Selection Check
            is_selected = i in selected_set
            
            if invert_selection and has_explicit_indices:
                is_selected = not is_selected
            
            if not is_selected:
                continue

            # 2. Size Filter Check
            # Use layer_info if available to get REAL size, otherwise use tensor size
            w, h = 0, 0
            real_w, real_h = 0, 0
            
            if i < len(layer_data_list):
                info = layer_data_list[i]
                real_w = int(info.get("width", 0))
                real_h = int(info.get("height", 0))
                w, h = real_w, real_h
            else:
                # Fallback to tensor shape if no info
                if is_list:
                    img_item = images[i]
                    if img_item.dim() == 4:
                        h, w = img_item.shape[1], img_item.shape[2]
                    else:
                        h, w = img_item.shape[0], img_item.shape[1]
                else:
                    h, w = images.shape[1], images.shape[2]
            
            size_pass = True
            if w < min_width or h < min_height:
                size_pass = False
            if max_width > 0 and w > max_width:
                size_pass = False
            if max_height > 0 and h > max_height:
                size_pass = False

            if invert_selection and (not has_explicit_indices):
                size_pass = not size_pass

            if not size_pass:
                continue
                
            final_indices.append(i)

        if not final_indices:
            # Return empty batch (1x1 black pixel) to prevent crash
            empty_img = torch.zeros((1, 1, 1, 3))
            return ([empty_img], "[]")

        # Construct Output with Re-batching
        filtered_images_list = []
        filtered_data_list = []
        
        for i in final_indices:
            # Get original image slice
            if is_list:
                img = images[i]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
            else:
                img = images[i:i+1] # Keep 4D [1, H, W, C]
            
            # Crop to actual size if layer info is available
            # This removes the huge padding introduced by batching with large layers
            if i < len(layer_data_list):
                info = layer_data_list[i]
                real_w = int(info.get("width", 0))
                real_h = int(info.get("height", 0))
                
                if real_w > 0 and real_h > 0:
                    curr_h, curr_w = img.shape[1], img.shape[2]
                    # Crop if real size is smaller than padded size
                    if real_w < curr_w or real_h < curr_h:
                        img = img[:, :real_h, :real_w, :]
                
                filtered_data_list.append(info)
            else:
                filtered_data_list.append({})
            
            filtered_images_list.append(img)
            
        # Return list directly to preserve original sizes
        # filtered_tensor = image_list_to_batch(filtered_images_list)
        
        return (filtered_images_list, json.dumps(filtered_data_list, indent=2, ensure_ascii=False))

NODE_CLASS_MAPPINGS = {
    "HAIGC_LayerFilter": HAIGC_LayerFilter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HAIGC_LayerFilter": "图层过滤 (HAIGC)"
}
