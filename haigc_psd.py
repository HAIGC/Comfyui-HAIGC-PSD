import os
import torch
import numpy as np
import struct
import base64
import io
import math
import json
import re
import uuid
from PIL import Image, ImageFilter, ImageChops
import folder_paths
import pytoshop
from pytoshop import enums
from pytoshop.user import nested_layers
from pytoshop.tagged_block import GenericTaggedBlock

try:
    from psd_tools import PSDImage
except Exception:
    PSDImage = None

BG_CONFIG_TYPE = "HAIGC_BG_CONFIG"
HAIGC_LAYER_BATCHES_KEY = "__haigc_psd_layer_batches__"

def get_blend_mode(mode_name):
    if not mode_name:
        return enums.BlendMode.normal
    
    name = str(mode_name).strip()
    
    mode_map = {
        "normal": enums.BlendMode.normal,
        "multiply": enums.BlendMode.multiply,
        "screen": enums.BlendMode.screen,
        "overlay": enums.BlendMode.overlay,
        "darken": enums.BlendMode.darken,
        "lighten": enums.BlendMode.lighten,
        "color_dodge": enums.BlendMode.color_dodge,
        "color_burn": enums.BlendMode.color_burn,
        "hard_light": enums.BlendMode.hard_light,
        "soft_light": enums.BlendMode.soft_light,
        "difference": enums.BlendMode.difference,
        "exclusion": enums.BlendMode.exclusion,
        "hue": enums.BlendMode.hue,
        "saturation": enums.BlendMode.saturation,
        "color": enums.BlendMode.color,
        "luminosity": enums.BlendMode.luminosity,
        
        # Chinese Mappings
        "正常": enums.BlendMode.normal,
        "正片叠底": enums.BlendMode.multiply,
        "滤色": enums.BlendMode.screen,
        "叠加": enums.BlendMode.overlay,
        "变暗": enums.BlendMode.darken,
        "变亮": enums.BlendMode.lighten,
        "颜色减淡": enums.BlendMode.color_dodge,
        "颜色加深": enums.BlendMode.color_burn,
        "强光": enums.BlendMode.hard_light,
        "柔光": enums.BlendMode.soft_light,
        "差值": enums.BlendMode.difference,
        "排除": enums.BlendMode.exclusion,
        "色相": enums.BlendMode.hue,
        "饱和度": enums.BlendMode.saturation,
        "颜色": enums.BlendMode.color,
        "明度": enums.BlendMode.luminosity,
    }
    
    return mode_map.get(name, enums.BlendMode.normal)

def hex_to_rgb(hex_color):
    if not hex_color:
        return None
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return None
    return None

def pil_to_image_tensor(pil_img):
    img = pil_img.convert("RGB")
    np_img = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).unsqueeze(0)

def pil_to_mask_tensor(pil_img):
    img = pil_img.convert("RGBA")
    alpha = np.array(img)[:, :, 3].astype(np.float32) / 255.0
    return torch.from_numpy(alpha)

def psd_blend_mode_to_name(blend_mode):
    if blend_mode is None:
        return "normal"
    if hasattr(blend_mode, "name"):
        return str(blend_mode.name).lower()
    value = str(blend_mode).strip().lower()
    if value.startswith("blendmode."):
        value = value.split(".", 1)[1]
    return value or "normal"

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

def is_group_like(node):
    attr = getattr(node, "is_group", None)
    if attr is not None:
        return attr() if callable(attr) else bool(attr)
    if hasattr(node, "__iter__") and not hasattr(node, "bbox"):
        return True
    return False

def serialize_object(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize_object(x) for x in obj]
    if hasattr(obj, "items"): # Dict-like (Descriptor)
        try:
            return {str(k): serialize_object(v) for k, v in obj.items()}
        except Exception:
            return str(obj)
    if hasattr(obj, "name"): # Enum-like
        return str(obj.name)
    # Fallback for other psd-tools types like UnitFloat
    if hasattr(obj, "value"):
        return serialize_object(obj.value)
    return str(obj)

def get_layer_effects(layer):
    if not hasattr(layer, "effects"):
        return None
    try:
        eff = layer.effects
        if not eff:
            return None
        
        res = {}
        effect_types = ["drop_shadow", "inner_shadow", "outer_glow", "inner_glow", 
                        "bevel_emboss", "satin", "color_overlay", "gradient_overlay", 
                        "pattern_overlay", "stroke"]
        
        enabled_any = False
        for et in effect_types:
            val = getattr(eff, et, None)
            if val:
                # val is usually a list of descriptors
                res[et] = serialize_object(val)
                enabled_any = True
                
        if not enabled_any:
            return None
        return res
    except Exception:
        return None

def apply_layer_styles(image, effects):
    if not image or not effects:
        return image, 0, 0
    
    # Collect all effects that need to be rendered behind the layer
    # Currently supporting: Drop Shadow, Outer Glow
    
    render_list = []
    
    # 1. Drop Shadow
    shadows = effects.get("drop_shadow")
    if shadows:
        if not isinstance(shadows, list):
            shadows = [shadows]
        for s in shadows:
            if s.get("enabled", True):
                s["_type"] = "shadow"
                render_list.append(s)

    # 2. Outer Glow
    glows = effects.get("outer_glow")
    if glows:
        if not isinstance(glows, list):
            glows = [glows]
        for g in glows:
            if g.get("enabled", True):
                g["_type"] = "glow"
                render_list.append(g)

    if not render_list:
        return image, 0, 0
        
    original_w, original_h = image.size
    min_x, min_y = 0, 0
    max_x, max_y = original_w, original_h
    
    processed_effects = []
    
    for eff in render_list:
        eff_type = eff.get("_type", "shadow")
        
        opacity = eff.get("opacity", 75.0) # Percent
        color_data = eff.get("color", {})
        blur = eff.get("blur", 5.0) # Size
        
        dx, dy = 0, 0
        
        if eff_type == "shadow":
            distance = eff.get("distance", 5.0)
            angle = eff.get("angle", 120.0)
            # use_global = eff.get("useGlobalLight", False) # Ignored
            rad = math.radians(angle)
            dx = -distance * math.cos(rad)
            dy = distance * math.sin(rad)
        
        # Heuristic for padding: 3 * sigma + shift
        pad = int(blur * 3) + 5
        
        sx = int(dx)
        sy = int(dy)
        
        processed_effects.append({
            "type": eff_type,
            "dx": sx,
            "dy": sy,
            "blur": blur,
            "color": color_data,
            "opacity": opacity
        })
        
        min_x = min(min_x, sx - pad)
        min_y = min(min_y, sy - pad)
        max_x = max(max_x, original_w + sx + pad)
        max_y = max(max_y, original_h + sy + pad)
        
    new_w = max_x - min_x
    new_h = max_y - min_y
    
    off_x = -min_x
    off_y = -min_y
    
    # Avoid creating huge images if something is wrong
    if new_w > 10000 or new_h > 10000:
        return image, 0, 0
        
    final_img = Image.new("RGBA", (new_w, new_h), (0,0,0,0))
    
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    src_alpha = image.split()[3]
    
    # Render effects (Bottom to Top? In PS, Shadow is below Glow usually? Or depends on order?)
    # Usually Drop Shadow is at the very bottom.
    # Outer Glow is above Drop Shadow.
    # We sort: shadow first, then glow.
    
    processed_effects.sort(key=lambda x: 0 if x["type"] == "shadow" else 1)
    
    for s in processed_effects:
        shad_canvas = Image.new("L", (new_w, new_h), 0)
        paste_x = int(off_x + s["dx"])
        paste_y = int(off_y + s["dy"])
        
        try:
            shad_canvas.paste(src_alpha, (paste_x, paste_y))
        except Exception:
            continue
            
        blur_radius = s["blur"]
        if blur_radius > 0:
            shad_canvas = shad_canvas.filter(ImageFilter.GaussianBlur(blur_radius))
            
        c_data = s["color"]
        r, g, b = 0, 0, 0
        if isinstance(c_data, dict):
            r = int(c_data.get("Rd  ", c_data.get("Red", 0)))
            g = int(c_data.get("Grn ", c_data.get("Green", 0)))
            b = int(c_data.get("Bl  ", c_data.get("Blue", 0)))
        
        op_val = s["opacity"]
        alpha_scale = op_val / 100.0
        
        # Apply opacity
        shad_canvas = shad_canvas.point(lambda i: int(i * alpha_scale))
        
        effect_rgba = Image.new("RGBA", (new_w, new_h), (r, g, b, 0))
        effect_rgba.putalpha(shad_canvas)
        
        final_img = Image.alpha_composite(final_img, effect_rgba)
        
    final_img.paste(image, (int(off_x), int(off_y)), image)
    
    return final_img, int(off_x), int(off_y)

def get_layer_bbox(layer):
    bbox = getattr(layer, "bbox", None)
    if bbox is not None:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            left, top, right, bottom = bbox
            return int(left), int(top), int(right), int(bottom)
        for keys in (("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")):
            if all(hasattr(bbox, k) for k in keys):
                left = int(getattr(bbox, keys[0]))
                top = int(getattr(bbox, keys[1]))
                right = int(getattr(bbox, keys[2]))
                bottom = int(getattr(bbox, keys[3]))
                return left, top, right, bottom
        try:
            left, top, right, bottom = bbox
            return int(left), int(top), int(right), int(bottom)
        except Exception:
            pass
    offset = getattr(layer, "offset", None)
    size = getattr(layer, "size", None)
    try:
        if offset is not None and size is not None:
            left, top = offset
            w, h = size
            left = int(left)
            top = int(top)
            right = left + int(w)
            bottom = top + int(h)
            return left, top, right, bottom
    except Exception:
        pass
    return None

def collect_leaf_layers(root_layer, include_hidden=False):
    collected = []
    def walk(node):
        if is_group_like(node):
            for child in list(node):
                walk(child)
            return
        if (not include_hidden) and (not getattr(node, "visible", True)):
            return
        collected.append(node)

    if is_group_like(root_layer):
        for child in list(root_layer):
            walk(child)
    else:
        walk(root_layer)
    return collected

def get_lum(rgb):
    return 0.3 * rgb[:,:,0] + 0.59 * rgb[:,:,1] + 0.11 * rgb[:,:,2]

def clip_color(rgb):
    l = get_lum(rgb)
    n = np.min(rgb, axis=2)
    x = np.max(rgb, axis=2)
    
    # n < 0
    mask_n = n < 0
    if np.any(mask_n):
        l_masked = l[mask_n]
        n_masked = n[mask_n]
        denom = l_masked - n_masked
        denom[denom == 0] = 1e-6
        
        l_exp = l_masked[:, None]
        denom_exp = denom[:, None]
        
        rgb_masked = rgb[mask_n]
        rgb[mask_n] = l_exp + (((rgb_masked - l_exp) * l_exp) / denom_exp)

    # x > 1
    x = np.max(rgb, axis=2)
    mask_x = x > 1
    if np.any(mask_x):
        l_masked = l[mask_x]
        x_masked = x[mask_x]
        denom = x_masked - l_masked
        denom[denom == 0] = 1e-6
        
        l_exp = l_masked[:, None]
        denom_exp = denom[:, None]
        
        rgb_masked = rgb[mask_x]
        rgb[mask_x] = l_exp + (((rgb_masked - l_exp) * (1 - l_exp)) / denom_exp)
        
    return rgb

def set_lum(rgb, l):
    d = l - get_lum(rgb)
    rgb = rgb + d[:,:,None]
    return clip_color(rgb)

def get_sat(rgb):
    return np.max(rgb, axis=2) - np.min(rgb, axis=2)

def set_sat(rgb, s):
    # rgb: (H, W, 3)
    # s: (H, W)
    
    idx = np.argsort(rgb, axis=2)
    sorted_rgb = np.take_along_axis(rgb, idx, axis=2)
    
    c_min = sorted_rgb[:,:,0]
    c_mid = sorted_rgb[:,:,1]
    c_max = sorted_rgb[:,:,2]
    
    mask_gt = c_max > c_min
    
    new_mid = np.zeros_like(c_mid)
    new_max = np.zeros_like(c_max)
    
    if np.any(mask_gt):
        denom = c_max[mask_gt] - c_min[mask_gt]
        denom[denom == 0] = 1e-6
        
        new_mid[mask_gt] = ((c_mid[mask_gt] - c_min[mask_gt]) * s[mask_gt]) / denom
        new_max[mask_gt] = s[mask_gt]
        
    new_min = np.zeros_like(c_min)
    
    res_sorted = np.stack([new_min, new_mid, new_max], axis=2)
    
    res = np.zeros_like(rgb)
    np.put_along_axis(res, idx, res_sorted, axis=2)
    
    return res

def blend_numpy(bg, fg, mode, opacity):
    # bg, fg: (H, W, 4) float 0-1
    # mode: string
    # opacity: float 0-1
    
    alpha_fg = fg[:,:,3:] * opacity
    alpha_bg = bg[:,:,3:]
    
    rgb_fg = fg[:,:,:3]
    rgb_bg = bg[:,:,:3]
    
    # Helper for blend math
    def get_blended_rgb(b, s, m):
        if m in ["normal", "正常"]:
            return s
        elif m in ["multiply", "正片叠底"]:
            return b * s
        elif m in ["screen", "滤色"]:
            return 1 - (1 - b) * (1 - s)
        elif m in ["overlay", "叠加"]:
            mask = b < 0.5
            res = np.zeros_like(b)
            res[mask] = 2 * b[mask] * s[mask]
            res[~mask] = 1 - 2 * (1 - b[~mask]) * (1 - s[~mask])
            return res
        elif m in ["soft_light", "柔光"]:
            # Photoshop formula approximation
            mask = s < 0.5
            res = np.zeros_like(b)
            res[mask] = 2 * b[mask] * s[mask] + (b[mask]**2) * (1 - 2 * s[mask])
            res[~mask] = 2 * b[~mask] * (1 - s[~mask]) + np.sqrt(np.maximum(b[~mask], 0)) * (2 * s[~mask] - 1)
            return res
        elif m in ["hard_light", "强光"]:
             # Overlay with b and s swapped
            mask = s < 0.5
            res = np.zeros_like(b)
            res[mask] = 2 * s[mask] * b[mask]
            res[~mask] = 1 - 2 * (1 - s[~mask]) * (1 - b[~mask])
            return res
        elif m in ["darken", "变暗"]:
            return np.minimum(b, s)
        elif m in ["lighten", "变亮"]:
            return np.maximum(b, s)
        elif m in ["color_dodge", "颜色减淡"]:
            # b / (1 - s)
            denom = 1 - s
            denom[denom == 0] = 1e-6 # Avoid div zero
            return np.minimum(1, b / denom)
        elif m in ["color_burn", "颜色加深"]:
            # 1 - (1 - b) / s
            s_safe = s.copy()
            s_safe[s_safe == 0] = 1e-6
            return 1 - np.minimum(1, (1 - b) / s_safe)
        elif m in ["difference", "差值"]:
            return np.abs(b - s)
        elif m in ["exclusion", "排除"]:
            return b + s - 2 * b * s
        elif m in ["hue", "色相"]:
             return set_lum(set_sat(s, get_sat(b)), get_lum(b))
        elif m in ["saturation", "饱和度"]:
             return set_lum(set_sat(b, get_sat(s)), get_lum(b))
        elif m in ["color", "颜色"]:
             return set_lum(s, get_lum(b))
        elif m in ["luminosity", "明度"]:
             return set_lum(b, get_lum(s))
        else:
            return s # Fallback normal

    blended_rgb = get_blended_rgb(rgb_bg, rgb_fg, mode)
    
    c_src = rgb_fg
    c_dst = rgb_bg
    c_blend = blended_rgb
    
    term1 = (1 - alpha_bg) * alpha_fg * c_src
    term2 = (1 - alpha_fg) * alpha_bg * c_dst
    term3 = alpha_bg * alpha_fg * c_blend
    
    out_rgb_premul = term1 + term2 + term3
    out_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
    
    # Avoid div by zero
    mask_zero = out_alpha == 0
    
    # mask_zero is (H, W, 1), but we want to index (H, W, 3).
    # Squeeze the last dim to get (H, W) boolean mask
    mask_sq = mask_zero[:, :, 0]
    
    out_rgb = np.zeros_like(out_rgb_premul)
    out_rgb[~mask_sq] = out_rgb_premul[~mask_sq] / out_alpha[~mask_sq]
    
    # Concatenate
    return np.concatenate([out_rgb, out_alpha], axis=2)

class _HAIGC_NoBatchPatchMeta(type):
    def __setattr__(cls, name, value):
        if name in ("add_layer", "INPUT_TYPES"):
            mod = getattr(value, "__module__", "")
            if isinstance(mod, str) and mod.startswith("haigc_batch"):
                return
        return super().__setattr__(name, value)

class HAIGC_Layer(metaclass=_HAIGC_NoBatchPatchMeta):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE", ),
                "每批合成方式": (["所有单元合成", "批次展开为图层", "单个单元合成"], {"default": "所有单元合成"}),
                "不透明度": ("INT", {"default": 100, "min": 0, "max": 100, "label": "不透明度"}),
                "混合模式": (["正常", "正片叠底", "滤色", "叠加", "变暗", "变亮", "颜色减淡", "颜色加深", "强光", "柔光", "差值", "排除", "色相", "饱和度", "颜色", "明度"], {"default": "正常"}),
                "排版方式": (["叠加", "向右", "向左", "向下", "向上"], {"default": "向下"}),
                "间距": ("INT", {"default": 0, "min": -10000, "max": 10000, "label": "间距"}),
                "间距颜色": ("STRING", {"default": "#FFFFFF", "label": "间距颜色"}),
                "匹配上一层大小": ("BOOLEAN", {"default": True, "label": "匹配上一层大小"}),
            },
            "optional": {
                "遮罩": ("MASK", ),
            }
        }

    RETURN_TYPES = ("PSD_LAYERS",)
    RETURN_NAMES = ("图层数据",)
    FUNCTION = "add_layer"
    CATEGORY = "HAIGC/PSD"

    def add_layer(self, 图像, 每批合成方式="所有单元合成", *args, **kwargs):
        layers = []

        if "批次合成方式" in kwargs and kwargs["批次合成方式"] is not None:
            每批合成方式 = kwargs["批次合成方式"]

        图层名 = kwargs.get("图层名", "")
        if isinstance(图层名, list):
            图层名 = 图层名[0] if 图层名 else ""
        图层名 = str(图层名 or "")

        不透明度 = kwargs.get("不透明度", 100)
        混合模式 = kwargs.get("混合模式", "normal")
        排版方式 = kwargs.get("排版方式", "向下")
        间距 = kwargs.get("间距", 0)
        间距颜色 = kwargs.get("间距颜色", "#FFFFFF")
        匹配上一层大小 = kwargs.get("匹配上一层大小", True)
        遮罩 = kwargs.get("遮罩", None)

        if args:
            args_list = list(args)
            if isinstance(args_list[0], str):
                图层名 = args_list[0]
                args_list = args_list[1:]
            if len(args_list) > 0:
                不透明度 = args_list[0]
            if len(args_list) > 1:
                混合模式 = args_list[1]
            if len(args_list) > 2:
                排版方式 = args_list[2]
            if len(args_list) > 3:
                间距 = args_list[3]
            if len(args_list) > 4:
                间距颜色 = args_list[4]
            if len(args_list) > 5:
                匹配上一层大小 = args_list[5]
            if len(args_list) > 6:
                遮罩 = args_list[6]

        try:
            不透明度 = int(不透明度)
        except Exception:
            不透明度 = 100
        try:
            间距 = int(间距)
        except Exception:
            间距 = 0
        try:
            匹配上一层大小 = bool(匹配上一层大小)
        except Exception:
            匹配上一层大小 = True

        if isinstance(混合模式, str):
            混合模式 = 混合模式.strip() or "normal"
        if isinstance(排版方式, str):
            排版方式 = 排版方式.strip() or "向下"
        if isinstance(间距颜色, str):
            间距颜色 = 间距颜色.strip() or "#FFFFFF"

        _mode_raw = str(每批合成方式 or "").strip()
        if _mode_raw in ("所有单元合成", "所有单元合成一个PSD", "所有批次合成一个PSD", "所有批合成"):
            mode_norm = "all"
        elif _mode_raw in ("单个单元合成", "单个单元合成PSD", "每个批次单独合成PSD", "每批合成"):
            mode_norm = "each"
        else:
            mode_norm = _mode_raw

        use_batch_suffix_in_name = (mode_norm != "all")

        def _new_uid():
            return uuid.uuid4().hex

        selected_images = 图像
        if isinstance(selected_images, list) and len(selected_images) == 1 and isinstance(selected_images[0], torch.Tensor):
            selected_images = selected_images[0]
        selected_names = None
        if hasattr(selected_images, "names"):
            selected_names = selected_images.names
        treat_sources_as_layers = isinstance(selected_images, list) and selected_names is not None

        if isinstance(selected_images, list) and selected_names is None:
            max_list_batch = 1
            for img in selected_images:
                if hasattr(img, "shape") and len(img.shape) == 4:
                    try:
                        max_list_batch = max(max_list_batch, int(img.shape[0]))
                    except Exception:
                        pass

            if max_list_batch <= 1:
                batch_images = []
                for img in selected_images:
                    if not hasattr(img, "shape"):
                        continue
                    if len(img.shape) == 4:
                        batch_images.append(img[0])
                    elif len(img.shape) == 3:
                        batch_images.append(img)

                batch_count = len(batch_images)
                if batch_count == 0:
                    return (layers,)

                mask_in = 遮罩
                if isinstance(mask_in, list) and len(mask_in) == 1 and isinstance(mask_in[0], torch.Tensor):
                    mask_in = mask_in[0]

                if mode_norm == "each":
                    batches = []
                    for b in range(batch_count):
                        img_tensor = batch_images[b]
                        height, width, _ = img_tensor.shape

                        mask_tensor = None
                        if mask_in is not None:
                            if isinstance(mask_in, list):
                                if b < len(mask_in) and isinstance(mask_in[b], torch.Tensor):
                                    m = mask_in[b]
                                    if m.dim() == 3:
                                        mask_tensor = m[0]
                                    else:
                                        mask_tensor = m
                            else:
                                if mask_in.shape[0] == 1:
                                    mask_tensor = mask_in[0]
                                elif b < mask_in.shape[0]:
                                    mask_tensor = mask_in[b]
                                else:
                                    mask_tensor = mask_in[0]

                        l_name = 图层名 or "图层1"
                        if use_batch_suffix_in_name and batch_count > 1:
                            l_name = f"{l_name} {b+1}"

                        batches.append([{
                            "__haigc_layer_uid": _new_uid(),
                            "image": img_tensor,
                            "mask": mask_tensor,
                            "name": l_name,
                            "opacity": int(不透明度 * 255 / 100),
                            "blend_mode": 混合模式,
                            "x": 0,
                            "y": 0,
                            "width": width,
                            "height": height
                        }])

                    return ({HAIGC_LAYER_BATCHES_KEY: batches},)

                if mode_norm == "all" and batch_count > 1:
                    base_x = 0
                    base_y = 0
                    prev_batch_w = 0
                    prev_batch_h = 0

                    for b in range(batch_count):
                        img_tensor = batch_images[b]
                        height, width, _ = img_tensor.shape

                        mask_tensor = None
                        if mask_in is not None:
                            if isinstance(mask_in, list):
                                if b < len(mask_in) and isinstance(mask_in[b], torch.Tensor):
                                    m = mask_in[b]
                                    if m.dim() == 3:
                                        mask_tensor = m[0]
                                    else:
                                        mask_tensor = m
                            else:
                                if mask_in.shape[0] == 1:
                                    mask_tensor = mask_in[0]
                                elif b < mask_in.shape[0]:
                                    mask_tensor = mask_in[b]
                                else:
                                    mask_tensor = mask_in[0]

                        current_x = 0
                        current_y = 0
                        if b == 0 or 排版方式 in ["overlap", "叠加"]:
                            current_x = 0
                            current_y = 0
                        elif 排版方式 in ["right", "向右"]:
                            current_x = base_x + prev_batch_w + 间距
                            current_y = base_y
                        elif 排版方式 in ["left", "向左"]:
                            current_x = base_x - width - 间距
                            current_y = base_y
                        elif 排版方式 in ["down", "向下"]:
                            current_x = base_x
                            current_y = base_y + prev_batch_h + 间距
                        elif 排版方式 in ["up", "向上"]:
                            current_x = base_x
                            current_y = base_y - height - 间距

                        if b > 0 and 间距 > 0 and 间距颜色 and 排版方式 not in ["overlap", "叠加"]:
                            rgb = hex_to_rgb(间距颜色)
                            if rgb:
                                spacer_x, spacer_y, spacer_w, spacer_h = 0, 0, 0, 0
                                if 排版方式 in ["right", "向右"]:
                                    spacer_x = base_x + prev_batch_w
                                    spacer_y = base_y
                                    spacer_w = 间距
                                    spacer_h = max(prev_batch_h, height)
                                elif 排版方式 in ["left", "向左"]:
                                    spacer_x = current_x + width
                                    spacer_y = base_y
                                    spacer_w = 间距
                                    spacer_h = max(prev_batch_h, height)
                                elif 排版方式 in ["down", "向下"]:
                                    spacer_x = base_x
                                    spacer_y = base_y + prev_batch_h
                                    spacer_w = max(prev_batch_w, width)
                                    spacer_h = 间距
                                elif 排版方式 in ["up", "向上"]:
                                    spacer_x = base_x
                                    spacer_y = current_y + height
                                    spacer_w = max(prev_batch_w, width)
                                    spacer_h = 间距

                                if spacer_w > 0 and spacer_h > 0:
                                    r, g, bb = [c / 255.0 for c in rgb]
                                    spacer_tensor = torch.tensor([[[r, g, bb]]]).repeat(spacer_h, spacer_w, 1)
                                    layers.append({
                                        "__haigc_layer_uid": _new_uid(),
                                        "image": spacer_tensor,
                                        "mask": None,
                                        "name": f"Spacing {len(layers)}",
                                        "opacity": 255,
                                        "blend_mode": "normal",
                                        "x": spacer_x,
                                        "y": spacer_y,
                                        "width": spacer_w,
                                        "height": spacer_h
                                    })

                        l_name = 图层名 or "图层1"
                        layers.append({
                            "__haigc_layer_uid": _new_uid(),
                            "image": img_tensor,
                            "mask": mask_tensor,
                            "name": l_name,
                            "opacity": int(不透明度 * 255 / 100),
                            "blend_mode": 混合模式,
                            "x": current_x,
                            "y": current_y,
                            "width": width,
                            "height": height,
                            "batch_index": b,
                            "layer_order": 0
                        })

                        base_x = current_x
                        base_y = current_y
                        prev_batch_w = width
                        prev_batch_h = height

                    return (layers,)

        sources = []
        if isinstance(selected_images, list):
            for idx, img in enumerate(selected_images):
                base_name = ""
                if selected_names is not None and idx < len(selected_names):
                    base_name = selected_names[idx] or ""
                sources.append((img, base_name))
        else:
            sources.append((selected_images, ""))

        batch_count = 1
        for img, _ in sources:
            if hasattr(img, "shape") and len(img.shape) == 4:
                try:
                    batch_count = max(batch_count, int(img.shape[0]))
                except Exception:
                    pass

        group_batches_in_psd = (mode_norm == "all" and batch_count > 1)

        if mode_norm == "each":
            batches = []
            for b in range(batch_count):
                batch_layers = []
                for s_idx, (img, base_name) in enumerate(sources):
                    if not hasattr(img, "shape"):
                        continue
                    if len(img.shape) == 4:
                        if int(img.shape[0]) == 1:
                            img_tensor = img[0]
                        elif b < int(img.shape[0]):
                            img_tensor = img[b]
                        else:
                            img_tensor = img[0]
                    elif len(img.shape) == 3:
                        img_tensor = img
                    else:
                        continue

                    height, width, _ = img_tensor.shape

                    mask_tensor = None
                    if 遮罩 is not None:
                        if 遮罩.shape[0] == 1:
                            mask_tensor = 遮罩[0]
                        elif b < 遮罩.shape[0]:
                            mask_tensor = 遮罩[b]
                        else:
                            mask_tensor = 遮罩[0]

                    if base_name:
                        l_name = base_name
                        if use_batch_suffix_in_name and batch_count > 1:
                            l_name = f"{base_name} {b+1}"
                    elif 图层名:
                        l_name = 图层名
                        if len(sources) > 1:
                            l_name = f"{图层名}{s_idx+1}"
                        if use_batch_suffix_in_name and batch_count > 1:
                            l_name = f"{l_name} {b+1}"
                    else:
                        l_name = f"图层{s_idx+1}"

                    batch_layers.append({
                        "__haigc_layer_uid": _new_uid(),
                        "image": img_tensor,
                        "mask": mask_tensor,
                        "name": l_name,
                        "opacity": int(不透明度 * 255 / 100),
                        "blend_mode": 混合模式,
                        "x": 0,
                        "y": 0,
                        "width": width,
                        "height": height
                    })

                batches.append(batch_layers)

            return ({HAIGC_LAYER_BATCHES_KEY: batches},)

        if group_batches_in_psd:
            if len(sources) > 1:
                for s_idx, (img, base_name) in enumerate(sources):
                    if not hasattr(img, "shape"):
                        continue
                    if len(img.shape) == 4:
                        local_batch = int(img.shape[0])
                    elif len(img.shape) == 3:
                        local_batch = 1
                    else:
                        continue

                    prev_layer = None
                    for b in range(local_batch):
                        if len(img.shape) == 4:
                            if local_batch == 1:
                                img_tensor = img[0]
                            elif b < local_batch:
                                img_tensor = img[b]
                            else:
                                img_tensor = img[0]
                        else:
                            img_tensor = img

                        height, width, _ = img_tensor.shape

                        mask_tensor = None
                        if 遮罩 is not None:
                            if 遮罩.shape[0] == 1:
                                mask_tensor = 遮罩[0]
                            elif b < 遮罩.shape[0]:
                                mask_tensor = 遮罩[b]
                            else:
                                mask_tensor = 遮罩[0]

                        if base_name:
                            l_name = base_name
                        elif 图层名:
                            l_name = 图层名
                            if len(sources) > 1:
                                l_name = f"{图层名}{s_idx+1}"
                        else:
                            l_name = f"图层{s_idx+1}"

                        if 匹配上一层大小 and prev_layer is not None:
                            prev_w = prev_layer.get("width", 0)
                            prev_h = prev_layer.get("height", 0)
                            target_w, target_h = width, height

                            if 排版方式 in ["left", "right", "向左", "向右"]:
                                if prev_h > 0 and height > 0:
                                    scale = prev_h / height
                                    target_h = prev_h
                                    target_w = int(width * scale)
                            elif 排版方式 in ["up", "down", "向上", "向下"]:
                                if prev_w > 0 and width > 0:
                                    scale = prev_w / width
                                    target_w = prev_w
                                    target_h = int(height * scale)
                            elif 排版方式 in ["overlap", "叠加"]:
                                if prev_w > 0 and prev_h > 0:
                                    target_w = prev_w
                                    target_h = prev_h

                            if target_w != width or target_h != height:
                                img_permuted = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                                img_resized = torch.nn.functional.interpolate(img_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
                                img_tensor = img_resized.permute(0, 2, 3, 1).squeeze(0)

                                if mask_tensor is not None:
                                    mask_permuted = mask_tensor.unsqueeze(0).unsqueeze(0)
                                    mask_resized = torch.nn.functional.interpolate(mask_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
                                    mask_tensor = mask_resized.squeeze(0).squeeze(0)

                                width, height = target_w, target_h

                        current_x = 0
                        current_y = 0
                        if prev_layer is not None and 排版方式 not in ["overlap", "叠加"]:
                            prev_x = prev_layer.get("x", 0)
                            prev_y = prev_layer.get("y", 0)
                            prev_w = prev_layer.get("width", 0)
                            prev_h = prev_layer.get("height", 0)
                            if 排版方式 in ["right", "向右"]:
                                current_x = prev_x + prev_w + 间距
                                current_y = prev_y
                            elif 排版方式 in ["left", "向左"]:
                                current_x = prev_x - width - 间距
                                current_y = prev_y
                            elif 排版方式 in ["down", "向下"]:
                                current_x = prev_x
                                current_y = prev_y + prev_h + 间距
                            elif 排版方式 in ["up", "向上"]:
                                current_x = prev_x
                                current_y = prev_y - height - 间距

                        ld = {
                            "__haigc_layer_uid": _new_uid(),
                            "image": img_tensor,
                            "mask": mask_tensor,
                            "name": l_name,
                            "opacity": int(不透明度 * 255 / 100),
                            "blend_mode": 混合模式,
                            "x": current_x,
                            "y": current_y,
                            "width": width,
                            "height": height,
                            "batch_index": s_idx,
                            "layer_order": b
                        }
                        layers.append(ld)
                        prev_layer = ld

                return (layers,)

            base_x = 0
            base_y = 0
            prev_batch_w = 0
            prev_batch_h = 0

            for b in range(batch_count):
                batch_layers = []
                prev_layer = None

                for s_idx, (img, base_name) in enumerate(sources):
                    if not hasattr(img, "shape"):
                        continue
                    if len(img.shape) == 4:
                        if int(img.shape[0]) == 1:
                            img_tensor = img[0]
                        elif b < int(img.shape[0]):
                            img_tensor = img[b]
                        else:
                            img_tensor = img[0]
                    elif len(img.shape) == 3:
                        img_tensor = img
                    else:
                        continue

                    height, width, _ = img_tensor.shape

                    mask_tensor = None
                    if 遮罩 is not None:
                        if 遮罩.shape[0] == 1:
                            mask_tensor = 遮罩[0]
                        elif b < 遮罩.shape[0]:
                            mask_tensor = 遮罩[b]
                        else:
                            mask_tensor = 遮罩[0]

                    if base_name:
                        l_name = base_name
                    elif 图层名:
                        l_name = 图层名
                        if len(sources) > 1:
                            l_name = f"{图层名}{s_idx+1}"
                    else:
                        l_name = f"图层{s_idx+1}"

                    if 匹配上一层大小 and prev_layer is not None:
                        prev_w = prev_layer.get("width", 0)
                        prev_h = prev_layer.get("height", 0)
                        target_w, target_h = width, height
                        if prev_w > 0 and prev_h > 0:
                            target_w = prev_w
                            target_h = prev_h

                        if target_w != width or target_h != height:
                            img_permuted = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                            img_resized = torch.nn.functional.interpolate(img_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            img_tensor = img_resized.permute(0, 2, 3, 1).squeeze(0)

                            if mask_tensor is not None:
                                mask_permuted = mask_tensor.unsqueeze(0).unsqueeze(0)
                                mask_resized = torch.nn.functional.interpolate(mask_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
                                mask_tensor = mask_resized.squeeze(0).squeeze(0)

                            width, height = target_w, target_h

                    ld = {
                        "__haigc_layer_uid": _new_uid(),
                        "image": img_tensor,
                        "mask": mask_tensor,
                        "name": l_name,
                        "opacity": int(不透明度 * 255 / 100),
                        "blend_mode": 混合模式,
                        "x": 0,
                        "y": 0,
                        "width": width,
                        "height": height,
                        "batch_index": b,
                        "layer_order": s_idx
                    }

                    batch_layers.append(ld)
                    prev_layer = ld

                for ld in batch_layers:
                    ld["x"] = 0
                    ld["y"] = 0

                if batch_layers:
                    batch_min_x = min(int(ld.get("x", 0)) for ld in batch_layers)
                    batch_min_y = min(int(ld.get("y", 0)) for ld in batch_layers)
                    if batch_min_x != 0 or batch_min_y != 0:
                        for ld in batch_layers:
                            ld["x"] = int(ld.get("x", 0)) - batch_min_x
                            ld["y"] = int(ld.get("y", 0)) - batch_min_y

                    batch_w = max(int(ld.get("x", 0)) + int(ld.get("width", 0)) for ld in batch_layers)
                    batch_h = max(int(ld.get("y", 0)) + int(ld.get("height", 0)) for ld in batch_layers)
                else:
                    batch_w = 0
                    batch_h = 0

                current_x = 0
                current_y = 0
                if b == 0 or 排版方式 in ["overlap", "叠加"]:
                    current_x = 0
                    current_y = 0
                elif 排版方式 in ["right", "向右"]:
                    current_x = base_x + prev_batch_w + 间距
                    current_y = base_y
                elif 排版方式 in ["left", "向左"]:
                    current_x = base_x - batch_w - 间距
                    current_y = base_y
                elif 排版方式 in ["down", "向下"]:
                    current_x = base_x
                    current_y = base_y + prev_batch_h + 间距
                elif 排版方式 in ["up", "向上"]:
                    current_x = base_x
                    current_y = base_y - batch_h - 间距

                if b > 0 and 间距 > 0 and 间距颜色 and 排版方式 not in ["overlap", "叠加"]:
                    rgb = hex_to_rgb(间距颜色)
                    if rgb:
                        spacer_x, spacer_y, spacer_w, spacer_h = 0, 0, 0, 0
                        if 排版方式 in ["right", "向右"]:
                            spacer_x = base_x + prev_batch_w
                            spacer_y = base_y
                            spacer_w = 间距
                            spacer_h = max(prev_batch_h, batch_h)
                        elif 排版方式 in ["left", "向左"]:
                            spacer_x = current_x + batch_w
                            spacer_y = base_y
                            spacer_w = 间距
                            spacer_h = max(prev_batch_h, batch_h)
                        elif 排版方式 in ["down", "向下"]:
                            spacer_x = base_x
                            spacer_y = base_y + prev_batch_h
                            spacer_w = max(prev_batch_w, batch_w)
                            spacer_h = 间距
                        elif 排版方式 in ["up", "向上"]:
                            spacer_x = base_x
                            spacer_y = current_y + batch_h
                            spacer_w = max(prev_batch_w, batch_w)
                            spacer_h = 间距

                        if spacer_w > 0 and spacer_h > 0:
                            r, g, bb = [c/255.0 for c in rgb]
                            spacer_tensor = torch.tensor([[[r, g, bb]]]).repeat(spacer_h, spacer_w, 1)
                            layers.append({
                                "__haigc_layer_uid": _new_uid(),
                                "image": spacer_tensor,
                                "mask": None,
                                "name": f"Spacing {len(layers)}",
                                "opacity": 255,
                                "blend_mode": "normal",
                                "x": spacer_x,
                                "y": spacer_y,
                                "width": spacer_w,
                                "height": spacer_h
                            })

                for ld in batch_layers:
                    ld["x"] = current_x + int(ld.get("x", 0))
                    ld["y"] = current_y + int(ld.get("y", 0))
                    layers.append(ld)

                base_x = current_x
                base_y = current_y
                prev_batch_w = batch_w
                prev_batch_h = batch_h

            return (layers,)

        input_images = []
        input_names = []

        if isinstance(selected_images, list):
            for idx, img in enumerate(selected_images):
                base_name = ""
                if selected_names is not None and idx < len(selected_names):
                    base_name = selected_names[idx] or ""

                if hasattr(img, 'shape'):
                    if len(img.shape) == 4:
                        for b in range(img.shape[0]):
                            input_images.append(img[b])
                            if base_name:
                                if use_batch_suffix_in_name and img.shape[0] > 1:
                                    input_names.append(f"{base_name} {b+1}")
                                else:
                                    input_names.append(base_name)
                            else:
                                input_names.append("")
                    elif len(img.shape) == 3:
                        input_images.append(img)
                        input_names.append(base_name or "")
        elif hasattr(selected_images, 'shape'):
            for b in range(selected_images.shape[0]):
                input_images.append(selected_images[b])
                input_names.append("")

        batch_size = len(input_images)
        force_overlap = (排版方式 in ["overlap", "叠加"])
        last_layer = layers[-1] if layers else None

        for i in range(batch_size):
            img_tensor = input_images[i]
            height, width, _ = img_tensor.shape

            mask_tensor = None
            if 遮罩 is not None:
                if 遮罩.shape[0] == 1:
                    mask_tensor = 遮罩[0]
                elif i < 遮罩.shape[0]:
                    mask_tensor = 遮罩[i]
                else:
                    mask_tensor = 遮罩[0]

            if 匹配上一层大小 and last_layer:
                prev_w = last_layer.get("width", 0)
                prev_h = last_layer.get("height", 0)

                target_w, target_h = width, height

                if 排版方式 in ["left", "right", "向左", "向右"]:
                    if prev_h > 0:
                        scale = prev_h / height
                        target_h = prev_h
                        target_w = int(width * scale)
                elif 排版方式 in ["up", "down", "向上", "向下"]:
                    if prev_w > 0:
                        scale = prev_w / width
                        target_w = prev_w
                        target_h = int(height * scale)
                elif 排版方式 in ["overlap", "叠加"]:
                    if prev_w > 0 and prev_h > 0:
                        target_w = prev_w
                        target_h = prev_h

                if target_w != width or target_h != height:
                    img_permuted = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                    img_resized = torch.nn.functional.interpolate(img_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    img_tensor = img_resized.permute(0, 2, 3, 1).squeeze(0)

                    if mask_tensor is not None:
                        mask_permuted = mask_tensor.unsqueeze(0).unsqueeze(0)
                        mask_resized = torch.nn.functional.interpolate(mask_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
                        mask_tensor = mask_resized.squeeze(0).squeeze(0)

                    width, height = target_w, target_h

            current_x = 0
            current_y = 0

            if not force_overlap:
                if last_layer:
                    prev_x = last_layer.get("x", 0)
                    prev_y = last_layer.get("y", 0)
                    prev_w = last_layer.get("width", 0)
                    prev_h = last_layer.get("height", 0)

                    if 排版方式 in ["overlap", "叠加"]:
                        current_x = 0
                        current_y = 0
                    elif 排版方式 in ["right", "向右"]:
                        current_x = prev_x + prev_w + 间距
                        current_y = prev_y
                    elif 排版方式 in ["left", "向左"]:
                        current_x = prev_x - width - 间距
                        current_y = prev_y
                    elif 排版方式 in ["down", "向下"]:
                        current_x = prev_x
                        current_y = prev_y + prev_h + 间距
                    elif 排版方式 in ["up", "向上"]:
                        current_x = prev_x
                        current_y = prev_y - height - 间距
                else:
                    current_x = 0
                    current_y = 0

            if (not force_overlap) and last_layer and 间距 > 0 and 间距颜色:
                rgb = hex_to_rgb(间距颜色)
                if rgb:
                    spacer_x, spacer_y, spacer_w, spacer_h = 0, 0, 0, 0

                    if 排版方式 in ["right", "向右"]:
                        spacer_x = prev_x + prev_w
                        spacer_y = prev_y
                        spacer_w = 间距
                        spacer_h = max(prev_h, height)
                    elif 排版方式 in ["left", "向左"]:
                        spacer_x = prev_x - 间距
                        spacer_y = prev_y
                        spacer_w = 间距
                        spacer_h = max(prev_h, height)
                    elif 排版方式 in ["down", "向下"]:
                        spacer_x = prev_x
                        spacer_y = prev_y + prev_h
                        spacer_w = max(prev_w, width)
                        spacer_h = 间距
                    elif 排版方式 in ["up", "向上"]:
                        spacer_x = prev_x
                        spacer_y = prev_y - 间距
                        spacer_w = max(prev_w, width)
                        spacer_h = 间距

                    if spacer_w > 0 and spacer_h > 0:
                        r, g, bb = [c/255.0 for c in rgb]
                        spacer_tensor = torch.tensor([[[r, g, bb]]]).repeat(spacer_h, spacer_w, 1)

                        layers.append({
                            "__haigc_layer_uid": _new_uid(),
                            "image": spacer_tensor,
                            "mask": None,
                            "name": f"Spacing {len(layers)}",
                            "opacity": 255,
                            "blend_mode": "normal",
                            "x": spacer_x,
                            "y": spacer_y,
                            "width": spacer_w,
                            "height": spacer_h
                        })

            stack_name = input_names[i] if i < len(input_names) else ""
            if stack_name:
                l_name = stack_name
            elif not 图层名:
                l_name = f"图层{len(layers) + 1}"
            else:
                l_name = 图层名
                if use_batch_suffix_in_name and batch_size > 1:
                    l_name = f"{图层名} {i+1}"

            new_layer = {
                "__haigc_layer_uid": _new_uid(),
                "image": img_tensor,
                "mask": mask_tensor,
                "name": l_name,
                "opacity": int(不透明度 * 255 / 100),
                "blend_mode": 混合模式,
                "x": current_x,
                "y": current_y,
                "width": width,
                "height": height,
                "layer_order": i
            }

            layers.append(new_layer)
            last_layer = new_layer

        return (layers,)

class HAIGC_SavePSD:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文件名前缀": ("STRING", {"default": "ComfyUI_PSD", "label": "文件名前缀"}),
            },
            "optional": {
                "图像": ("IMAGE", {"label": "图像"}),
                "遮罩": ("MASK", {"label": "遮罩"}),
                "图层数据": ("PSD_LAYERS", {"label": "图层数据"}),
                "背景配置": (BG_CONFIG_TYPE, {"label": "背景配置"}),
            },
            "hidden": {}
        }

    RETURN_TYPES = ()
    FUNCTION = "save_psd"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    CATEGORY = "HAIGC/PSD"

    def save_psd(self, 文件名前缀="ComfyUI_PSD", 图像=None, 遮罩=None, 图层数据=None, 背景配置=None, 保存触发=None, unique_id=None, **kwargs):
        def get_val(val, default):
            if isinstance(val, list):
                if len(val) > 0:
                    return val[0]
                return default
            return val

        文件名前缀 = get_val(文件名前缀, "ComfyUI_PSD")
        图像 = get_val(图像, None)
        遮罩 = get_val(遮罩, None)
        图层数据 = get_val(图层数据, None)
        背景配置 = get_val(背景配置, None)

        if isinstance(图像, list) and len(图像) == 1 and isinstance(图像[0], torch.Tensor):
            图像 = 图像[0]
        if isinstance(遮罩, list) and len(遮罩) == 1 and isinstance(遮罩[0], torch.Tensor):
            遮罩 = 遮罩[0]

        layer_data_in = 图层数据
        if isinstance(layer_data_in, list) and len(layer_data_in) == 1:
            layer_data_in = layer_data_in[0]

        if isinstance(layer_data_in, list) and len(layer_data_in) > 0:
            is_layers_list = all(isinstance(x, dict) and ("image" in x) for x in layer_data_in)
            if not is_layers_list:
                combined_batches = []
                combined_layers = []
                has_batches = False

                for item_idx, item in enumerate(layer_data_in):
                    if isinstance(item, dict) and HAIGC_LAYER_BATCHES_KEY in item:
                        has_batches = True
                        item_batches = item.get(HAIGC_LAYER_BATCHES_KEY) or []
                        if isinstance(item_batches, list):
                            for batch_idx, batch_layers in enumerate(item_batches):
                                if not isinstance(batch_layers, list):
                                    continue
                                while len(combined_batches) <= batch_idx:
                                    combined_batches.append([])
                                for l in batch_layers:
                                    if not isinstance(l, dict):
                                        continue
                                    ld = l.copy()
                                    existing_order = ld.get("layer_order")
                                    if isinstance(existing_order, int):
                                        ld["layer_order"] = int(item_idx) * 10000 + int(existing_order)
                                    else:
                                        ld["layer_order"] = int(item_idx) * 10000
                                    combined_batches[batch_idx].append(ld)
                    elif isinstance(item, list):
                        item_layers = [l for l in item if isinstance(l, dict)]
                        min_x = None
                        min_y = None
                        max_x = None
                        max_y = None
                        for l in item_layers:
                            try:
                                lx = int(l.get("x", 0))
                                ly = int(l.get("y", 0))
                                lw = int(l.get("width", 0))
                                lh = int(l.get("height", 0))
                            except Exception:
                                continue

                            if min_x is None or lx < min_x:
                                min_x = lx
                            if min_y is None or ly < min_y:
                                min_y = ly
                            if max_x is None or (lx + lw) > max_x:
                                max_x = lx + lw
                            if max_y is None or (ly + lh) > max_y:
                                max_y = ly + lh

                        shift_x = 0
                        shift_y = 0
                        if min_x is not None and min_y is not None:
                            shift_x = -min_x
                            shift_y = -min_y

                        has_item_batch = any(isinstance(l.get("batch_index"), (int, np.integer)) for l in item_layers)
                        if has_item_batch:
                            for l in item_layers:
                                ld = l.copy()
                                ld["x"] = int(ld.get("x", 0)) + shift_x
                                ld["y"] = int(ld.get("y", 0)) + shift_y
                                existing_order = ld.get("layer_order")
                                if isinstance(existing_order, int):
                                    ld["layer_order"] = int(item_idx) * 10000 + int(existing_order)
                                else:
                                    ld["layer_order"] = int(item_idx) * 10000
                                combined_layers.append(ld)
                        else:
                            for l in item_layers:
                                ld = l.copy()
                                ld["x"] = int(ld.get("x", 0)) + shift_x
                                ld["y"] = int(ld.get("y", 0)) + shift_y
                                combined_layers.append(ld)

                if has_batches:
                    layer_data_in = {HAIGC_LAYER_BATCHES_KEY: combined_batches}
                else:
                    layer_data_in = combined_layers

        图层数据 = layer_data_in

        if isinstance(图层数据, dict) and HAIGC_LAYER_BATCHES_KEY in 图层数据:
            batches = 图层数据.get(HAIGC_LAYER_BATCHES_KEY) or []
            merged_ui = {"images": [], "psd_filename": [], "subfolder": []}
            for i, batch_layers in enumerate(batches):
                if not isinstance(batch_layers, list):
                    continue
                
                # Extract specific image/mask for this batch if available
                batch_image = None
                if 图像 is not None:
                    if hasattr(图像, 'shape') and len(图像.shape) == 4:
                        if 图像.shape[0] == 1:
                            batch_image = 图像 # Broadcast single image
                        elif i < 图像.shape[0]:
                            batch_image = 图像[i].unsqueeze(0) # Keep 4D shape [1, H, W, C]
                        else:
                             # Fallback or cycle? 
                             # If we have fewer images than batches, assume the last one repeats or the first one?
                             # Standard ComfyUI behavior is usually to repeat the last one, or first.
                             # Let's use the first one (standard broadcast) if we run out.
                             batch_image = 图像[0].unsqueeze(0)
                    elif isinstance(图像, list):
                         if len(图像) == 1:
                             batch_image = 图像 # Broadcast
                         elif i < len(图像):
                             # Keep as list of one tensor, or just the tensor?
                             # recursive save_psd handles both. 
                             # But best to keep consistent structure.
                             batch_image = [图像[i]]
                         else:
                             batch_image = [图像[0]]

                batch_mask = None
                if 遮罩 is not None:
                    if hasattr(遮罩, 'shape') and len(遮罩.shape) == 3: # Mask is [B, H, W]
                        if 遮罩.shape[0] == 1:
                            batch_mask = 遮罩
                        elif i < 遮罩.shape[0]:
                            batch_mask = 遮罩[i].unsqueeze(0)
                        else:
                            batch_mask = 遮罩[0].unsqueeze(0)

                r = self.save_psd(
                    文件名前缀=f"{文件名前缀}_{i+1}",
                    图像=batch_image,
                    遮罩=batch_mask,
                    图层数据=batch_layers,
                    背景配置=背景配置,
                )
                if isinstance(r, dict) and "ui" in r:
                    ui = r.get("ui", {}) or {}
                    merged_ui["images"].extend(ui.get("images", []) or [])
                    merged_ui["psd_filename"].extend(ui.get("psd_filename", []) or [])
                    merged_ui["subfolder"].extend(ui.get("subfolder", []) or [])
            return {"ui": merged_ui}

        layers_to_process = []
        
        if 图层数据 is not None:
            seen_uids = set()
            seen_fallback = set()
            for l in 图层数据:
                if not isinstance(l, dict):
                    continue
                if "image" not in l or not isinstance(l.get("image"), torch.Tensor):
                    continue
                uid = l.get("__haigc_layer_uid")
                if isinstance(uid, str) and uid:
                    if uid in seen_uids:
                        continue
                    seen_uids.add(uid)
                    layers_to_process.append(l.copy())
                    continue

                img = l.get("image")
                m = l.get("mask")
                key = (
                    id(img),
                    id(m),
                    l.get("name"),
                    l.get("x"),
                    l.get("y"),
                    l.get("width"),
                    l.get("height"),
                    l.get("opacity"),
                    l.get("blend_mode"),
                    l.get("batch_index"),
                    l.get("layer_order"),
                )
                if key in seen_fallback:
                    continue
                seen_fallback.add(key)
                layers_to_process.append(l.copy())
            
        images_to_process = []
        names_to_process = []
        
        if 图像 is not None:
             images_to_process = 图像
             if hasattr(图像, "names"):
                 names_to_process = 图像.names
             
        # Normalize to list of tensors and corresponding names
        final_images = []
        final_names = []
        
        if isinstance(images_to_process, list):
            # List of Tensors (from HAIGC_ImageSequence)
            for idx, img_batch in enumerate(images_to_process):
                batch_name = ""
                if idx < len(names_to_process):
                    batch_name = names_to_process[idx]
                
                if hasattr(img_batch, 'shape'):
                    # Handle 4D Tensor [B, H, W, C]
                    if len(img_batch.shape) == 4:
                        for b in range(img_batch.shape[0]):
                            final_images.append(img_batch[b])
                            if batch_name:
                                if img_batch.shape[0] > 1:
                                    final_names.append(f"{batch_name} {b+1}")
                                else:
                                    final_names.append(batch_name)
                            else:
                                final_names.append("")
                    # Handle 3D Tensor [H, W, C] (Just in case)
                    elif len(img_batch.shape) == 3:
                        final_images.append(img_batch)
                        final_names.append(batch_name)
                        
        elif images_to_process is not None and hasattr(images_to_process, 'shape'):
             # Single Tensor input [B, H, W, C]
             for b in range(images_to_process.shape[0]):
                 final_images.append(images_to_process[b])
                 
                 n = ""
                 if b < len(names_to_process):
                     n = names_to_process[b]
                 final_names.append(n)
        
        if final_images:
            # Reverse order so earlier images are on top
            final_images.reverse()
            final_names.reverse()
            
            if 遮罩 is not None and 遮罩.shape[0] > 1:
                # Reverse relevant masks to match images
                limit = min(遮罩.shape[0], len(final_images))
                if limit > 0:
                    # We only reverse the part that maps 1:1. 
                    # If mask count < image count, the fallback logic (index 0) 
                    # makes it hard to map perfectly, but flipping the available masks is the best bet.
                    # Actually, if we just flip the whole mask tensor, it might be safer if user provided extra masks.
                    # But if we slice, we ensure we flip the masks that were actually used.
                    # Let's just flip the whole tensor, assuming the user provided a matching or superset batch.
                    遮罩 = torch.flip(遮罩, [0])

            for i, img_tensor in enumerate(final_images):
                h, w, c = img_tensor.shape
                
                mask_tensor = None
                if 遮罩 is not None:
                    if 遮罩.shape[0] == 1:
                        mask_tensor = 遮罩[0]
                    elif i < 遮罩.shape[0]:
                        mask_tensor = 遮罩[i]
                    else:
                        mask_tensor = 遮罩[0]
                
                l_name = ""
                if i < len(final_names):
                     l_name = final_names[i]
                
                if not l_name:
                     l_name = f"图层{len(layers_to_process) + 1}"
                
                layers_to_process.append({
                    "__haigc_layer_uid": uuid.uuid4().hex,
                    "image": img_tensor,
                    "mask": mask_tensor,
                    "name": l_name,
                    "opacity": 255,
                    "blend_mode": "normal",
                    "x": 0,
                    "y": 0,
                    "width": w,
                    "height": h
                })

        # Resolve Background Config
        bg_config = None
        if isinstance(背景配置, dict):
            bg_config = 背景配置

        # Calculate Canvas Bounds
        if not layers_to_process and (bg_config is None or bg_config.get("image") is None):
            # Fallback for empty or just solid color without layers
            min_x, min_y, max_x, max_y = 0, 0, 512, 512
        else:
            min_x, min_y = 0, 0
            max_x, max_y = 0, 0
            
            # Initialize with background if present and has image
            has_bg_image = bg_config and bg_config.get("image") is not None
            adapt_mode = bg_config.get("adapt_mode", "背景适应图层(居中)") if bg_config else "背景适应图层(居中)"
            
            # Map legacy keys if adapt_mode is "无" but old keys exist
            if bg_config and adapt_mode == "无":
                if bg_config.get("adapt_layers_to_bg"):
                    adapt_mode = "图层适应背景(居中)"
                elif bg_config.get("adapt_to_layer_size"):
                    adapt_mode = "背景适应图层(拉伸)"
                else:
                    # Legacy "无" maps to default behavior if no specific flags
                    # If user explicitly selected "无" in old version, we should probably respect that behavior?
                    # The user asked to REMOVE "无" option and default to "背景适应图层(居中)".
                    # But for existing workflows that saved "无", what should happen?
                    # If we change default, we change behavior.
                    # Let's map explicit "无" to "背景适应图层(居中)" as per request "默认改为背景适应图层居中"
                    # Wait, "无" behavior was "Canvas adapts to everything".
                    # "背景适应图层(居中)" behavior is "Canvas = Background Size".
                    # If we force this, old workflows without background image might break or behave differently.
                    # But user instruction is "去除无选项，默认改为背景适应图层居中".
                    # So let's default to "背景适应图层(居中)".
                    adapt_mode = "背景适应图层(居中)"

            # Map legacy "背景适应图层" to "背景适应图层(拉伸)"
            if adapt_mode == "背景适应图层":
                adapt_mode = "背景适应图层(拉伸)"

            # Case: Adapt Layers to Background (Centered)
            if has_bg_image and adapt_mode == "图层适应背景(居中)":
                 bg_img = bg_config["image"]
                 bg_h, bg_w = bg_img.shape[1], bg_img.shape[2]
                 max_x = bg_w
                 max_y = bg_h
                 min_x, min_y = 0, 0
                 
                 canvas_w, canvas_h = bg_w, bg_h
                 
                 if adapt_mode == "图层适应背景(居中)":
                     # Group Scale to Fit (Preserve Aspect Ratio) & Center
                     if layers_to_process:
                         # 1. Calculate Group Bounds
                         g_min_x = layers_to_process[0].get("x", 0)
                         g_min_y = layers_to_process[0].get("y", 0)
                         g_max_x = g_min_x + layers_to_process[0].get("width", 0)
                         g_max_y = g_min_y + layers_to_process[0].get("height", 0)
                         
                         for layer in layers_to_process:
                             lx = layer.get("x", 0)
                             ly = layer.get("y", 0)
                             lw = layer.get("width", 0)
                             lh = layer.get("height", 0)
                             
                             if lx < g_min_x: g_min_x = lx
                             if ly < g_min_y: g_min_y = ly
                             if lx + lw > g_max_x: g_max_x = lx + lw
                             if ly + lh > g_max_y: g_max_y = ly + lh
                         
                         group_w = g_max_x - g_min_x
                         group_h = g_max_y - g_min_y
                         
                         # 2. Calculate Scale to Fit Background (Uniform)
                         scale = 1.0
                         if group_w > 0 and group_h > 0:
                             scale_x = canvas_w / group_w
                             scale_y = canvas_h / group_h
                             scale = min(scale_x, scale_y)
                             
                         # 3. Calculate Centering Offset
                         new_group_w = int(group_w * scale)
                         new_group_h = int(group_h * scale)
                         
                         offset_x = (canvas_w - new_group_w) // 2
                         offset_y = (canvas_h - new_group_h) // 2
                         
                         # 4. Apply Transformation
                         for layer in layers_to_process:
                             # Original Properties
                             old_x = layer.get("x", 0)
                             old_y = layer.get("y", 0)
                             old_w = layer.get("width", 0)
                             old_h = layer.get("height", 0)
                             
                             # New Dimensions
                             new_w = int(old_w * scale)
                             new_h = int(old_h * scale)
                             
                             # New Position (Relative to Group + Offset)
                             rel_x = old_x - g_min_x
                             rel_y = old_y - g_min_y
                             
                             new_rel_x = int(rel_x * scale)
                             new_rel_y = int(rel_y * scale)
                             
                             layer["x"] = offset_x + new_rel_x
                             layer["y"] = offset_y + new_rel_y
                             layer["width"] = new_w
                             layer["height"] = new_h
                             
                             # Resize Image Content
                             img_tensor = layer["image"]
                             if new_w > 0 and new_h > 0 and (new_w != old_w or new_h != old_h):
                                 # Ensure CHW for interpolate
                                 img_permuted = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                                 img_resized = torch.nn.functional.interpolate(img_permuted, size=(new_h, new_w), mode='bilinear', align_corners=False)
                                 layer["image"] = img_resized.permute(0, 2, 3, 1).squeeze(0)
                                 
                                 if layer["mask"] is not None:
                                      mask_tensor = layer["mask"]
                                      mask_permuted = mask_tensor.unsqueeze(0).unsqueeze(0)
                                      mask_resized = torch.nn.functional.interpolate(mask_permuted, size=(new_h, new_w), mode='bilinear', align_corners=False)
                                      layer["mask"] = mask_resized.squeeze(0).squeeze(0)



            else:
                # Default or "背景适应图层" or No BG Image
                # If "背景适应图层", we ignore BG image for initial bounds.
                # If "无", we include BG image in bounds.
                
                should_use_bg_size = has_bg_image and (adapt_mode == "无" or not layers_to_process)
                
                if should_use_bg_size:
                    bg_img = bg_config["image"]
                    bg_h, bg_w = bg_img.shape[1], bg_img.shape[2]
                    max_x = bg_w
                    max_y = bg_h
                
                # If we have layers, we might need to expand bounds
                if layers_to_process:
                    # Initialize with first layer if we didn't use bg size
                    first = layers_to_process[0]
                    
                    if not should_use_bg_size:
                        min_x = first.get("x", 0)
                        min_y = first.get("y", 0)
                        max_x = min_x + first.get("width", 512)
                        max_y = min_y + first.get("height", 512)

                    for layer in layers_to_process:
                        lx = layer.get("x", 0)
                        ly = layer.get("y", 0)
                        lw = layer.get("width", 0)
                        lh = layer.get("height", 0)
                        
                        if lx < min_x: min_x = lx
                        if ly < min_y: min_y = ly
                        if lx + lw > max_x: max_x = lx + lw
                        if ly + lh > max_y: max_y = ly + lh

        canvas_width = max_x - min_x
        canvas_height = max_y - min_y
        
        # Create output filename
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(文件名前缀, self.output_dir, canvas_width, canvas_height)
            
        file_name = f"{filename}_{counter:05}_.psd"
        file_path = os.path.join(full_output_folder, file_name)

        output_layers = []
        did_group_batches = False

        def build_nested_image(layer_data):
            img_tensor = layer_data["image"]
            h, w, c = img_tensor.shape
            img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            mask_np = None
            if c == 4:
                mask_np = img_np[:, :, 3]

            if layer_data.get("mask") is not None:
                m_tensor = layer_data["mask"]
                mask_np = (m_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            if mask_np is None:
                mask_np = np.full((h, w), 255, dtype=np.uint8)

            channels = {
                0: img_np[:, :, 0],
                1: img_np[:, :, 1],
                2: img_np[:, :, 2],
                -1: mask_np
            }

            l_name = (layer_data.get("name", "") or "").replace("\x00", "")
            l_opacity = layer_data.get("opacity", 255)
            l_blend = get_blend_mode(layer_data.get("blend_mode", "normal"))

            raw_x = layer_data.get("x", 0)
            raw_y = layer_data.get("y", 0)
            final_left = raw_x - min_x
            final_top = raw_y - min_y

            return nested_layers.Image(
                name=l_name,
                visible=True,
                opacity=l_opacity,
                group_id=0,
                blend_mode=l_blend,
                top=final_top,
                left=final_left,
                bottom=final_top + h,
                right=final_left + w,
                channels=channels
            )

        # Process Background
        if bg_config is not None:
            bg_tensor = None
            adapt_bg_mode = bg_config.get("adapt_mode", "背景适应图层(居中)")
            if adapt_bg_mode == "无" and bg_config.get("adapt_to_layer_size"):
                 adapt_bg_mode = "背景适应图层(拉伸)"
            elif adapt_bg_mode == "无":
                 adapt_bg_mode = "背景适应图层(居中)"
            
            # Map legacy
            if adapt_bg_mode == "背景适应图层":
                 adapt_bg_mode = "背景适应图层(拉伸)"

            if bg_config.get("image") is not None:
                bg_tensor = bg_config["image"][0]
                
                if adapt_bg_mode == "背景适应图层(拉伸)":
                    # Resize to canvas size (Stretch)
                    bg_permuted = bg_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                    bg_resized = torch.nn.functional.interpolate(bg_permuted, size=(canvas_height, canvas_width), mode='bilinear', align_corners=False)
                    bg_tensor = bg_resized.permute(0, 2, 3, 1).squeeze(0)
                    
                elif adapt_bg_mode == "背景适应图层(居中)":
                    # Resize to Cover (Preserve Aspect Ratio) then Crop
                    bg_h_raw, bg_w_raw = bg_tensor.shape[0], bg_tensor.shape[1]
                    
                    if bg_w_raw > 0 and bg_h_raw > 0:
                        scale = max(canvas_width / bg_w_raw, canvas_height / bg_h_raw)
                        new_bg_w = int(bg_w_raw * scale)
                        new_bg_h = int(bg_h_raw * scale)
                        
                        bg_permuted = bg_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                        bg_resized = torch.nn.functional.interpolate(bg_permuted, size=(new_bg_h, new_bg_w), mode='bilinear', align_corners=False)
                        # [1, C, H, W]
                        
                        # Center Crop
                        start_x = (new_bg_w - canvas_width) // 2
                        start_y = (new_bg_h - canvas_height) // 2
                        
                        # Clamp start points (should be >= 0 due to scale calculation)
                        start_x = max(0, start_x)
                        start_y = max(0, start_y)
                        
                        # End points
                        end_x = start_x + canvas_width
                        end_y = start_y + canvas_height
                        
                        # Ensure we don't go out of bounds (due to rounding)
                        # If resized image is slightly smaller due to int truncation, we might need padding?
                        # But scale is calculated with float division, so int(w*scale) usually >= target if we ceil?
                        # int() floors, so it might be 1px smaller.
                        # Let's use ceil for scale or add 1 to safe.
                        
                        # Safer approach: slicing
                        cropped = bg_resized[:, :, start_y:end_y, start_x:end_x]
                        
                        # If cropped is smaller than canvas (rare rounding edge case), pad it
                        if cropped.shape[2] != canvas_height or cropped.shape[3] != canvas_width:
                             pad_h = canvas_height - cropped.shape[2]
                             pad_w = canvas_width - cropped.shape[3]
                             if pad_h > 0 or pad_w > 0:
                                 cropped = torch.nn.functional.pad(cropped, (0, pad_w, 0, pad_h), mode='constant', value=0)
                        
                        bg_tensor = cropped.permute(0, 2, 3, 1).squeeze(0)

            else:
                # Generate Solid Color
                rgb = hex_to_rgb(bg_config.get("color", "#FFFFFF"))
                if not rgb: rgb = (255, 255, 255)
                r, g, b = [c/255.0 for c in rgb]
                bg_tensor = torch.tensor([[[r, g, b]]]).repeat(canvas_height, canvas_width, 1)

            bg_h, bg_w, bg_c = bg_tensor.shape[0], bg_tensor.shape[1], bg_tensor.shape[2]
            bg_np = (bg_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            bg_alpha = None
            if bg_c == 4:
                bg_alpha = bg_np[:, :, 3]
            
            if bg_alpha is None:
                bg_alpha = np.full((bg_h, bg_w), 255, dtype=np.uint8)

            bg_channels = {
                0: bg_np[:, :, 0],
                1: bg_np[:, :, 1],
                2: bg_np[:, :, 2],
                -1: bg_alpha
            }
            
            # Background layer position
            # If auto-size (no image), it fills the canvas starting at (0,0) in canvas space.
            # Canvas space starts at (min_x, min_y) in world space.
            # So if we place it at (min_x, min_y), it fills the canvas.
            # BUT pytoshop expects coordinates relative to the PSD canvas top-left? 
            # No, pytoshop expects absolute coordinates if using 'top', 'left'.
            # Wait, `top` and `left` in `nested_layers.Image` are coordinates relative to the canvas origin (0,0)?
            # In my previous code:
            # bg_left = 0 - min_x
            # bg_top = 0 - min_y
            # This logic assumes the background image was at world (0,0).
            
            if bg_config.get("image") is not None and adapt_bg_mode == "无":
                 bg_left = 0 - min_x
                 bg_top = 0 - min_y
            else:
                 # Auto-generated background fills the canvas OR resized background
                 bg_left = 0
                 bg_top = 0
            
            bg_opacity = int(bg_config.get("opacity", 100) * 255 / 100)
            
            bg_layer = nested_layers.Image(
                name="Background", 
                visible=True,
                opacity=bg_opacity,
                group_id=0,
                blend_mode=enums.BlendMode.normal,
                top=bg_top, 
                left=bg_left, 
                bottom=bg_top + bg_h, 
                right=bg_left + bg_w,
                channels=bg_channels
            )
            output_layers.append(bg_layer)


        layers_with_batch = []
        layers_without_batch = []
        batch_indices = set()
        for layer_data in layers_to_process:
            if isinstance(layer_data, dict) and "batch_index" in layer_data and isinstance(layer_data.get("batch_index"), (int, np.integer)):
                layers_with_batch.append(layer_data)
                batch_indices.add(int(layer_data.get("batch_index")))
            else:
                layers_without_batch.append(layer_data)

        if batch_indices:
            batch_to_layers = {}
            for layer_data in layers_with_batch:
                idx = int(layer_data.get("batch_index"))
                batch_to_layers.setdefault(idx, []).append(layer_data)

            batch_keys = sorted(batch_to_layers.keys())

            def _to_int(v):
                if v is None:
                    return None
                if isinstance(v, bool):
                    return int(v)
                if isinstance(v, (int, np.integer)):
                    return int(v)
                if isinstance(v, float):
                    if v.is_integer():
                        return int(v)
                    return None
                if isinstance(v, str):
                    s = v.strip()
                    if s and (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
                        try:
                            return int(s)
                        except Exception:
                            return None
                return None

            def _layer_key(x):
                v = _to_int(x.get("layer_order"))
                if v is not None:
                    return v
                n = x.get("name")
                if isinstance(n, str):
                    m = re.search(r"\d+", n)
                    if m:
                        try:
                            return int(m.group(0))
                        except Exception:
                            return 0
                return 0

            if len(batch_keys) > 1:
                did_group_batches = True
                for batch_idx in batch_keys:
                    child_layers = []
                    for ld in sorted(batch_to_layers[batch_idx], key=_layer_key):
                        child_layers.append(build_nested_image(ld))
                    child_layers.reverse()
                    output_layers.append(
                        nested_layers.Group(
                            name=f"批次 {batch_idx + 1}".replace("\x00", ""),
                            visible=True,
                            opacity=255,
                            group_id=0,
                            blend_mode=enums.BlendMode.pass_through,
                            layers=child_layers,
                            closed=True
                        )
                    )
            else:
                only_batch_idx = batch_keys[0]
                child_layers = []
                for ld in sorted(batch_to_layers[only_batch_idx], key=_layer_key):
                    child_layers.append(build_nested_image(ld))
                child_layers.reverse()
                for l in child_layers:
                    output_layers.append(l)

            for layer_data in layers_without_batch:
                output_layers.append(build_nested_image(layer_data))
        else:
            for layer_data in layers_to_process:
                output_layers.append(build_nested_image(layer_data))
            
        output_layers.reverse()
        try:
            from pytoshop import codecs as psd_codecs
            packbits_available = hasattr(psd_codecs, "packbits")
        except Exception:
            packbits_available = False

        preferred_compression = enums.Compression.rle if packbits_available else enums.Compression.raw

        try:
            output_psd = nested_layers.nested_layers_to_psd(
                output_layers,
                color_mode=enums.ColorMode.rgb,
                depth=8,
                size=(canvas_width, canvas_height),
                compression=preferred_compression,
            )
        except Exception as e:
            msg = repr(e)
            is_packbits_issue = (
                (isinstance(e, NameError) and "packbits" in msg)
                or (isinstance(e, ModuleNotFoundError) and "packbits" in msg)
                or (isinstance(e, ImportError) and "packbits" in msg)
            )
            if not is_packbits_issue:
                raise
            output_psd = nested_layers.nested_layers_to_psd(
                output_layers,
                color_mode=enums.ColorMode.rgb,
                depth=8,
                size=(canvas_width, canvas_height),
                compression=enums.Compression.raw,
            )
        
        # --- Inject Raw Effects ---
        if not did_group_batches:
            try:
                source_infos = []
                for layer_data in layers_to_process:
                    source_infos.append(layer_data)

                source_infos.reverse()

                if bg_config is not None:
                    source_infos.append({})

                if hasattr(output_psd, 'layer_and_mask_info') and output_psd.layer_and_mask_info:
                    layer_records = output_psd.layer_and_mask_info.layer_info.layer_records

                    for i, record in enumerate(layer_records):
                        if i < len(source_infos):
                            info = source_infos[i]
                            if "raw_effects" in info:
                                for effect in info["raw_effects"]:
                                    try:
                                        key = base64.b64decode(effect["key"])
                                        data = base64.b64decode(effect["data"])

                                        block = GenericTaggedBlock(code=key, data=data)
                                        record.tagged_blocks[key] = block
                                    except Exception as e:
                                        print(f"Failed to inject effect block: {e}")

            except Exception as e:
                print(f"Error processing effects injection: {e}")

        # Lock Background Layer if requested
        if bg_config and bg_config.get("locked"):
            if hasattr(output_psd, 'layer_and_mask_info') and output_psd.layer_and_mask_info:
                layer_records = output_psd.layer_and_mask_info.layer_info.layer_records
                for layer_record in layer_records:
                    if layer_record.name == "Background":
                        # Set Transparency Protected flag (Basic lock)
                        if hasattr(layer_record, 'transparency_protected'):
                            layer_record.transparency_protected = True
                        
                        # Add 'lspf' block for Full Lock (Transparency + Composite + Position)
                        # Bits: 0=Transparency, 1=Composite, 2=Position
                        # Value: 1 | 2 | 4 = 7
                        try:
                            lock_flags = (1 << 0) | (1 << 1) | (1 << 2) 
                            # If nesting is needed, maybe bit 4? But 7 is standard "Lock All" pixels/pos/trans.
                            lspf_block = GenericTaggedBlock(code=b'lspf', data=struct.pack('>I', lock_flags))
                            layer_record.blocks.append(lspf_block)
                        except Exception as e:
                            print(f"HAIGC PSD Warning: Could not add lock block. Error: {e}")

        with open(file_path, 'wb') as f:
            output_psd.write(f)
            
        # Generate Preview with Blend Modes
        canvas_np = np.zeros((canvas_height, canvas_width, 4), dtype=np.float32)
        
        if bg_config is not None:
             # Re-use logic for background generation/placement
             # To avoid code duplication, we could have prepared bg_np earlier.
             # But for simplicity, let's regenerate or reuse if possible.
             # We need bg_np (H, W, 4)
             
             # Reuse bg_np calculated above? 
             # Wait, bg_np above was (H, W, C) where C could be 3 or 4.
             # And we need to place it on canvas.
             
             # Let's just recalculate quickly or use the tensor if we had it.
             # We have bg_tensor (H, W, C).
             
             bg_np_preview = bg_tensor.cpu().numpy()
             if bg_np_preview.shape[2] == 3:
                 bg_np_preview = np.concatenate([bg_np_preview, np.ones((bg_np_preview.shape[0], bg_np_preview.shape[1], 1), dtype=np.float32)], axis=2)
             
             # Apply opacity
             bg_opacity_preview = bg_config.get("opacity", 100) / 100.0
             bg_np_preview[:, :, 3] *= bg_opacity_preview
             
             h_bg, w_bg = bg_np_preview.shape[:2]
             
             adapt_bg_mode = bg_config.get("adapt_mode", "背景适应图层(居中)")
             if adapt_bg_mode == "无" and bg_config.get("adapt_to_layer_size"):
                  adapt_bg_mode = "背景适应图层(拉伸)"
             elif adapt_bg_mode == "无":
                  adapt_bg_mode = "背景适应图层(居中)"
             if adapt_bg_mode == "背景适应图层":
                  adapt_bg_mode = "背景适应图层(拉伸)"

             if bg_config.get("image") is not None and adapt_bg_mode == "无":
                  bg_left = 0 - min_x
                  bg_top = 0 - min_y
             else:
                  bg_left = 0
                  bg_top = 0
             
             y1 = max(0, bg_top)
             x1 = max(0, bg_left)
             y2 = min(canvas_height, bg_top + h_bg)
             x2 = min(canvas_width, bg_left + w_bg)
             
             src_y1 = y1 - bg_top
             src_x1 = x1 - bg_left
             src_y2 = src_y1 + (y2 - y1)
             src_x2 = src_x1 + (x2 - x1)
             
             if y2 > y1 and x2 > x1:
                 canvas_np[y1:y2, x1:x2] = bg_np_preview[src_y1:src_y2, src_x1:src_x2]
             
        for layer_data in reversed(layers_to_process):
            img_tensor = layer_data["image"]
            img_np = img_tensor.cpu().numpy()
            
            if img_np.shape[2] == 3:
                img_np = np.concatenate([img_np, np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=np.float32)], axis=2)
            
            if layer_data["mask"] is not None:
                m_tensor = layer_data["mask"]
                mask_np = m_tensor.cpu().numpy()
                if mask_np.ndim == 2:
                    mask_np = mask_np[:, :, np.newaxis]
                img_np[:, :, 3:] *= mask_np
                
            raw_x = layer_data.get("x", 0)
            raw_y = layer_data.get("y", 0)
            final_left = raw_x - min_x
            final_top = raw_y - min_y
            
            opacity = layer_data.get("opacity", 255) / 255.0
            blend_mode = layer_data.get("blend_mode", "normal")
            
            h, w = img_np.shape[:2]
            y1 = max(0, final_top)
            x1 = max(0, final_left)
            y2 = min(canvas_height, final_top + h)
            x2 = min(canvas_width, final_left + w)
            
            src_y1 = y1 - final_top
            src_x1 = x1 - final_left
            src_y2 = src_y1 + (y2 - y1)
            src_x2 = src_x1 + (x2 - x1)
            
            if y2 > y1 and x2 > x1:
                bg_slice = canvas_np[y1:y2, x1:x2]
                fg_slice = img_np[src_y1:src_y2, src_x1:src_x2]
                
                blended = blend_numpy(bg_slice, fg_slice, blend_mode, opacity)
                canvas_np[y1:y2, x1:x2] = blended
            
        preview_img = Image.fromarray((canvas_np * 255.0).clip(0, 255).astype(np.uint8))
        preview_filename = f"{filename}_{counter:05}_.png"
        preview_path = os.path.join(full_output_folder, preview_filename)
        preview_img.save(preview_path)

        return { 
            "ui": { 
                "images": [
                    { "filename": preview_filename, "subfolder": subfolder, "type": self.type },
                    { "filename": file_name, "subfolder": subfolder, "type": self.type }
                ],
                "psd_filename": [file_name],
                "subfolder": [subfolder]
            } 
        }

class NamedImageList(list):
    def __init__(self, items, names=None):
        super().__init__(items)
        self.names = names or []

class ImageBatchList:
    def __init__(self, images, names=None):
        self.images = images
        self.names = names or []

class HAIGC_ImageSequence:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE",),
                "图层名": ("STRING", {"default": ""}),
            },
            "optional": {
                "上一层": ("HAIGC_IMAGE_STACK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "HAIGC_IMAGE_STACK",)
    RETURN_NAMES = ("图像序列", "图层连接",)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "HAIGC/PSD"
    
    def process(self, 图像, 图层名="", 上一层=None):
        def get_val(val, default):
            if isinstance(val, list):
                if len(val) > 0:
                    return val[0]
                return default
            return val

        图层名 = get_val(图层名, "")
        上一层 = get_val(上一层, None)

        if 上一层 is None:
            current_list = []
            current_names = []
        else:
            if isinstance(上一层, ImageBatchList):
                current_list = 上一层.images
                current_names = getattr(上一层, "names", []) or []
            else:
                current_list = []
                current_names = []

        images_in = 图像
        if not isinstance(images_in, list):
            images_in = [images_in]
        while len(images_in) == 1 and isinstance(images_in[0], list):
            images_in = images_in[0]

        images_to_add = []
        for item in images_in:
            if not isinstance(item, torch.Tensor):
                continue
            if item.dim() == 3:
                item = item.unsqueeze(0)
            if item.dim() != 4:
                continue
            if int(item.shape[0]) > 1:
                for i in range(int(item.shape[0])):
                    images_to_add.append(item[i:i+1])
            else:
                images_to_add.append(item)

        new_list = list(current_list) + images_to_add
        new_names = current_names + [(图层名 or "") for _ in range(len(images_to_add))]
        
        named_list = NamedImageList(new_list, new_names)
        
        return (named_list, ImageBatchList(new_list, new_names))

class HAIGC_PSD_Background:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "颜色": ("STRING", {"default": "#FFFFFF", "label": "颜色"}),
                "不透明度": ("INT", {"default": 100, "min": 0, "max": 100, "label": "不透明度"}),
                "锁定": ("BOOLEAN", {"default": True, "label": "锁定"}),
                "适应模式": (["背景适应图层(居中)", "背景适应图层(拉伸)", "图层适应背景(居中)"], {"default": "背景适应图层(居中)", "label": "适应模式"}),
            },
            "optional": {
                "图像": ("IMAGE", {"label": "图像"}),
            }
        }
    
    RETURN_TYPES = (BG_CONFIG_TYPE,)
    RETURN_NAMES = ("背景配置",)
    FUNCTION = "process"
    CATEGORY = "HAIGC/PSD"
    
    def process(self, 颜色="#FFFFFF", 不透明度=100, 锁定=True, 适应模式="背景适应图层(居中)", 图像=None):
        return ({
            "image": 图像,
            "color": 颜色,
            "opacity": 不透明度,
            "locked": 锁定,
            "adapt_mode": 适应模式
        },)


class HAIGC_LoadPSD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "PSD文件路径": ("STRING", {"default": ""}),
                "目录路径": ("STRING", {"default": ""}),
                "图层输出模式": (["按PSD原位置", "完整原图层"], {"default": "完整原图层"}),
                "图层索引": ("INT", {"default": -1, "min": -1, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("合成图像", "图层图像", "图层名称", "图层信息")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "load_psd"
    CATEGORY = "HAIGC/PSD"

    def load_psd(self, PSD文件路径="", 目录路径="", 图层输出模式="完整原图层", 图层索引=-1):
        # Default parameters for removed inputs
        屏来源 = "整个PSD"
        输出方式 = "合成+分层"
        文件索引 = -1
        屏索引 = -1
        包含隐藏图层 = False

        # Default to rasterizing layer styles (user request)
        图层样式 = "栅格化图层样式"
        
        if PSDImage is None:
            raise RuntimeError("缺少依赖：psd-tools。请先安装后重启ComfyUI。")

        file_paths = []
        psd_path = (PSD文件路径 or "").strip().strip('"')
        directory = (目录路径 or "").strip().strip('"')

        if psd_path:
             if not os.path.isfile(psd_path):
                 raise ValueError(f"PSD文件路径无效: {psd_path}")
             file_paths = [psd_path]
        elif directory:
             if not os.path.isdir(directory):
                 raise ValueError(f"目录路径无效: {directory}")
             for name in os.listdir(directory):
                 if name.lower().endswith(".psd"):
                     file_paths.append(os.path.join(directory, name))
             file_paths.sort()
        else:
             raise ValueError("请提供PSD文件路径或目录路径。")

        if not file_paths:
            raise ValueError("未找到PSD文件。")

        if 文件索引 >= 0:
            if 文件索引 >= len(file_paths):
                raise IndexError("文件索引超出范围。")
            file_paths = [file_paths[文件索引]]

        composite_images = []
        layer_images = []
        layer_data_list = []

        for psd_path in file_paths:
            psd = PSDImage.open(psd_path)

            screens = []
            if 屏来源 == "顶层组(屏)":
                for layer in list(psd):
                    if is_group_like(layer):
                        screens.append(layer)
                if not screens:
                    screens = [psd]
            else:
                screens = [psd]

            if 屏索引 >= 0:
                if 屏索引 >= len(screens):
                    raise IndexError("屏索引超出范围。")
                screens = [screens[屏索引]]

            for screen in screens:
                composite_pil = screen.composite()
                if composite_pil is not None:
                    composite_images.append(pil_to_image_tensor(composite_pil))

                if 输出方式 == "合成+分层":
                    layers = collect_leaf_layers(screen, include_hidden=包含隐藏图层)
                    if 图层索引 >= 0:
                        if 图层索引 >= len(layers):
                            raise IndexError("图层索引超出范围。")
                        layers = [layers[图层索引]]

                    # Determine canvas size and offset for the current screen
                    canvas_width = 0
                    canvas_height = 0
                    offset_x = 0
                    offset_y = 0

                    if hasattr(screen, "size"): # PSDImage usually
                        canvas_width, canvas_height = screen.size
                        offset_x, offset_y = 0, 0
                    elif hasattr(screen, "bbox"): # Group/Layer
                         # Use bbox as canvas if it's a group screen
                         # But user might expect PSD size. 
                         # If "Screen Source" is "Top Group", treating Group as Canvas is logical.
                         # But let's check if we can get PSD size from layer.
                         # layer._psd.size? Accessing private member is risky.
                         # Let's fallback to Group bbox.
                         bbox = screen.bbox
                         offset_x, offset_y = bbox.left, bbox.top
                         canvas_width, canvas_height = bbox.width, bbox.height
                    
                    # Fallback if size is 0
                    if canvas_width == 0 or canvas_height == 0:
                        # Try to guess from composite
                        if composite_pil:
                            canvas_width, canvas_height = composite_pil.size
                            offset_x, offset_y = 0, 0
                        else:
                            # Try to guess from layers
                            max_r, max_b = 0, 0
                            for l in layers:
                                bbox = get_layer_bbox(l)
                                if bbox:
                                    max_r = max(max_r, bbox[2])
                                    max_b = max(max_b, bbox[3])
                            if max_r > 0 and max_b > 0:
                                canvas_width = max_r
                                canvas_height = max_b

                    for layer in layers:
                        bbox = get_layer_bbox(layer)
                        if bbox is None:
                            continue
                        left, top, right, bottom = bbox
                        width = max(0, right - left)
                        height = max(0, bottom - top)
                        if width == 0 or height == 0:
                            continue

                        # Try to get image via composite(), fallback to topil() or numpy()
                        layer_pil = None
                        try:
                            # If using "Keep PSD Position" mode, try to render the full canvas for this layer.
                            # This helps capturing effects (like Drop Shadow) that extend beyond the layer's bbox.
                            # BUT if we are going to render effects manually, we prefer the cropped image to avoid huge canvas processing.
                            if 图层输出模式 == "按PSD原位置" and 图层样式 != "栅格化图层样式":
                                viewport = (int(offset_x), int(offset_y), int(offset_x + canvas_width), int(offset_y + canvas_height))
                                try:
                                    layer_pil = layer.composite(viewport=viewport)
                                except Exception:
                                    layer_pil = None
                                
                                # Fallback if viewport composition fails or returns None
                                if layer_pil is None:
                                    layer_pil = layer.composite()
                            else:
                                layer_pil = layer.composite()
                        except Exception:
                            try:
                                layer_pil = layer.composite()
                            except:
                                pass
                        
                        if layer_pil is None:
                            # Fallback for text layers or other special layers
                            if hasattr(layer, "topil"):
                                try:
                                    layer_pil = layer.topil()
                                except Exception:
                                    pass
                            
                            if layer_pil is None and hasattr(layer, "numpy"):
                                try:
                                    arr = layer.numpy()
                                    if arr is not None:
                                        layer_pil = Image.fromarray(arr)
                                except Exception:
                                    pass

                        if layer_pil is None:
                            continue

                        # Extract effects early
                        effects_data = get_layer_effects(layer)
                        
                        if 图层样式 == "栅格化图层样式" and effects_data:
                            # Apply Layer Styles (Shadow, Glow, etc.)
                            new_pil, off_x, off_y = apply_layer_styles(layer_pil, effects_data)
                            if new_pil is not layer_pil:
                                layer_pil = new_pil
                                # Update bbox position (it moved up/left by off_x/off_y)
                                left -= off_x
                                top -= off_y
                                # Update bounds variables
                                right = left + layer_pil.width
                                bottom = top + layer_pil.height

                        # If we have a screen size (PSD size) and the layer image is already full size, crop it first to bbox
                        # (Some psd-tools methods might return full size with transparency, but composite() usually returns cropped)
                        # Actually, composite() returns image at bbox size.
                        
                        # Process based on output mode
                        final_layer_pil = None
                        final_x = left
                        final_y = top
                        
                        if 图层输出模式 == "按PSD原位置":
                            # Check if we already got the full canvas image via viewport
                            is_full_canvas = False
                            if layer_pil.size == (canvas_width, canvas_height):
                                is_full_canvas = True

                            if is_full_canvas:
                                final_layer_pil = layer_pil
                                final_x = 0 # Relative to the output image, content is at (0,0) offset visually
                                final_y = 0
                            else:
                                # Create full canvas image
                                full_img = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
                                
                                # Calculate paste position relative to canvas (screen)
                                paste_x = left - offset_x
                                paste_y = top - offset_y
                                
                                # Handle cases where paste position is negative (layer outside canvas)
                                # Or layer is larger than canvas?
                                # Paste
                                full_img.paste(layer_pil, (paste_x, paste_y))
                                
                                final_layer_pil = full_img
                                final_x = 0 # Relative to the output image, content is at (0,0) offset visually
                                final_y = 0
                        else:
                            # Crop mode (Return original layer content size)
                            final_layer_pil = layer_pil
                            
                            # Calculate expected dimensions from bbox
                            bbox_w = max(0, right - left)
                            bbox_h = max(0, bottom - top)
                            
                            curr_w, curr_h = final_layer_pil.size
                            
                            # If image size matches bbox size, we are good.
                            # If not, and we have valid canvas dimensions, we attempt to crop.
                            # This handles cases where composite() returns full canvas images.
                            # We also try to crop if image is clearly full canvas size even if canvas dimensions are 0 (fallback)
                            need_crop = False
                            if (curr_w != bbox_w or curr_h != bbox_h):
                                if canvas_width > 0 and canvas_height > 0:
                                    need_crop = True
                                elif curr_w > bbox_w and curr_h > bbox_h:
                                    # Assume full canvas if significantly larger
                                    need_crop = True

                            if need_crop:
                                # We assume the image corresponds to the canvas/screen area
                                # Crop to the layer's bounding box relative to the screen
                                c_left = left - offset_x
                                c_top = top - offset_y
                                c_right = c_left + bbox_w
                                c_bottom = c_top + bbox_h
                                
                                # Clamp to image bounds (just in case)
                                c_left = max(0, int(c_left))
                                c_top = max(0, int(c_top))
                                c_right = min(curr_w, int(c_right))
                                c_bottom = min(curr_h, int(c_bottom))
                                
                                if c_right > c_left and c_bottom > c_top:
                                    final_layer_pil = final_layer_pil.crop((c_left, c_top, c_right, c_bottom))
                            
                            final_x = left
                            final_y = top

                        layer_img_tensor = pil_to_image_tensor(final_layer_pil)
                        # layer_mask_tensor = pil_to_mask_tensor(final_layer_pil) # Generate mask from final image

                        layer_images.append(layer_img_tensor)

                        name = getattr(layer, "name", "") or "Layer"
                        opacity = int(getattr(layer, "opacity", 255))
                        blend_mode = psd_blend_mode_to_name(getattr(layer, "blend_mode", None))
                        
                        layer_info = {
                            "filename": os.path.basename(psd_path),
                            "name": name,
                            "opacity": opacity,
                            "blend_mode": blend_mode,
                            "x": final_x,
                            "y": final_y,
                            "width": int(layer_img_tensor.shape[2]),
                            "height": int(layer_img_tensor.shape[1]),
                            "type": "text" if getattr(layer, "kind", "") == "type" else "pixel"
                        }

                        # Add text content if available
                        if layer_info["type"] == "text" and hasattr(layer, "text"):
                            layer_info["text_content"] = str(layer.text)

                        # Add effects if available
                        # effects_data was already extracted earlier
                        if effects_data:
                            layer_info["effects"] = effects_data

                        # Extract raw effects blocks for preservation
                        raw_effects = []
                        try:
                            if hasattr(layer, "_record") and hasattr(layer._record, "tagged_blocks"):
                                blocks = layer._record.tagged_blocks
                                # Keys of interest (bytes)
                                # lfx2: Object Based Effects (modern)
                                # lrFX: Effects Layer Info (legacy)
                                # iOpa: Fill Opacity (often needed for effects to look right)
                                # frFX: Stroke (Frame Effects)
                                target_keys = [b'lfx2', b'lrFX', b'dsdw', b'isdw', b'oglw', b'iglw', b'bevl', b'sofi', b'iOpa', b'frFX']
                                
                                for key in target_keys:
                                    if key in blocks:
                                        block = blocks[key]
                                        # Serialize block
                                        with io.BytesIO() as f:
                                            block.write(f)
                                            data = f.getvalue()
                                            
                                        # data contains Signature(4)+Key(4)+Length(4)+Payload
                                        if len(data) > 12 and data[:4] == b'8BIM':
                                            # Key check
                                            b_key = data[4:8]
                                            if b_key == key:
                                                length = struct.unpack(">I", data[8:12])[0]
                                                payload = data[12:12+length]
                                                
                                                b64_payload = base64.b64encode(payload).decode('utf-8')
                                                raw_effects.append({
                                                    "key": base64.b64encode(key).decode('utf-8'),
                                                    "data": b64_payload
                                                })
                        except Exception as e:
                            print(f"Error extracting raw effects: {e}")

                        if raw_effects:
                            layer_info["raw_effects"] = raw_effects

                        layer_data_list.append(layer_info)

        composite_batch = image_list_to_batch(composite_images)
        # layer_batch = image_list_to_batch(layer_images)

        # Generate grouped layer names string
        grouped_names = {}
        for d in layer_data_list:
            fname = d.get('filename', 'Unknown')
            if fname not in grouped_names:
                grouped_names[fname] = []
            grouped_names[fname].append(d.get('name', ''))
        
        output_lines = []
        for fname, names in grouped_names.items():
            # Join layer names with space
            joined_names = " ".join(names)
            output_lines.append(f"{fname}：{joined_names}")
        
        layer_names_str = "\n".join(output_lines)
        layer_info_json = json.dumps(layer_data_list, indent=2, ensure_ascii=False)
        
        return (composite_batch, layer_images, layer_names_str, layer_info_json)


class HAIGC_LayerIterator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图层图像": ("IMAGE", ),
            },
            "optional": {
                "图层信息": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "iterate"
    CATEGORY = "HAIGC/PSD"

    def iterate(self, 图层图像, 图层信息=None):
        images_in = 图层图像
        
        def get_val(val, default):
            if isinstance(val, list):
                if len(val) > 0:
                    return val[0]
                return default
            return val

        layer_info = get_val(图层信息, None)

        if len(images_in) == 1 and isinstance(images_in[0], torch.Tensor) and images_in[0].dim() == 4 and images_in[0].shape[0] > 1:
            images = images_in[0]
            batch_size = images.shape[0]
            is_list = False
        else:
            images = images_in
            batch_size = len(images)
            is_list = True
        
        # Parse metadata once
        meta_data_list = []
        if 图层信息:
            try:
                parsed = json.loads(图层信息)
                if isinstance(parsed, list):
                    meta_data_list = parsed
            except:
                pass
        
        image_list = []
        
        for i in range(batch_size):
            # Process Image
            if is_list:
                current_img = images[i]
                if current_img.dim() == 3:
                    current_img = current_img.unsqueeze(0)
            else:
                current_img = images[i:i+1] # [1, H, W, C]
            
            # Check for cropping need
            if i < len(meta_data_list):
                meta = meta_data_list[i]
                orig_w = meta.get("width", 0)
                orig_h = meta.get("height", 0)
                
                # Check dimensions
                current_h, current_w = current_img.shape[1], current_img.shape[2]
                
                # If metadata size is valid and smaller than tensor size, crop it
                # Assuming top-left padding (which is what image_list_to_batch does)
                if orig_w > 0 and orig_h > 0 and (orig_w < current_w or orig_h < current_h):
                    # Only crop if specifically requested via metadata implies it's smaller
                    # This handles the "Complete Original Layer" mode where we want the unpadded image
                    current_img = current_img[:, :orig_h, :orig_w, :]
            
            image_list.append(current_img)
        
        return (image_list,)

class HAIGC_MergeLayers:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图层图像": ("IMAGE", ),
                "图层信息": ("STRING", {"forceInput": True}),
                "合并索引": ("STRING", {"default": "1,2", "multiline": False}),
                "合并模式": (["按PSD位置合并", "直接堆叠"], {"default": "按PSD位置合并"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("合并图像", "其他图层")
    OUTPUT_IS_LIST = (False, True)
    INPUT_IS_LIST = True
    FUNCTION = "merge_layers"
    CATEGORY = "HAIGC/PSD"

    def merge_layers(self, 图层图像, 图层信息, 合并索引, 合并模式="按PSD位置合并"):
        images_in = 图层图像
        
        def get_val(val, default):
            if isinstance(val, list):
                if len(val) > 0:
                    return val[0]
                return default
            return val

        layer_info_str = get_val(图层信息, None)
        merge_indices_str = get_val(合并索引, "1,2")
        merge_mode = get_val(合并模式, "按PSD位置合并")

        try:
            # Handle List or Tensor input logic
            if len(images_in) == 1 and isinstance(images_in[0], torch.Tensor) and images_in[0].dim() == 4 and images_in[0].shape[0] > 1:
                images = images_in[0]
                batch_size = images.shape[0]
                is_list = False
            else:
                images = images_in
                batch_size = len(images)
                is_list = True

            # Parse layer metadata
            layer_data_list = []
            try:
                if layer_info_str:
                    layer_data_list = json.loads(layer_info_str)
            except:
                pass
                
            # batch_size is already set above
            
            # Handle comma/newline separated indices or names
            items = [x.strip() for x in re.split(r'[,\n\s]+', merge_indices_str.strip()) if x.strip()]
            
            valid_indices = []
            
            # Helper to find index by name
            def find_index_by_name(name):
                 # Try exact match first
                 for i, data in enumerate(layer_data_list):
                     if str(data.get("name", "")) == name:
                         return i
                 # Try case-insensitive
                 for i, data in enumerate(layer_data_list):
                     if str(data.get("name", "")).lower() == name.lower():
                         return i
                 return -1

            for item in items:
                # Try integer index
                try:
                    idx = int(item)
                    zero_idx = idx - 1
                    if 0 <= zero_idx < batch_size:
                        valid_indices.append(zero_idx)
                except ValueError:
                    # Try name matching
                    idx = find_index_by_name(item)
                    if idx != -1 and idx < batch_size:
                        valid_indices.append(idx)
            
            # Deduplicate while preserving order
            seen = set()
            final_indices = []
            for i in valid_indices:
                if i not in seen:
                    final_indices.append(i)
                    seen.add(i)
            valid_indices = final_indices
            
            # Calculate other indices (all indices not in valid_indices)
            selected_set = set(valid_indices)
            other_indices = [i for i in range(batch_size) if i not in selected_set]
            
            def do_merge(indices, images, mode, layer_data):
                if not indices:
                    return torch.zeros((1, 1, 1, 3))
                    
                # Determine device
                first_idx = indices[0]
                if is_list:
                    device = images[first_idx].device
                else:
                    device = images.device

                if mode == "直接堆叠":
                    if is_list:
                         max_h, max_w = 0, 0
                         for idx in indices:
                             img = images[idx]
                             h = img.shape[0] if img.dim() == 3 else img.shape[1]
                             w = img.shape[1] if img.dim() == 3 else img.shape[2]
                             max_h = max(max_h, h)
                             max_w = max(max_w, w)
                         h, w = max_h, max_w
                    else:
                         h, w = images.shape[1], images.shape[2]

                    canvas = torch.zeros((1, h, w, 4), dtype=torch.float32, device=device)
                    
                    for idx in indices:
                        if is_list:
                             img = images[idx]
                             if img.dim() == 3: img = img.unsqueeze(0)
                        else:
                             img = images[idx:idx+1]

                        curr_h, curr_w = img.shape[1], img.shape[2]
                        
                        # Pad if needed (Top-Left alignment)
                        if curr_h < h or curr_w < w:
                             pad_w = w - curr_w
                             pad_h = h - curr_h
                             img = img.permute(0, 3, 1, 2)
                             img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
                             img = img.permute(0, 2, 3, 1)

                        if img.shape[3] == 3:
                            img = torch.cat([img, torch.ones((1, h, w, 1), device=img.device)], dim=3)
                    
                        alpha = img[:, :, :, 3:4]
                        canvas = img * alpha + canvas * (1 - alpha)
                    
                    return canvas

                else: # 按PSD位置合并
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = float('-inf'), float('-inf')
                    
                    layers_to_merge = []
                    
                    for idx in indices:
                        meta = {}
                        if idx < len(layer_data):
                            meta = layer_data[idx]
                    
                        if is_list:
                            img = images[idx]
                            if img.dim() == 4: img = img.squeeze(0)
                        else:
                            img = images[idx]

                        h, w = img.shape[0], img.shape[1]
                    
                        l_x = meta.get("x", 0)
                        l_y = meta.get("y", 0)
                        l_w = meta.get("width", w)
                        l_h = meta.get("height", h)
                    
                        # Check for padding (Original Layer mode)
                        # If meta width/height is smaller than tensor, crop
                        if l_w > 0 and l_h > 0 and (l_w < w or l_h < h):
                            img = img[:l_h, :l_w, :]
                            h, w = l_h, l_w
                    
                        min_x = min(min_x, l_x)
                        min_y = min(min_y, l_y)
                        max_x = max(max_x, l_x + l_w)
                        max_y = max(max_y, l_y + l_h)
                    
                        layers_to_merge.append({
                            "img": img,
                            "x": l_x,
                            "y": l_y,
                            "w": l_w,
                            "h": l_h
                        })

                    if min_x == float('inf'):
                         return torch.zeros((1, 1, 1, 3))
                         
                    canvas_w = int(max_x - min_x)
                    canvas_h = int(max_y - min_y)
                    
                    if canvas_w <= 0 or canvas_h <= 0:
                         return torch.zeros((1, 1, 1, 3))
                         
                    canvas = torch.zeros((canvas_h, canvas_w, 4), dtype=torch.float32, device=images.device)
                    
                    for layer in layers_to_merge:
                        img = layer["img"]
                        if img.shape[2] == 3:
                            img = torch.cat([img, torch.ones((img.shape[0], img.shape[1], 1), device=img.device)], dim=2)
                    
                        dest_x = int(layer["x"] - min_x)
                        dest_y = int(layer["y"] - min_y)
                    
                        h_s, w_s = img.shape[0], img.shape[1]
                        d_x2 = dest_x + w_s
                        d_y2 = dest_y + h_s
                    
                        # Bounds check
                        if dest_x >= canvas_w or dest_y >= canvas_h or d_x2 <= 0 or d_y2 <= 0:
                            continue
                    
                        # Clip
                        # src region
                        sx1, sy1 = 0, 0
                        sx2, sy2 = w_s, h_s
                    
                        # dest region
                        dx1, dy1 = dest_x, dest_y
                        dx2, dy2 = d_x2, d_y2
                    
                        if dx1 < 0:
                            sx1 = -dx1
                            dx1 = 0
                        if dy1 < 0:
                            sy1 = -dy1
                            dy1 = 0
                        if dx2 > canvas_w:
                            sx2 = w_s - (dx2 - canvas_w)
                            dx2 = canvas_w
                        if dy2 > canvas_h:
                            sy2 = h_s - (dy2 - canvas_h)
                            dy2 = canvas_h
                            
                        if sx2 <= sx1 or sy2 <= sy1:
                            continue
                            
                        bg_slice = canvas[dy1:dy2, dx1:dx2]
                        fg_slice = img[sy1:sy2, sx1:sx2]
                    
                        alpha_fg = fg_slice[:, :, 3:4]
                        blended = fg_slice * alpha_fg + bg_slice * (1 - alpha_fg)
                    
                        canvas[dy1:dy2, dx1:dx2] = blended
                    
                    return canvas.unsqueeze(0)

            # Perform merges
            merged_img = do_merge(valid_indices, 图层图像, 合并模式, layer_data_list)
            other_list = []
            for idx in other_indices:
                img = 图层图像[idx:idx+1]
                target_w = None
                target_h = None
                if idx < len(layer_data_list) and isinstance(layer_data_list[idx], dict):
                    try:
                        target_w = int(layer_data_list[idx].get("width", 0))
                        target_h = int(layer_data_list[idx].get("height", 0))
                    except Exception:
                        target_w = None
                        target_h = None

                if target_w is not None and target_h is not None and target_w > 0 and target_h > 0:
                    h_cur = int(img.shape[1])
                    w_cur = int(img.shape[2])
                    crop_h = min(target_h, h_cur)
                    crop_w = min(target_w, w_cur)
                    img = img[:, :crop_h, :crop_w, :]

                other_list.append(img)

            if not other_list:
                other_list = [torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=图层图像.device)]
            
            return (merged_img, other_list)
            
        except Exception as e:
            print(f"HAIGC_MergeLayers Error: {e}")
            import traceback
            traceback.print_exc()
            blank = torch.zeros((1, 1, 1, 3))
            return (blank, [blank])

class HAIGC_CombineUnits:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图层数据": ("PSD_LAYERS", {"label": "图层数据"}),
                "排版方式": (["向下", "向上", "向左", "向右"], {"default": "向下", "label": "排版方式"}),
                "间距": ("INT", {"default": 0, "min": -10000, "max": 10000, "label": "间距"}),
                "间距颜色": ("STRING", {"default": "#FFFFFF", "label": "间距颜色"}),
            }
        }

    RETURN_TYPES = ("PSD_LAYERS",)
    RETURN_NAMES = ("图层数据",)
    FUNCTION = "combine"
    INPUT_IS_LIST = True
    CATEGORY = "HAIGC/PSD"

    def combine(self, 图层数据=None, 排版方式="向下", 间距=0, 间距颜色="#FFFFFF"):
        def get_val(val, default):
            if isinstance(val, list):
                if len(val) > 0:
                    return val[0]
                return default
            return val

        if isinstance(图层数据, list) and len(图层数据) == 1:
            图层数据 = 图层数据[0]
        排版方式 = get_val(排版方式, "向下")
        间距 = int(get_val(间距, 0) or 0)
        间距颜色 = str(get_val(间距颜色, "#FFFFFF") or "#FFFFFF")

        def is_layer_dict(x):
            return isinstance(x, dict) and ("image" in x)

        def extract_units(data, allow_split_by_batch_index=True):
            if data is None:
                return []
            if isinstance(data, dict) and HAIGC_LAYER_BATCHES_KEY in data:
                batches = data.get(HAIGC_LAYER_BATCHES_KEY) or []
                return [b for b in batches if isinstance(b, list)]
            if isinstance(data, list):
                if data and all(is_layer_dict(x) for x in data):
                    batch_values = []
                    has_batch = False
                    for l in data:
                        v = l.get("batch_index")
                        if isinstance(v, (int, np.integer)):
                            has_batch = True
                            batch_values.append(int(v))
                    if allow_split_by_batch_index and has_batch:
                        grouped = {}
                        for l in data:
                            v = l.get("batch_index")
                            idx = int(v) if isinstance(v, (int, np.integer)) else 0
                            grouped.setdefault(idx, []).append(l)
                        return [grouped[k] for k in sorted(grouped.keys())]
                    return [data]
                units = []
                for item in data:
                    units.extend(extract_units(item, allow_split_by_batch_index=False))
                return units
            return []

        units_in = extract_units(图层数据)
        if not units_in:
            return ([],)

        prepared_units = []
        for unit_idx, unit_layers in enumerate(units_in):
            norm_layers = []
            min_x = None
            min_y = None
            max_x = None
            max_y = None

            for li, l in enumerate(unit_layers):
                if not is_layer_dict(l):
                    continue
                ld = l.copy()
                img = ld.get("image")
                if not isinstance(img, torch.Tensor) or img.dim() != 3:
                    continue

                try:
                    w = int(ld.get("width", 0))
                    h = int(ld.get("height", 0))
                except Exception:
                    w = 0
                    h = 0

                if w <= 0 or h <= 0:
                    h_img, w_img = int(img.shape[0]), int(img.shape[1])
                    if w <= 0:
                        w = w_img
                    if h <= 0:
                        h = h_img
                    ld["width"] = w
                    ld["height"] = h

                try:
                    x = int(ld.get("x", 0))
                    y = int(ld.get("y", 0))
                except Exception:
                    x = 0
                    y = 0

                ld["x"] = x
                ld["y"] = y
                ld["batch_index"] = unit_idx

                if "layer_order" not in ld:
                    ld["layer_order"] = li

                if min_x is None or x < min_x:
                    min_x = x
                if min_y is None or y < min_y:
                    min_y = y
                if max_x is None or (x + w) > max_x:
                    max_x = x + w
                if max_y is None or (y + h) > max_y:
                    max_y = y + h

                norm_layers.append(ld)

            if not norm_layers:
                continue

            unit_w = int((max_x or 0) - (min_x or 0))
            unit_h = int((max_y or 0) - (min_y or 0))

            for ld in norm_layers:
                ld["x"] = 0
                ld["y"] = 0

            unit_w = max(int(ld.get("width", 0) or 0) for ld in norm_layers)
            unit_h = max(int(ld.get("height", 0) or 0) for ld in norm_layers)

            prepared_units.append((norm_layers, unit_w, unit_h))

        if not prepared_units:
            return ([],)

        merged_layers = []
        origin_x = 0
        origin_y = 0
        prev_origin_x = 0
        prev_origin_y = 0
        prev_w = 0
        prev_h = 0
        rgb = hex_to_rgb(间距颜色) if (间距 > 0 and 间距颜色) else None
        spacer_device = None

        for unit_i, (unit_layers, unit_w, unit_h) in enumerate(prepared_units):
            if unit_i == 0:
                origin_x = 0
                origin_y = 0
            else:
                new_origin_x = origin_x
                new_origin_y = origin_y
                if 排版方式 in ["向右", "right"]:
                    new_origin_x = origin_x + prev_w + 间距
                elif 排版方式 in ["向左", "left"]:
                    new_origin_x = origin_x - unit_w - 间距
                elif 排版方式 in ["向下", "down"]:
                    new_origin_y = origin_y + prev_h + 间距
                elif 排版方式 in ["向上", "up"]:
                    new_origin_y = origin_y - unit_h - 间距

                if rgb is not None:
                    if spacer_device is None:
                        for l in unit_layers:
                            img = l.get("image") if isinstance(l, dict) else None
                            if isinstance(img, torch.Tensor):
                                spacer_device = img.device
                                break
                        if spacer_device is None:
                            spacer_device = torch.device("cpu")

                    spacer_x = 0
                    spacer_y = 0
                    spacer_w = 0
                    spacer_h = 0
                    if 排版方式 in ["向右", "right"]:
                        spacer_x = prev_origin_x + prev_w
                        spacer_y = prev_origin_y
                        spacer_w = 间距
                        spacer_h = max(prev_h, unit_h)
                    elif 排版方式 in ["向左", "left"]:
                        spacer_x = new_origin_x + unit_w
                        spacer_y = prev_origin_y
                        spacer_w = 间距
                        spacer_h = max(prev_h, unit_h)
                    elif 排版方式 in ["向下", "down"]:
                        spacer_x = prev_origin_x
                        spacer_y = prev_origin_y + prev_h
                        spacer_w = max(prev_w, unit_w)
                        spacer_h = 间距
                    elif 排版方式 in ["向上", "up"]:
                        spacer_x = prev_origin_x
                        spacer_y = new_origin_y + unit_h
                        spacer_w = max(prev_w, unit_w)
                        spacer_h = 间距

                    if spacer_w > 0 and spacer_h > 0:
                        r, g, bb = [c / 255.0 for c in rgb]
                        spacer_tensor = torch.tensor([[[r, g, bb]]], dtype=torch.float32, device=spacer_device).repeat(spacer_h, spacer_w, 1)
                        merged_layers.append({
                            "__haigc_layer_uid": uuid.uuid4().hex,
                            "image": spacer_tensor,
                            "mask": None,
                            "name": f"Spacing {unit_i}",
                            "opacity": 255,
                            "blend_mode": "normal",
                            "x": int(spacer_x),
                            "y": int(spacer_y),
                            "width": int(spacer_w),
                            "height": int(spacer_h),
                            "batch_index": int(unit_i - 1),
                            "layer_order": -1
                        })

                origin_x = new_origin_x
                origin_y = new_origin_y

            for ld in unit_layers:
                out = ld.copy()
                out["x"] = int(out.get("x", 0)) + origin_x
                out["y"] = int(out.get("y", 0)) + origin_y
                merged_layers.append(out)

            prev_origin_x = origin_x
            prev_origin_y = origin_y
            prev_w = unit_w
            prev_h = unit_h

        g_min_x = None
        g_min_y = None
        for ld in merged_layers:
            try:
                x = int(ld.get("x", 0))
                y = int(ld.get("y", 0))
            except Exception:
                continue
            if g_min_x is None or x < g_min_x:
                g_min_x = x
            if g_min_y is None or y < g_min_y:
                g_min_y = y

        if (g_min_x is not None and g_min_x < 0) or (g_min_y is not None and g_min_y < 0):
            sx = -int(g_min_x or 0) if (g_min_x is not None and g_min_x < 0) else 0
            sy = -int(g_min_y or 0) if (g_min_y is not None and g_min_y < 0) else 0
            if sx or sy:
                for ld in merged_layers:
                    ld["x"] = int(ld.get("x", 0)) + sx
                    ld["y"] = int(ld.get("y", 0)) + sy

        return (merged_layers,)

NODE_CLASS_MAPPINGS = {
    "HAIGC_SavePSD": HAIGC_SavePSD,
    "HAIGC_Layer": HAIGC_Layer,
    "HAIGC_ImageSequence": HAIGC_ImageSequence,
    "HAIGC_PSD_Background": HAIGC_PSD_Background,
    "HAIGC_LoadPSD": HAIGC_LoadPSD,
    "HAIGC_LayerIterator": HAIGC_LayerIterator,
    "HAIGC_CombineUnits": HAIGC_CombineUnits
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HAIGC_SavePSD": "PSD保存 (HAIGC)",
    "HAIGC_Layer": "PSD图层 (HAIGC)",
    "HAIGC_ImageSequence": "多图串联输入 (HAIGC)",
    "HAIGC_PSD_Background": "PSD背景 (HAIGC)",
    "HAIGC_LoadPSD": "PSD加载 (HAIGC)",
    "HAIGC_LayerIterator": "图层迭代 (HAIGC)",
    "HAIGC_CombineUnits": "合并PSD单元 (HAIGC)"
}
