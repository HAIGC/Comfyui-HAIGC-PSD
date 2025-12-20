import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import pytoshop
from pytoshop import enums
from pytoshop.user import nested_layers

class HAIGC_SavePSD:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI_PSD"}),
            },
            "optional": {
                "masks": ("MASK", ),
                "background_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_psd"
    OUTPUT_NODE = True
    CATEGORY = "HAIGC/PSD"

    def save_psd(self, images, filename_prefix="ComfyUI_PSD", masks=None, background_image=None):
        results = list()
        
        # Determine Canvas Size
        if background_image is not None:
            canvas_height, canvas_width = background_image.shape[1], background_image.shape[2]
        else:
            canvas_height, canvas_width = images.shape[1], images.shape[2]
        
        # Create output filename
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, canvas_width, canvas_height)
            
        file_name = f"{filename}_{counter:05}_.psd"
        file_path = os.path.join(full_output_folder, file_name)

        # Prepare layers list
        psd_layers = []

        # Process Background Image if present
        if background_image is not None:
            bg_tensor = background_image[0]
            bg_h, bg_w, bg_c = background_image.shape[1], background_image.shape[2], background_image.shape[3]
            
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
            
            bg_layer = nested_layers.Image(
                name="Background",
                visible=True,
                opacity=255,
                group_id=0,
                blend_mode=enums.BlendMode.normal,
                top=0,
                left=0,
                bottom=bg_h,
                right=bg_w,
                channels=bg_channels
            )
            psd_layers.append(bg_layer)

        # Process each image in the batch as a layer
        batch_size, height, width, channels_count = images.shape
        
        for i in range(batch_size):
            img_tensor = images[i]
            
            # Process Image (Tensor to uint8 numpy)
            img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            # Process Mask
            mask_np = None
            
            # Check if image is RGBA
            if channels_count == 4:
                mask_np = img_np[:, :, 3]

            # If no alpha from image, try explicit masks input
            if mask_np is None and masks is not None:
                if masks.shape[0] == 1:
                    mask_tensor = masks[0]
                elif i < masks.shape[0]:
                    mask_tensor = masks[i]
                else:
                    mask_tensor = None
                
                if mask_tensor is not None:
                    # Mask is [H, W] float range 0..1
                    mask_np = (mask_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            if mask_np is None:
                mask_np = np.full((height, width), 255, dtype=np.uint8)

            # Create 4 channels: R, G, B, A
            # img_np is [H, W, 3]
            channels = {
                0: img_np[:, :, 0], # R
                1: img_np[:, :, 1], # G
                2: img_np[:, :, 2], # B
                -1: mask_np         # A
            }
            
            # Determine layer name
            if background_image is not None:
                layer_name = f"Layer {i+1}"
            else:
                layer_name = f"Layer {i}"
                if i == 0:
                    layer_name = "Background"

            # Create pytoshop Layer
            # We use nested_layers.ImageLayer for convenience
            layer = nested_layers.Image(
                name=layer_name,
                visible=True,
                opacity=255,
                group_id=0,
                blend_mode=enums.BlendMode.normal,
                top=0,
                left=0,
                bottom=height,
                right=width,
                channels=channels,
                metadata=None
            )
            psd_layers.append(layer)

        # Generate PSD file
        # pytoshop expects layers in Top -> Bottom order (List[0] is Top)
        # We constructed [Background, Layer 0, ... Layer N]
        # We want Background at Bottom, and Layer N at Top.
        # So we want list: [Layer N, ..., Layer 0, Background]
        psd_layers.reverse()
        
        output_psd = nested_layers.nested_layers_to_psd(psd_layers, color_mode=enums.ColorMode.rgb, depth=8, size=(canvas_width, canvas_height))

        # Save PSD
        with open(file_path, 'wb') as f:
            output_psd.write(f)

        # Generate Preview Image (Flattened PNG)
        # Create base canvas
        preview_img = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        
        # Paste Background
        if background_image is not None:
            bg_pil = Image.fromarray(bg_np)
            if bg_pil.mode != 'RGBA':
                bg_pil = bg_pil.convert('RGBA')
            # Assuming bg fills canvas as per logic
            preview_img.paste(bg_pil, (0, 0), bg_pil)
        
        # Paste Layers (in order 0 to N)
        # We need to reconstruct the loop or use the data we already processed.
        # Since we have the original tensors, let's iterate again simply for the preview composition.
        # The psd_layers list was reversed for saving, but for visual stacking we want Bottom->Top.
        # We constructed psd_layers as [Background, Layer 0, ... Layer N] before reversing.
        # So we just follow the batch order.
        
        for i in range(batch_size):
            img_tensor = images[i]
            img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            # Re-derive mask for preview
            p_mask_np = None
            if channels_count == 4:
                p_mask_np = img_np[:, :, 3]
            
            if p_mask_np is None and masks is not None:
                if masks.shape[0] == 1:
                    mask_tensor = masks[0]
                elif i < masks.shape[0]:
                    mask_tensor = masks[i]
                else:
                    mask_tensor = None
                
                if mask_tensor is not None:
                    p_mask_np = (mask_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            if p_mask_np is None:
                p_mask_np = np.full((height, width), 255, dtype=np.uint8)
                
            # Create PIL Image
            # img_np is [H, W, C]
            # If C=4, it has alpha. If C=3, we apply p_mask_np as alpha.
            pil_layer = Image.fromarray(img_np[:, :, :3]) # Get RGB
            pil_layer = pil_layer.convert("RGBA")
            pil_layer.putalpha(Image.fromarray(p_mask_np))
            
            # Paste (assuming 0,0 alignment for now as per pytoshop logic)
            preview_img.paste(pil_layer, (0, 0), pil_layer)
            
        # Save Preview
        preview_filename = f"{filename}_{counter:05}_.png"
        preview_path = os.path.join(full_output_folder, preview_filename)
        preview_img.save(preview_path)

        return { 
            "ui": { 
                "images": [{ "filename": preview_filename, "subfolder": subfolder, "type": self.type }],
                "psd_filename": [file_name],
                "subfolder": [subfolder]
            } 
        }

NODE_CLASS_MAPPINGS = {
    "HAIGC_SavePSD": HAIGC_SavePSD
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HAIGC_SavePSD": "Save PSD (HAIGC)"
}
