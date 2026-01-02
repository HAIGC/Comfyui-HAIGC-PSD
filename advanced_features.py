"""
提示词批量队列节点
输入多段提示词，队列输出，让工作流自动运行多次
"""

import random
import hashlib


class PromptBatchQueue:
    """提示词批量队列 - 输入多段提示词，队列输出，让工作流运行多次"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词文本": ("STRING", {"default": "", "multiline": True}),
                "分割方式": (["按行分割", "按段落分割", "按分隔符分割"], {"default": "按行分割"}),
            },
            "optional": {
                "自定义分隔符": ("STRING", {"default": "---"}),
                "跳过空白": ("BOOLEAN", {"default": True}),
                "去除首尾空格": ("BOOLEAN", {"default": True}),
                "随机顺序": ("BOOLEAN", {"default": False}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "添加前缀": ("STRING", {"default": ""}),
                "添加后缀": ("STRING", {"default": ""}),
                "最少字符数": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "display": "number"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("提示词", "总数量", "队列信息")
    FUNCTION = "create_queue"
    CATEGORY = "HAIGC/高级功能"
    OUTPUT_IS_LIST = (True, False, False)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 基于输入内容和种子生成哈希
        content = str(kwargs.get("提示词文本", ""))
        seed = kwargs.get("随机种子", 0)
        min_len = kwargs.get("最少字符数", 0)
        combined = f"{content}_{seed}_{min_len}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def create_queue(self, 提示词文本, 分割方式, 自定义分隔符="---", 跳过空白=True, 
                    去除首尾空格=True, 随机顺序=False, 随机种子=0, 添加前缀="", 添加后缀="", 最少字符数=0):
        """
        将输入的多段提示词分割成列表，工作流会根据提示词数量自动运行多次
        例如：输入3段提示词，工作流就会运行3次
        """
        
        # 分割提示词
        if 分割方式 == "按行分割":
            prompts = 提示词文本.splitlines()
        elif 分割方式 == "按段落分割":
            prompts = 提示词文本.split('\n\n')
        else:  # 按分隔符分割
            prompts = 提示词文本.split(自定义分隔符)
        
        # 处理空白
        if 跳过空白:
            prompts = [p for p in prompts if p.strip()]
        
        # 去除首尾空格
        if 去除首尾空格:
            prompts = [p.strip() for p in prompts]
            
        # 过滤过短的行
        if 最少字符数 > 0:
            prompts = [p for p in prompts if len(p) >= 最少字符数]
        
        # 检查是否为空
        if not prompts:
            return (["无有效提示词"], 0, "提示词列表为空")
        
        # 添加前后缀
        if 添加前缀 or 添加后缀:
            prompts = [f"{添加前缀}{p}{添加后缀}" for p in prompts]
        
        # 如果开启随机顺序，使用种子打乱
        if 随机顺序:
            random.seed(随机种子)
            random.shuffle(prompts)
        
        count = len(prompts)
        
        # 生成队列信息
        info = f"=== 提示词批量队列 ===\n"
        info += f"分割方式: {分割方式}\n"
        info += f"提示词数量: {count}\n"
        info += f"随机顺序: {'是' if 随机顺序 else '否'}\n"
        if 最少字符数 > 0:
            info += f"最少字符数: {最少字符数}\n"
        if 随机顺序:
            info += f"随机种子: {随机种子}\n"
        if 添加前缀:
            info += f"前缀: {添加前缀}\n"
        if 添加后缀:
            info += f"后缀: {添加后缀}\n"
        info += f"✨ 工作流将自动运行 {count} 次\n"
        info += f"每次运行使用一个提示词"
        
        return (prompts, count, info)


# ComfyUI 节点映射 - 必须导出这些映射才能让ComfyUI识别节点
NODE_CLASS_MAPPINGS = {
    "PromptBatchQueue": PromptBatchQueue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptBatchQueue": "提示词批量队列",
}
