import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Tuple, Union
import numpy as np

class H2RFPromptFusion(nn.Module):
    """
    H2RF-Net 提示融合模块，集成视觉令牌、句子和层级推理知识。
    
    设计对齐论文：
    - 在嵌入级别融合（NOT input_ids）
    - 序列结构：[CLS] + O (visual tokens) + [SEP] + prompt_text + [SEP]
    - 支持句子级和方面级模板
    """
    
    SENTENCE_TEMPLATES = {
        "Ps1": 'The sentence "{S}" combined with the "{R2}" has [MASK] emotion',
        "Ps2": 'The sentence "{S}" combined with the "{R2}" presents a [MASK] sentiment',
        "Ps3": 'The details in the sentence "{S}" and the analysis "{R2}" reveal a [MASK] sentiment',
    }
    
    ASPECT_TEMPLATES = {
        "Pa1": 'For the aspect "{A}", "{S}" combined with the "{R2}" has [MASK] emotion',
        "Pa2": 'For the aspect "{A}", "{S}" combined with the "{R2}" presents a [MASK] sentiment',
        "Pa3": 'For the aspect "{A}", the details in the sentence "{S}" and the analysis "{R2}" reveal a [MASK] sentiment',
    }
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=False)
        self.mlm_head = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.hidden_size = self.bert.config.hidden_size
    
    def _normalize_visual_tokens(
        self,
        O: Union[torch.Tensor, np.ndarray],
        device: torch.device
    ) -> torch.Tensor:
        """
        规范化视觉令牌输入。
        
        支持多种输入格式：
        - torch.Tensor: 直接使用
        - numpy.ndarray: 转换为 torch.Tensor
        
        Args:
            O: 视觉令牌，形状应为 (B, M, H) 或 (M, H)
               - B: batch_size
               - M: num_visual tokens
               - H: hidden_size (应与 BERT hidden_size 一致)
            device: 目标设备
        
        Returns:
            torch.Tensor: 规范化后的视觉令牌 (B, M, H)
        
        Raises:
            ValueError: 输入维度不正确或 hidden_size 不匹配
        """
        # 转换 numpy 数组
        if isinstance(O, np.ndarray):
            O = torch.from_numpy(O).float()
        
        # 确保是 torch.Tensor
        if not isinstance(O, torch.Tensor):
            raise ValueError(f"O 必须是 torch.Tensor 或 numpy.ndarray，得到 {type(O)}")
        
        # 移到指定设备
        O = O.to(device)
        
        # 检查维度
        if O.dim() == 2:
            # 单个样本 (M, H) -> 扩展为 (1, M, H)
            O = O.unsqueeze(0)
        elif O.dim() != 3:
            raise ValueError(f"O 必须是 2D 或 3D 张量，得到形状 {O.shape}")
        
        batch_size, num_visual, hidden_size = O.shape
        
        # 检查 hidden_size 是否匹配
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"视觉令牌的 hidden_size ({hidden_size}) "
                f"不匹配 BERT hidden_size ({self.hidden_size})"
            )
        
        return O
        
    def _format_prompt(
        self,
        s: str,
        r2: str,
        mode: str,
        template_id: str,
        a: Optional[str] = None,
    ) -> str:
        """根据模板格式化提示文本"""
        templates = self.SENTENCE_TEMPLATES if mode == "sentence" else self.ASPECT_TEMPLATES
        template = templates[template_id]
        
        if mode == "sentence":
            return template.format(S=s, R2=r2)
        else:  # aspect
            return template.format(A=a, S=s, R2=r2)
    
    def _find_mask_position(self, tokens: List[str]) -> int:
        """找到 [MASK] 令牌在令牌列表中的位置"""
        try:
            return tokens.index("[MASK]")
        except ValueError:
            raise ValueError("[MASK] 令牌未在提示文本中找到")
    
    def forward(
        self,
        O: Union[torch.Tensor, np.ndarray],  # (B, M, H) 或 (M, H) 视觉令牌
        S_list: List[str],  # 句子列表
        R2_list: List[str],  # 推理知识列表
        mode: str,  # "sentence" 或 "aspect"
        template_id: str,  # Ps1/Ps2/Ps3 或 Pa1/Pa2/Pa3
        A_list: Optional[List[str]] = None,  # 方面列表（如果 mode == "aspect" 则需要）
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        融合视觉令牌和文本提示。
        
        【输入接口说明】
        
        Args:
            O: 视觉令牌嵌入，支持多种格式
               格式 1: (B, M, H)  - 批处理输入
                   B: batch_size (与 S_list, R2_list 长度一致)
                   M: num_visual_tokens (视觉令牌数)
                   H: hidden_size (必须 == self.hidden_size = 768 for BERT)
               格式 2: (M, H)  - 单个样本，自动扩展为 (1, M, H)
               类型: torch.Tensor 或 numpy.ndarray
               
            S_list: 句子列表，长度 B
                   例: ["This product is amazing.", "The service was poor."]
            
            R2_list: 推理知识列表（来自 HRF 的 R2），长度 B
                   例: ["positive indicators", "mixed sentiment"]
            
            mode: 融合模式
                  "sentence" - 句子级融合
                  "aspect" - 方面级融合（此时需提供 A_list）
            
            template_id: 使用的模板
                  句子模式: "Ps1", "Ps2", "Ps3"
                  方面模式: "Pa1", "Pa2", "Pa3"
            
            A_list: 方面列表，仅当 mode == "aspect" 时需要
                   例: ["quality", "service", "price"]
        
        Returns:
            mask_logits: (B, vocab_size) 掩码位置的逻辑
            mask_pos: 长度 B 的列表，每个样本中 [MASK] 的位置
        
        【使用示例】
        ```python
        fusion = H2RFPromptFusion(model_name="bert-base-uncased").to(device)
        
        # 情况 1: 从 CLIP 或其他编码器获得 O（torch.Tensor）
        O_visual = torch.randn(2, 10, 768, device=device)  # (B=2, M=10, H=768)
        
        S_list = ["Amazing product", "Poor service"]
        R2_list = ["positive indicators", "mixed sentiment"]
        
        mask_logits, mask_pos = fusion(
            O=O_visual,
            S_list=S_list,
            R2_list=R2_list,
            mode="sentence",
            template_id="Ps1"
        )
        
        # 情况 2: 从 numpy 数组或文件加载 O
        O_visual_np = np.load("visual_tokens.npy")  # shape: (2, 10, 768)
        mask_logits, mask_pos = fusion(
            O=O_visual_np,
            S_list=S_list,
            R2_list=R2_list,
            mode="sentence",
            template_id="Ps1"
        )
        ```
        """
        # 规范化和验证视觉令牌输入
        device = next(self.parameters()).device
        O = self._normalize_visual_tokens(O, device)
        
        batch_size, num_visual, hidden_size = O.shape
        
        if mode == "aspect" and A_list is None:
            raise ValueError("mode == 'aspect' 时 A_list 不能为 None")
        
        if A_list is None:
            A_list = [None] * batch_size
        
        # 构建批次的融合序列
        fused_inputs_embeds_list = []
        mask_pos_list = []
        
        for b in range(batch_size):
            # 格式化提示文本
            prompt_text = self._format_prompt(
                S_list[b], R2_list[b], mode, template_id, A_list[b]
            )
            
            # 标记化提示文本（不添加特殊令牌）
            prompt_tokens_dict = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            prompt_input_ids = prompt_tokens_dict["input_ids"][0]  # (L_p,)
            prompt_tokens = self.tokenizer.convert_ids_to_tokens(prompt_input_ids)
            
            # 找到 [MASK] 位置（在原始提示令牌中）
            mask_idx_in_prompt = self._find_mask_position(prompt_tokens)
            
            # 获取嵌入
            with torch.no_grad():
                # CLS 和 SEP 的嵌入
                cls_id = self.tokenizer.cls_token_id
                sep_id = self.tokenizer.sep_token_id
                
                special_embeds = self.bert.embeddings.word_embeddings(
                    torch.tensor([cls_id, sep_id], device=device)
                )
                cls_embed = special_embeds[0:1]  # (1, H)
                sep_embed = special_embeds[1:2]  # (1, H)
                
                # 获取提示令牌的嵌入
                prompt_embeds = self.bert.embeddings.word_embeddings(
                    prompt_input_ids.to(device)
                )  # (L_p, H)
            
            # 构造融合序列嵌入
            # [CLS] + O + [SEP] + prompt_embeds + [SEP]
            fused_embed = torch.cat([
                cls_embed,                          # (1, H)
                O[b],                               # (M, H)
                sep_embed,                          # (1, H)
                prompt_embeds,                      # (L_p, H)
                sep_embed,                          # (1, H)
            ], dim=0)  # (1 + M + 1 + L_p + 1, H)
            
            # 计算融合序列中 [MASK] 的位置
            # 位置 = 1（CLS） + M（视觉）+ 1（SEP） + mask_idx_in_prompt
            mask_pos_in_fused = 1 + num_visual + 1 + mask_idx_in_prompt
            
            fused_inputs_embeds_list.append(fused_embed)
            mask_pos_list.append(mask_pos_in_fused)
        
        # 批处理：填充到相同长度
        max_seq_len = max(emb.shape[0] for emb in fused_inputs_embeds_list)
        
        fused_inputs_embeds_padded = []
        attention_mask = []
        
        for emb, mask_pos in zip(fused_inputs_embeds_list, mask_pos_list):
            seq_len = emb.shape[0]
            pad_len = max_seq_len - seq_len
            
            # 填充嵌入
            if pad_len > 0:
                emb_padded = torch.cat([
                    emb,
                    torch.zeros(pad_len, hidden_size, device=device)
                ], dim=0)
            else:
                emb_padded = emb
            
            fused_inputs_embeds_padded.append(emb_padded)
            attention_mask.append([1] * seq_len + [0] * pad_len)
        
        # 堆叠为批次张量
        fused_inputs_embeds = torch.stack(fused_inputs_embeds_padded, dim=0)  # (B, max_seq_len, H)
        attention_mask = torch.tensor(attention_mask, device=device, dtype=torch.long)  # (B, max_seq_len)
        
        # 通过 BERT（仅获取最后一层隐藏状态）
        bert_output = self.bert(
            inputs_embeds=fused_inputs_embeds,
            attention_mask=attention_mask,
        )
        last_hidden_state = bert_output.last_hidden_state  # (B, max_seq_len, H)
        
        # 提取 [MASK] 位置的隐藏状态并计算逻辑
        mask_logits_list = []
        for b in range(batch_size):
            mask_pos = mask_pos_list[b]
            mask_hidden = last_hidden_state[b, mask_pos, :]  # (H,)
            mask_logits = self.mlm_head(mask_hidden)  # (vocab_size,)
            mask_logits_list.append(mask_logits)
        
        mask_logits = torch.stack(mask_logits_list, dim=0)  # (B, vocab_size)
        
        return mask_logits, mask_pos_list


class VisualTokenLoader:
    """
    视觉令牌加载和管理工具类。
    用于从各种来源加载、验证和处理视觉令牌。
    """
    
    @staticmethod
    def load_from_numpy(filepath: str) -> torch.Tensor:
        """
        从 numpy 文件加载视觉令牌。
        
        Args:
            filepath: .npy 文件路径
        
        Returns:
            torch.Tensor: 视觉令牌
        """
        O_np = np.load(filepath)
        return torch.from_numpy(O_np).float()
    
    @staticmethod
    def load_from_torch(filepath: str) -> torch.Tensor:
        """
        从 PyTorch 文件加载视觉令牌。
        
        Args:
            filepath: .pt 或 .pth 文件路径
        
        Returns:
            torch.Tensor: 视觉令牌
        """
        return torch.load(filepath)
    
    @staticmethod
    def create_dummy_visual_tokens(
        batch_size: int,
        num_visual: int,
        hidden_size: int = 768,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        创建虚拟视觉令牌用于测试。
        
        Args:
            batch_size: 批处理大小
            num_visual: 视觉令牌数
            hidden_size: 隐藏维度（默认 768 for BERT）
            device: 目标设备
        
        Returns:
            torch.Tensor: 随机初始化的视觉令牌 (B, M, H)
        """
        if device is None:
            device = torch.device("cpu")
        return torch.randn(batch_size, num_visual, hidden_size, device=device)
    
    @staticmethod
    def validate_visual_tokens(
        O: Union[torch.Tensor, np.ndarray],
        expected_hidden_size: int = 768,
    ) -> bool:
        """
        验证视觉令牌的格式和维度。
        
        Args:
            O: 视觉令牌
            expected_hidden_size: 期望的隐藏维度
        
        Returns:
            bool: 是否通过验证
        
        Raises:
            ValueError: 验证失败时抛出异常
        """
        if isinstance(O, np.ndarray):
            O = torch.from_numpy(O)
        
        if not isinstance(O, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor or numpy.ndarray, got {type(O)}")
        
        if O.dim() not in (2, 3):
            raise ValueError(f"Expected 2D or 3D tensor, got shape {O.shape}")
        
        if O.dim() == 3:
            _, _, hidden_size = O.shape
        else:  # dim == 2
            _, hidden_size = O.shape
        
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"Expected hidden_size {expected_hidden_size}, got {hidden_size}"
            )
        
        return True


# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模块
    fusion = H2RFPromptFusion(model_name="bert-base-uncased").to(device)
    
    # 准备测试数据
    S_list = [
        "This product is amazing and works great.",
        "The service was poor but food was delicious."
    ]
    R2_list = [
        "positive emotional indicators with strong satisfaction",
        "mixed sentiment with quality concerns"
    ]
    A_list = [
        "product quality",
        "service"
    ]
    
    # ============= 输入方式 1: 直接使用 torch.Tensor =============
    print("=" * 70)
    print("【输入方式 1】 torch.Tensor 格式 (批处理)")
    print("=" * 70)
    batch_size, num_visual, hidden_size = 2, 10, 768
    O_tensor = torch.randn(batch_size, num_visual, hidden_size, device=device)
    print(f"O 形状: {O_tensor.shape}")
    
    mask_logits_s, mask_pos_s = fusion(
        O=O_tensor,
        S_list=S_list,
        R2_list=R2_list,
        mode="sentence",
        template_id="Ps1"
    )
    print(f"✓ 句子级融合成功")
    print(f"  mask_logits 形状: {mask_logits_s.shape}")
    print(f"  mask 位置: {mask_pos_s}\n")
    
    # ============= 输入方式 2: numpy.ndarray 格式 =============
    print("=" * 70)
    print("【输入方式 2】 numpy.ndarray 格式")
    print("=" * 70)
    O_numpy = np.random.randn(batch_size, num_visual, hidden_size).astype(np.float32)
    print(f"O 形状: {O_numpy.shape}")
    
    mask_logits_a, mask_pos_a = fusion(
        O=O_numpy,
        S_list=S_list,
        R2_list=R2_list,
        mode="aspect",
        template_id="Pa1",
        A_list=A_list
    )
    print(f"✓ 方面级融合成功")
    print(f"  mask_logits 形状: {mask_logits_a.shape}")
    print(f"  mask 位置: {mask_pos_a}\n")
    
    # ============= 输入方式 3: 单样本输入 (自动扩展) =============
    print("=" * 70)
    print("【输入方式 3】 单样本 torch.Tensor (自动扩展为批处理)")
    print("=" * 70)
    O_single = torch.randn(num_visual, hidden_size, device=device)
    print(f"O 形状 (输入): {O_single.shape}")
    
    mask_logits_single, mask_pos_single = fusion(
        O=O_single,
        S_list=S_list[:1],
        R2_list=R2_list[:1],
        mode="sentence",
        template_id="Ps2"
    )
    print(f"✓ 单样本融合成功 (自动扩展为 (1, {num_visual}, {hidden_size}))")
    print(f"  mask_logits 形状: {mask_logits_single.shape}")
    print(f"  mask 位置: {mask_pos_single}\n")
    
    # ============= 使用 VisualTokenLoader 的示例 =============
    print("=" * 70)
    print("【使用 VisualTokenLoader 的示例】")
    print("=" * 70)
    
    O_dummy = VisualTokenLoader.create_dummy_visual_tokens(
        batch_size=2,
        num_visual=10,
        hidden_size=768,
        device=device
    )
    print(f"✓ 创建虚拟视觉令牌: {O_dummy.shape}")
    
    is_valid = VisualTokenLoader.validate_visual_tokens(O_dummy, expected_hidden_size=768)
    print(f"✓ 视觉令牌验证通过: {is_valid}\n")
    
    # ============= 输出说明 =============
    print("=" * 70)
    print("【输出说明】")
    print("=" * 70)
    print(f"mask_logits: 每个样本在 [MASK] 位置的词表预测分数")
    print(f"  形状: (batch_size, vocab_size) = {mask_logits_s.shape}")
    print(f"  可用于: 预测 [MASK] 应该是哪个词\n")
    print(f"mask_pos: 每个样本中 [MASK] 在融合序列中的位置")
    print(f"  说明: [CLS] + O + [SEP] + prompt + [SEP] 中的位置索引")