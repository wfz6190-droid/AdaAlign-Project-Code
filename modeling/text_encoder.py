"""
Text Encoder with PEFT support for SBIR
Supports CLIP and SiGLIP models with Adapter/LoRA
"""
import os
import torch
import torch.nn as nn
from transformers import CLIPTextModel, SiglipTextModel, AutoTokenizer
from modeling.adapter import AdapterBottleneck

# 自动配置HuggingFace镜像和代理设置（国内网络优化）
def _configure_huggingface_environment():
    """配置HuggingFace环境，禁用代理并使用镜像"""
    import os
    
    # 1. 清除所有代理环境变量
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 
                  'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY',
                  'ftp_proxy', 'FTP_PROXY', 'rsync_proxy', 'RSYNC_PROXY']
    
    cleared = []
    for var in proxy_vars:
        if var in os.environ:
            cleared.append(var)
            del os.environ[var]
    
    if cleared:
        print(f"⚠️  已清除 {len(cleared)} 个代理设置: {', '.join(cleared)}")
    
    # 2. 设置HuggingFace镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 确保不是离线模式
    
    # 3. 禁用requests的环境代理
    try:
        import requests
        # 强制requests不使用环境变量中的代理
        original_session = requests.Session
        
        def no_proxy_session():
            session = original_session()
            session.trust_env = False  # 关键：不信任环境变量
            return session
        
        requests.Session = no_proxy_session
        requests.sessions.Session = no_proxy_session
        print("✓ 已禁用requests代理")
    except:
        pass
    
    print(f"✓ HuggingFace镜像: {os.environ.get('HF_ENDPOINT')}")
    
# 执行配置
_configure_huggingface_environment()


class TextEncoderWithAdapter(nn.Module):
    """Text Encoder wrapper with Adapter support"""
    def __init__(self, model_name='openai/clip-vit-base-patch32', add_adapter=True, adapter_reduction=4, freeze_base=True, output_dim=1024):
        super(TextEncoderWithAdapter, self).__init__()
        self.model_name = model_name
        self.add_adapter = add_adapter
        self.output_dim = output_dim  # 目标输出维度（与视觉特征匹配）
        
        print(f"Loading text encoder: {model_name}...")
        
        # Load text encoder based on model type
        try:
            if 'clip' in model_name.lower():
                print(f"  正在从 {os.environ.get('HF_ENDPOINT', 'huggingface.co')} 加载CLIP模型...")
                print(f"  模型: {model_name}")
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_name,
                    resume_download=True,  # 支持断点续传
                    local_files_only=False  # 允许下载
                )
                self.hidden_size = self.text_encoder.config.hidden_size
                print(f"  ✓ CLIP模型加载成功，隐藏层维度: {self.hidden_size}")
            elif 'siglip' in model_name.lower():
                print(f"  正在从 {os.environ.get('HF_ENDPOINT', 'huggingface.co')} 加载SiGLIP模型...")
                print(f"  模型: {model_name}")
                from transformers import SiglipModel
                siglip_model = SiglipModel.from_pretrained(
                    model_name,
                    resume_download=True,
                    local_files_only=False
                )
                self.text_encoder = siglip_model.text_model
                self.hidden_size = self.text_encoder.config.hidden_size
                print(f"  ✓ SiGLIP模型加载成功，隐藏层维度: {self.hidden_size}")
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            print(f"\n❌ 加载文本编码器失败: {e}")
            print(f"\n💡 可能的解决方案:")
            print(f"  1. 检查网络连接")
            print(f"  2. 使用镜像源: export HF_ENDPOINT=https://hf-mirror.com")
            print(f"  3. 手动下载模型到本地: https://hf-mirror.com/{model_name}")
            print(f"  4. 如果已下载，设置缓存路径: export HF_HOME=/path/to/cache")
            raise
        
        # Add projection layer if output_dim differs from hidden_size
        if self.hidden_size != self.output_dim:
            self.projection = nn.Linear(self.hidden_size, self.output_dim, bias=False)
            print(f"  ✓ 添加投影层: {self.hidden_size} -> {self.output_dim}")
        else:
            self.projection = None
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Add adapters to transformer layers
        if self.add_adapter:
            self._inject_adapters(adapter_reduction)
    
    def _inject_adapters(self, reduction=4):
        """Inject adapter modules after each transformer layer"""
        # For CLIP/SiGLIP, adapters are added after MLP in each layer
        # CLIP uses text_model.encoder.layers, SiGLIP uses encoder.layers
        if hasattr(self.text_encoder, 'text_model'):
            encoder_layers = self.text_encoder.text_model.encoder.layers  # CLIP
        elif hasattr(self.text_encoder, 'encoder'):
            encoder_layers = self.text_encoder.encoder.layers  # SiGLIP
        else:
            raise AttributeError("Cannot find encoder layers in text model")
        for layer_idx, layer in enumerate(encoder_layers):
            # Add adapter after MLP
            layer.adapter = AdapterBottleneck(self.hidden_size, reduction=reduction)
            
            # Modify forward to include adapter
            original_forward = layer.forward
            
            def make_forward_with_adapter(original_fwd, adapter_module):
                def forward_with_adapter(*args, **kwargs):
                    outputs = original_fwd(*args, **kwargs)
                    if isinstance(outputs, tuple):
                        hidden_states = outputs[0]
                        hidden_states = adapter_module(hidden_states)
                        return (hidden_states,) + outputs[1:]
                    else:
                        return adapter_module(outputs)
                return forward_with_adapter
            
            layer.forward = make_forward_with_adapter(original_forward, layer.adapter)
        
        print(f"✓ Injected adapters into {len(encoder_layers)} transformer layers")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through text encoder"""
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Return pooled output (CLS token representation)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # For models without pooler, use last hidden state's first token
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        return embeddings
    
    def encode_text(self, texts, tokenizer, device='cuda'):
        """Encode text prompts into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get embeddings
        with torch.set_grad_enabled(self.training):
            embeddings = self.forward(input_ids, attention_mask)
        
        return embeddings


class TextEncoderWithLoRA(nn.Module):
    """Text Encoder wrapper with LoRA support"""
    def __init__(self, model_name='openai/clip-vit-base-patch32', lora_r=16, lora_alpha=16, 
                 lora_dropout=0.1, freeze_base=True, output_dim=1024):
        super(TextEncoderWithLoRA, self).__init__()
        from peft import LoraConfig, get_peft_model
        self.output_dim = output_dim
        
        # Load base model
        if 'clip' in model_name.lower():
            base_model = CLIPTextModel.from_pretrained(model_name)
            self.hidden_size = base_model.config.hidden_size
        elif 'siglip' in model_name.lower():
            from transformers import SiglipModel
            siglip_model = SiglipModel.from_pretrained(model_name)
            base_model = siglip_model.text_model
            self.hidden_size = base_model.config.hidden_size
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Apply to attention Q and V projections
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        # Apply LoRA
        self.text_encoder = get_peft_model(base_model, lora_config)
        self.text_encoder.print_trainable_parameters()
        
        # Add projection layer if needed
        if self.hidden_size != self.output_dim:
            self.projection = nn.Linear(self.hidden_size, self.output_dim, bias=False)
            print(f"  ✓ 添加投影层: {self.hidden_size} -> {self.output_dim}")
        else:
            self.projection = None
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through text encoder"""
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        return embeddings
    
    def encode_text(self, texts, tokenizer, device='cuda'):
        """Encode text prompts into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.set_grad_enabled(self.training):
            embeddings = self.forward(input_ids, attention_mask)
        
        return embeddings


def build_text_encoder(args):
    """Factory function to build text encoder with PEFT"""
    # Map arch_CLIP to HuggingFace model names
    # 使用本地缓存的CLIP Large模型（已存在于系统中）
    LOCAL_CACHED_MODEL = '/home/gpu/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
    
    model_map = {
        'RN50': LOCAL_CACHED_MODEL,  # 使用已缓存的CLIP Large
        'ViT-B/32': LOCAL_CACHED_MODEL,
        'ViT-B/16': LOCAL_CACHED_MODEL,
        'ViT-L/14': LOCAL_CACHED_MODEL,
        'siglip-base-p16': 'google/siglip-base-patch16-224',
    }
    
    model_name = model_map.get(args.arch_CLIP, 'openai/clip-vit-base-patch32')  # 默认使用CLIP
    print(f"Text encoder model: {model_name} (from arch_CLIP={args.arch_CLIP})")
    
    # 设置输出维度（与视觉特征维度匹配）
    output_dim = getattr(args, 'clip_feature', 1024)  # 默认1024维
    
    if args.text_lora:
        text_encoder = TextEncoderWithLoRA(
            model_name=model_name,
            lora_r=args.text_lora_r,
            lora_alpha=args.text_lora_alpha,
            lora_dropout=args.text_lora_dropout,
            freeze_base=True,
            output_dim=output_dim
        )
    elif args.text_adapter:
        text_encoder = TextEncoderWithAdapter(
            model_name=model_name,
            add_adapter=True,
            adapter_reduction=args.text_adapter_reduction,
            freeze_base=True,
            output_dim=output_dim
        )
    else:
        # Return None if no PEFT is applied to text encoder
        return None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return text_encoder, tokenizer

