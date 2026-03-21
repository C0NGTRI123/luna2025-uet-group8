import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights

# =============================================================================
# 1. DROP PATH (Stochastic Depth)
# =============================================================================
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# =============================================================================
# 2. SE BLOCK
# =============================================================================
class SEBlock3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

# =============================================================================
# 3. GEGLU
# =============================================================================
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

# =============================================================================
# 4. TRANSFORMER BLOCK
# =============================================================================
class ImprovedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.layer_scale1 = nn.Parameter(torch.ones(embed_dim) * 1e-5)

        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dùng GEGLU (Input Linear nhân đôi dimension để split)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim * 2), 
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.layer_scale2 = nn.Parameter(torch.ones(embed_dim) * 1e-5)

    def forward(self, x):
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y)
        x = x + self.drop_path1(self.layer_scale1 * attn_out)
        
        y2 = self.norm2(x)
        ff_out = self.ff(y2)
        x = x + self.drop_path2(self.layer_scale2 * ff_out)
        return x

# =============================================================================
# 5. MAIN MODEL: PULSE 3D V2
# =============================================================================
class Pulse3D_v2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 1, 
        pool_size=(8, 4, 4),   
        dropout_prob: float = 0.1,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        ff_ratio: int = 4,
        drop_path_rate: float = 0.2,
        freeze_bn: bool = False,
    ):
        super().__init__()
        # --- A. BACKBONE (ResNet3D-18) ---
        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        
        pretrained_first_conv = self.backbone.stem[0].weight.clone()
        self.backbone.stem[0] = nn.Conv3d(
            input_channels, 64,
            kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False
        )
        # Tính trung bình theo chiều channels (dim=1)
        if input_channels == 1:
             self.backbone.stem[0].weight.data = pretrained_first_conv.mean(dim=1, keepdim=True)
        
        # Bỏ Classification Head
        orig_in = self.backbone.fc.in_features # thường là 512
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # --- B. ATTENTION & TOKENS ---
        self.se_block = SEBlock3D(orig_in) 
        self.embed_dim = orig_in
        self.pool_size = pool_size

        # --- C. TRANSFORMER ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_pe = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        
        # Positional Embedding khởi tạo
        num_patches = pool_size[0] * pool_size[1] * pool_size[2]
        self.pe_3d = nn.Parameter(torch.randn(1, num_patches, self.embed_dim) * 0.02)
        
        ff_dim = self.embed_dim * ff_ratio
        dpr = torch.linspace(0, drop_path_rate, num_transformer_layers).tolist()
        
        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerBlock(self.embed_dim, num_heads, ff_dim, dropout_prob, dpr[i])
            for i in range(num_transformer_layers)
        ])
        self.pos_dropout = nn.Dropout(dropout_prob)

        # --- D. HEAD ---
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.embed_dim, num_classes)
        )

        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward_features(self, x):
        # Forward qua các tầng của ResNet
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feat = self.backbone.layer4(x) 
        
        feat = self.se_block(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Channels, Depth, Height, Width]
        
        # 1. Trích xuất đặc trưng 3D
        feat = self.forward_features(x) # Output shape: (B, 512, T, H, W)
        B, C, T, H, W = feat.shape
        
        # 2. Flatten thành chuỗi Tokens
        tokens = feat.flatten(2).transpose(1, 2) # (B, N, C)
        
        pe = self.pe_3d

        tokens = tokens + pe
        tokens = self.pos_dropout(tokens)

        # 4. Transformer Encoder
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pe
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        # 5. Classification
        cls_out = tokens[:, 0]
        return self.head(cls_out)

if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pulse3D_v2(input_channels=1).to(device)
    
    inp = torch.randn(2, 1, 64, 64, 64).to(device)
    out = model(inp)
    print(f"✅ Model check passed! Input: {inp.shape} -> Output: {out.shape}")