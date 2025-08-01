import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OrthogonalAttentionModule(nn.Module):
    """
    高效的、向量化的三平面正交注意力实现。
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 输入的三个平面共享 Q, K, V 投影层
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        # 输出投影层
        self.to_out = nn.Linear(dim, dim)

    def forward(self, p_xy, p_xz, p_yz):
        # p_xy, p_xz, p_yz 的输入形状: (B, H, W, C)
        # 在DiT中，H=W，我们用 S (size) 表示
        B, S, _, C = p_xy.shape

        # 1. 将所有平面的特征一次性投影到 Q, K, V
        #    cat -> (B, 3, S, S, C); linear -> (B, 3, S, S, 3*C); unbind
        qkv = self.to_qkv(torch.stack([p_xy, p_xz, p_yz], dim=1)).reshape(B, 3, S, S, 3, self.num_heads, self.head_dim).permute(4, 0, 1, 5, 2, 3, 6)
        q, k, v = qkv[0], qkv[1], qkv[2] # Shape for each: (B, 3, num_heads, S, S, head_dim)

        # 分别获取每个平面的 q, k, v
        q_xy, k_xy, v_xy = q[:,0], k[:,0], v[:,0]
        q_xz, k_xz, v_xz = q[:,1], k[:,1], v[:,1]
        q_yz, k_yz, v_yz = q[:,2], k[:,2], v[:,2]

        # 2. 计算注意力分量 (核心向量化操作)
        
        # OAx(P_xy, P_xz): 沿 x 轴 (维度 S) 对齐
        # Q from P_xy, KV from P_xz
        # (B, H, S, S, D) -> (B*H, S, S, D) -> attention -> (B*H, S, S, D) -> (B, H, S, S, D)
        q_xy_x = q_xy.permute(0, 2, 1, 3, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        k_xz_x = k_xz.permute(0, 2, 1, 3, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        v_xz_x = v_xz.permute(0, 2, 1, 3, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        oax_on_pxy = F.scaled_dot_product_attention(q_xy_x, k_xz_x, v_xz_x)
        oax_on_pxy = oax_on_pxy.reshape(B, S, self.num_heads, S, self.head_dim).permute(0, 2, 1, 3, 4)

        # OAy(P_xy, P_yz): 沿 y 轴 (维度 S) 对齐, 需要先 permute
        # Q from P_xy, KV from P_yz
        q_xy_y = q_xy.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        k_yz_y = k_yz.permute(0, 2, 1, 3, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        v_yz_y = v_yz.permute(0, 2, 1, 3, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        oay_on_pxy = F.scaled_dot_product_attention(q_xy_y, k_yz_y, v_yz_y)
        oay_on_pxy = oay_on_pxy.reshape(B, S, self.num_heads, S, self.head_dim).permute(0, 2, 3, 1, 4)

        # ... 对称地计算其他平面的分量 ...
        # OAx(P_xz, P_xy)
        q_xz_x = q_xz.permute(0, 2, 1, 3, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        k_xy_x = k_xy.permute(0, 2, 1, 3, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        v_xy_x = v_xy.permute(0, 2, 1, 3, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        oax_on_pxz = F.scaled_dot_product_attention(q_xz_x, k_xy_x, v_xy_x)
        oax_on_pxz = oax_on_pxz.reshape(B, S, self.num_heads, S, self.head_dim).permute(0, 2, 1, 3, 4)
        
        # OAz(P_xz, P_yz) -> 沿 z 轴对齐，等价于 P_yz 的 y 轴
        q_xz_z = q_xz.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        k_yz_z = k_yz.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        v_yz_z = v_yz.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        oaz_on_pxz = F.scaled_dot_product_attention(q_xz_z, k_yz_z, v_yz_z)
        oaz_on_pxz = oaz_on_pxz.reshape(B, S, self.num_heads, S, self.head_dim).permute(0, 2, 3, 1, 4)
        
        # OAy(P_yz, P_xy)
        q_yz_y = q_yz.permute(0, 2, 1, 3, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        k_xy_y = k_xy.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        v_xy_y = v_xy.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        oay_on_pyz = F.scaled_dot_product_attention(q_yz_y, k_xy_y, v_xy_y)
        oay_on_pyz = oay_on_pyz.reshape(B, S, self.num_heads, S, self.head_dim).permute(0, 2, 1, 3, 4)

        # OAz(P_yz, P_xz)
        q_yz_z = q_yz.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        k_xz_z = k_xz.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        v_xz_z = v_xz.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, S, self.head_dim)
        oaz_on_pyz = F.scaled_dot_product_attention(q_yz_z, k_xz_z, v_xz_z)
        oaz_on_pyz = oaz_on_pyz.reshape(B, S, self.num_heads, S, self.head_dim).permute(0, 2, 3, 1, 4)

        # 3. 组合结果
        # (B, num_heads, S, S, head_dim) -> (B, S, S, C)
        def reshape_output(x):
            return x.permute(0, 2, 3, 1, 4).reshape(B, S, S, C)

        out_xy = self.to_out(reshape_output(oax_on_pxy) + reshape_output(oay_on_pxy))
        out_xz = self.to_out(reshape_output(oax_on_pxz) + reshape_output(oaz_on_pxz))
        out_yz = self.to_out(reshape_output(oay_on_pyz) + reshape_output(oaz_on_pyz))

        return out_xy, out_xz, out_yz