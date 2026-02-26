#-*- coding:utf-8 -*-
# *Main part of the code is adopted from: https://github.com/openai/guided-diffusion
# This variant keeps optional Cross-Slice Attention (CSA) and adds an optional Axial Bottleneck Block.

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .modules import *

NUM_CLASSES = 1


# -------------------------------
# Cross-slice attention (2D) — unchanged from your CSA version
class CrossSliceAttention2D(nn.Module):
    def __init__(self, q_channels, heads=4, head_dim=48, mask_gate=False):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.mask_gate = mask_gate

        self.q_proj = nn.Conv2d(q_channels, self.inner_dim, kernel_size=1, bias=True)
        self.k_proj = nn.LazyConv2d(self.inner_dim, kernel_size=1, bias=True)
        self.v_proj = nn.LazyConv2d(self.inner_dim, kernel_size=1, bias=True)
        self.out = nn.Conv2d(self.inner_dim, q_channels, kernel_size=1, bias=True)

        self.q_norm = normalization(q_channels)
        self.kv_norm = nn.Identity()
        self.scale = (head_dim) ** -0.5

    def forward(self, q_feat, kv_feat, soft_mask=None):
        """
        q_feat : [B, Cq, H, W]
        kv_feat: [B, Ck, H, W]
        soft_mask: [B, 1, H, W] in [0,1], optional
        """
        B, Cq, H, W = q_feat.shape
        _, Ck, Hk, Wk = kv_feat.shape
        assert H == Hk and W == Wk, "CSA: spatial size mismatch"

        q = self.q_proj(self.q_norm(q_feat))
        k = self.k_proj(self.kv_norm(kv_feat))
        v = self.v_proj(self.kv_norm(kv_feat))

        if self.mask_gate and soft_mask is not None:
            m = F.avg_pool2d(soft_mask.clamp(0,1), kernel_size=3, stride=1, padding=1)
            k = k * m
            v = v * m

        def reshape_mh(t):
            B, C, H, W = t.shape
            t = t.view(B, self.heads, self.head_dim, H * W)
            t = t.permute(0,1,3,2).reshape(B * self.heads, H * W, self.head_dim)
            return t

        qh = reshape_mh(q)
        kh = reshape_mh(k)
        vh = reshape_mh(v)

        attn = th.softmax((qh @ kh.transpose(1,2)) * self.scale, dim=-1)
        out = attn @ vh
        out = out.view(B, self.heads, H * W, self.head_dim).permute(0,1,3,2)
        out = out.reshape(B, self.inner_dim, H, W)
        out = self.out(out) + q_feat
        return out


# -------------------------------
# Axial Bottleneck (depthwise 3D conv along z, very light)
class AxialBottleneck3D(nn.Module):
    """
    Mix short z-context at bottleneck with a depthwise 3D conv (kz x 1 x 1)
    followed by a pointwise 1x1x1 conv. Returns the center slice (current) with a residual.
    """
    def __init__(self, channels, z_kernel=3):
        super().__init__()
        assert z_kernel % 2 == 1, "z_kernel must be odd"
        pad = z_kernel // 2
        self.dw = nn.Conv3d(channels, channels, kernel_size=(z_kernel,1,1),
                            padding=(pad,0,0), groups=channels, bias=True)
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.norm = normalization(channels)
        self.act = nn.SiLU()

    def forward(self, zstack):  # [B, C, Z, H, W]
        h0 = zstack  # residual path in 3D
        x = self.norm(zstack)
        x = self.act(x)
        x = self.dw(x)
        x = self.pw(x)
        x = x + h0
        # take the center z slice as the updated current-slice feature
        zc = zstack.shape[2] // 2
        return x[:, :, zc, :, :]  # [B, C, H, W]


class UNetModel(nn.Module):
    """
    UNet with optional Cross-Slice Attention (CSA) at selected downsample factors (csa_ds),
    and optional Axial Bottleneck Block at the middle stage.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        # ---- CSA knobs ----
        use_csa=False,
        csa_ds=(8,16),
        csa_heads=4,
        csa_head_dim=48,
        csa_kv_bands="LL",
        csa_mask_gate=False,
        # ---- Axial Bottleneck knobs ----
        use_axial_bottleneck=False,
        axial_depth=5,           # desired depth of z-window (we'll cap by available neighbors)
        axial_z_kernel=3,        # 3x1x1 depthwise kernel
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # CSA config
        self.use_csa = bool(use_csa)
        if isinstance(csa_ds, str):
            csa_ds = tuple(int(x) for x in csa_ds.split(",") if x.strip() != "")
        self.csa_ds = set(csa_ds)
        self.csa_heads = int(csa_heads)
        self.csa_head_dim = int(csa_head_dim)
        self.csa_mask_gate = bool(csa_mask_gate)
        self.csa_kv_bands = csa_kv_bands

        # Axial bottleneck config
        self.use_axial_bottleneck = bool(use_axial_bottleneck)
        self.axial_depth = int(axial_depth)
        self.axial_block = None  # created after channels are known
        self.axial_z_kernel = int(axial_z_kernel)

        # Helper: number of bands per neighbor in your cond (e.g., "LL,LH,HL" -> 3)
        self._bands_per_neighbor = max(1, len([b for b in str(self.csa_kv_bands).split(",") if b.strip() != ""]))

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        # CSA placeholders (encoder/decoder)
        self._input_csa = nn.ModuleDict()
        self._output_csa = nn.ModuleDict()

        # ---------- encoder ----------
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint,
                            num_heads=num_heads, num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    if self.use_csa and (ds in self.csa_ds):
                        layers.append(nn.Identity())
                        self._input_csa[str(len(self.input_blocks))] = nn.Identity()
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch,
                            dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm, down=True,
                        ) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # ---------- middle ----------
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint,
                num_heads=num_heads, num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # === FIX: create axial components at the bottleneck channel width ===
        self._bottleneck_ch = ch
        if self.use_axial_bottleneck:
            self.axial_block = AxialBottleneck3D(channels=self._bottleneck_ch, z_kernel=self.axial_z_kernel)
            # reducer: (bands_per_neighbor) → (bottleneck channels)
            self._nb_reduce = nn.Conv2d(self._bands_per_neighbor, self._bottleneck_ch, kernel_size=1, bias=True)
        else:
            self.axial_block = None
            self._nb_reduce = None

        # Axial bottleneck (built now that ch is known)
        if self.use_axial_bottleneck:
            self.axial_block = AxialBottleneck3D(channels=ch, z_kernel=self.axial_z_kernel)

        # ---------- decoder ----------
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample, num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    if self.use_csa and (ds in self.csa_ds):
                        layers.append(nn.Identity())
                        self._output_csa[str(len(self.output_blocks))] = nn.Identity()

                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch,
                            dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm, up=True,
                        ) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

        # finalize CSA placeholders now that channels are known
        self._build_csa_modules()

        # band reducer to turn each neighbor's band-group into bottleneck channels (shared across neighbors)
        # Only needed for axial bottleneck, created here to know 'ch' and bands-per-neighbor.
        #if self.use_axial_bottleneck:
        #    self._nb_reduce = nn.Conv2d(self._bands_per_neighbor, ch, kernel_size=1, bias=True)

    # Build CSA modules with the correct in/out channels
    def _build_csa_modules(self):
        if not self.use_csa:
            return
        for key in list(self._input_csa.keys()):
            idx = int(key)
            block = self.input_blocks[idx]
            ch = self._infer_block_out_channels(block)
            self._input_csa[key] = CrossSliceAttention2D(
                q_channels=ch,
                heads=self.csa_heads,
                head_dim=self.csa_head_dim,
                mask_gate=self.csa_mask_gate
            )
        for key in list(self._output_csa.keys()):
            idx = int(key)
            block = self.output_blocks[idx]
            ch = self._infer_block_out_channels(block)
            self._output_csa[key] = CrossSliceAttention2D(
                q_channels=ch,
                heads=self.csa_heads,
                head_dim=self.csa_head_dim,
                mask_gate=self.csa_mask_gate
            )

    @staticmethod
    def _infer_block_out_channels(block: nn.Module):
        ch = None
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                ch = m.out_channels
        if ch is None:
            raise RuntimeError("Could not infer block output channels for CSA wiring.")
        return ch

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def _prep_kv(self, kv_src, target_hw):
        B, Cn, H, W = kv_src.shape
        Ht, Wt = target_hw
        if (H, W) != (Ht, Wt):
            kv_src = F.interpolate(kv_src, size=(Ht, Wt), mode="bilinear", align_corners=False)
        return kv_src

    def _build_axial_stack(self, h, neighbors, mask=None):
        """
        Build [B, C, Z, H, W] for axial mixing at bottleneck.
        neighbors: [B, Cn, Hb, Wb], where Cn = num_neighbors * bands_per_neighbor
        We group bands per neighbor, reduce bands -> C via 1x1, and stack neighbors + current h.
        """
        B, Cn, Hb, Wb = neighbors.shape
        Ht, Wt = h.shape[-2:]
        if (Hb, Wb) != (Ht, Wt):
            neighbors = F.interpolate(neighbors, size=(Ht, Wt), mode="bilinear", align_corners=False)

        bands = self._bands_per_neighbor
        assert Cn % bands == 0, "neighbors channels must be divisible by bands-per-neighbor"
        n_nb = Cn // bands

        z_slices = []
        for i in range(n_nb):
            sl = neighbors[:, i*bands:(i+1)*bands]         # [B, bands, H, W]
            sl = self._nb_reduce(sl)                        # [B, C, H, W]
            z_slices.append(sl)
        # append current feature last → center index = n_nb//2 if symmetric; we take final center = len(z_slices)
        z_slices.append(h)                                   # current slice as last position

        zstack = th.stack(z_slices, dim=2)                   # [B, C, Z, H, W]
        # Optionally, soft mask could be used to gate neighbor slices before stack if desired (not required here)
        return zstack

    def forward(self, x, timesteps, y=None, neighbors=None, neighbor_mask=None):
        """
        x: [B, C, H, W], neighbors: [B, Cn, H, W] (coeff-res), neighbor_mask: [B,1,H,W] in [-1,1] or [0,1]
        """
        assert (y is not None) == (self.num_classes is not None), "class-conditional misuse"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)

        # ---------- encoder with optional CSA ----------
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if self.use_csa and (str(i) in self._input_csa) and (neighbors is not None):
                kv = self._prep_kv(neighbors.type_as(h), target_hw=h.shape[-2:])
                m = None
                if (neighbor_mask is not None) and self.csa_mask_gate:
                    m = F.interpolate(((neighbor_mask+1)/2).clamp(0,1).type_as(h), size=h.shape[-2:], mode="bilinear", align_corners=False)
                h = self._input_csa[str(i)](h, kv, soft_mask=m)
            hs.append(h)

        # ---------- middle ----------
        h = self.middle_block(h, emb)

        # Optional axial bottleneck mixing
        if self.use_axial_bottleneck and (neighbors is not None):
            kv = self._prep_kv(neighbors.type_as(h), target_hw=h.shape[-2:])
            zstack = self._build_axial_stack(h, kv, neighbor_mask)
            h = self.axial_block(zstack)

        # ---------- decoder with optional CSA ----------
        for i, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            if self.use_csa and (str(i) in self._output_csa) and (neighbors is not None):
                kv = self._prep_kv(neighbors.type_as(h), target_hw=h.shape[-2:])
                m = None
                if (neighbor_mask is not None) and self.csa_mask_gate:
                    m = F.interpolate(((neighbor_mask+1)/2).clamp(0,1).type_as(h), size=h.shape[-2:], mode="bilinear", align_corners=False)
                h = self._output_csa[str(i)](h, kv, soft_mask=m)

        h = h.type(x.dtype)
        return self.out(h)


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    in_channels=1,
    out_channels=1,
    # ---- CSA knobs ----
    use_csa=False,
    csa_ds="8,16",
    csa_heads=4,
    csa_head_dim=48,
    csa_kv_bands="LL",
    csa_mask_gate=False,
    # ---- Axial Bottleneck knobs ----
    use_axial_bottleneck=False,
    axial_depth=5,
    axial_z_kernel=3,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(1*out_channels if not learn_sigma else 2*out_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        dims=2,
        # CSA knobs
        use_csa=use_csa,
        csa_ds=csa_ds,
        csa_heads=csa_heads,
        csa_head_dim=csa_head_dim,
        csa_kv_bands=csa_kv_bands,
        csa_mask_gate=csa_mask_gate,
        # Axial bottleneck knobs
        use_axial_bottleneck=use_axial_bottleneck,
        axial_depth=axial_depth,
        axial_z_kernel=axial_z_kernel,
    )
