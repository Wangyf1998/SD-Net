from xmodaler.utils.registry import Registry

CLIP_REGISTRY = Registry("CLIP")
CLIP_REGISTRY.__doc__ = """
Registry for clipfilter
"""

def build_clipfilter(cfg):
    clip_filter = CLIP_REGISTRY.get(cfg.MODEL.CLIP)(cfg) if len(cfg.MODEL.CLIP) > 0 else None
    return clip_filter

def add_clip_config(cfg, tmp_cfg):
    if len(tmp_cfg.MODEL.CLIP) > 0:
        CLIP_REGISTRY.get(tmp_cfg.MODEL.CLIP).add_config(cfg)