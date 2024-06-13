from .eva_vit import create_eva_vit_g

def build_vision_encoder(vision_model, **kwargs):
    if vision_model == "eva_clip_g":
        return create_eva_vit_g(**kwargs)
    if vision_model == "med_vit":
        pass