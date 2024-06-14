from transformers import CLIPModel
import torch.nn as nn

class PubmedCLIPViT(nn.Module):
    def __init__(self):
        super(PubmedCLIPViT, self).__init__()
        self.model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32").vision_model
        self.num_features = 768

    def forward(self, x):
        h, w = x.shape[2:]
        assert h == 224 and w == 224, "The resolution of the image must be (224, 224)"
        return self.model(x).last_hidden_state
    
def create_pubmed_clip_vit(**kwargs):
    precision = kwargs.get("precision", "fp16")
    model = PubmedCLIPViT()
    if precision == "fp16":
        model = model.half()
    return model