import argparse
import json
from medlvlm.datasets.datasets.vindrcxr_dataset import VinDrCXRDataset
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import torch
from medlvlm.common.eval_utils import prepare_texts
from medlvlm.common.registry import registry
from medlvlm.common.config import Config
from medlvlm.conversation.conversation import Conversation, SeparatorStyle

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args

def init_model(cfg):
    print('Initialization Model')
    cfg = Config(cfg)
    # cfg.model_cfg.ckpt = args.ckpt
    # cfg.model_cfg.lora_r = args.lora_r
    # cfg.model_cfg.lora_alpha = args.lora_alpha

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

#     import pudb; pudb.set_trace()
    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    text_processor_cfg = cfg.datasets_cfg.get(key).text_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
    print('Initialization Finished')
    return model, vis_processor, text_processor

args = parse_args()
cfg = Config(args)
model, vis_processor, text_processor = init_model(args)
model.eval()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

conv_temp = CONV_VISION.copy()

for dataset in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open(eval_file_path, "r") as f:
        vindrcxr = json.load(f)

    data = VinDrCXRDataset(
        vis_processor=vis_processor,
        text_processor=text_processor,
        ann_path=eval_file_path,
        vis_root=img_path
    )

    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    predict = defaultdict(list)

    for batch in eval_dataloader:
        image = batch["image"].half()
        instruction_input = batch["instruction_input"]
        answer = batch["answer"]
        image_id = batch["image_id"]
        texts = prepare_texts(instruction_input, conv_temp)
        answers = model.generate(images=image, texts=instruction_input, max_new_tokens=max_new_tokens, do_sample=False)
        break