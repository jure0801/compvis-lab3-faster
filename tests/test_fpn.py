import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision
from fpn import FeaturePyramidNetwork
from resnet import resnet50
from utils import normalize_img
from torchvision.models.detection._utils import overwrite_eps


if __name__ == '__main__':
    backbone = resnet50(
        norm_layer=torchvision.ops.misc.FrozenBatchNorm2d
    )
    backbone.load_state_dict(torch.load('data/backbone_params.pth'))
    backbone.eval()
    overwrite_eps(backbone, 0.0)

    fpn = FeaturePyramidNetwork(
        input_keys_list=['res2', 'res3', 'res4', 'res5'],
        input_channels_list=[256, 512, 1024, 2048],
        output_channels=256,
        norm_layer=None,
        pool=True
    )
    fpn.load_state_dict(torch.load('data/fpn_params.pth'))
    fpn.eval()

    img = Image.open('bb44.png')
    img_pth = torch.from_numpy(np.array(img))
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_std = torch.tensor([0.229, 0.224, 0.225])
    img_pth = normalize_img(img_pth, image_mean, image_std)

    out_bb = backbone(img_pth.unsqueeze(0))
    out = fpn(out_bb)

    print("Testing FPN outputs...")
    for key in out:
        x = out[key].detach().numpy()
        x_target = np.load(f"data/{key}_bb44.npy")
        if x.shape == x_target.shape:
            print(f"Features {key} shape test: passed!")
            if np.allclose(x, x_target, atol=1e-5):
                print(f"Features {key} values test: passed!")
            else:
                print(f"Features {key} values test: failed!")
        else:
            print(f"Features {key} shape test: failed!")
        print("---------------------------------")
