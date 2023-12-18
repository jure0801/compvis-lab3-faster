import torch
from utils import decode_boxes

if __name__ == '__main__':
    codes = torch.tensor(
        [[0.0000, -0.0333, -0.1823, -0.1178],
         [-0.0128, 0.0135, 0.0253, 0.0780],
         [0.1538, 0.1296, 0.1431, 0.1054]]
    )
    proposals = torch.tensor(
        [[90, 95, 210, 320],
         [5, 5, 200, 190],
         [340, 180, 470, 450]]
    )

    boxes = torch.tensor(
        [[100, 100, 200, 300],
         [0, 0, 200, 200],
         [350, 200, 500, 500.]]
    )

    decoded_boxes = decode_boxes(codes, proposals, (1.0, 1.0, 1.0, 1.0))
    print("Testing box decoding...")

    print("Running simple test...")
    if torch.allclose(decoded_boxes, boxes, atol=0.1, rtol=0.1):
        print("Test passed!")
    else:
        print("Test failed!")
