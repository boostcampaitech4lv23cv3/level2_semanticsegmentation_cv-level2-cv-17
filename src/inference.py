import warnings
from argparse import ArgumentParser

import albumentations as A
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import make_dataloader
from network import define_network

warnings.filterwarnings("ignore")


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="/opt/ml/input/code/saved/fcn_resnet50_best_model(pretrained).pt",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/opt/ml/input/code/submission/fcn_resnet50_best_model(pretrained).csv",
    )

    args = parser.parse_args()

    return args


def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print("Start prediction.")

    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))["out"]
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed["mask"]
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i["file_name"] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array


def save_result(model, test_loader, output_path):
    # sample_submisson.csv 열기
    submission = pd.read_csv(
        "/opt/ml/input/code/submission/sample_submission.csv", index_col=None
    )

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append(
            {
                "image_id": file_name,
                "PredictionString": " ".join(str(e) for e in string.tolist()),
            },
            ignore_index=True,
        )

    # submission.csv로 저장
    submission.to_csv(output_path, index=False)


def main(args):
    train_loader, val_loader, test_loader = make_dataloader()
    model, criterion, optimizer = define_network(pretrained=args.model_path)
    model = model.to(device)
    save_result(model, test_loader, args.output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
