import os
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import wandb
from dataset import make_dataloader
from network import define_network
from utils import add_hist, label_accuracy_score

warnings.filterwarnings("ignore")

print(f"pytorch version: {torch.__version__}")
print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")

n_class = 11
device = "cuda" if torch.cuda.is_available() else "cpu"

categories = (
    "Backgroud",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)

class_list = {
    0: "Backgroud",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--saved_dir", type=str, default="/opt/ml/input/code/saved")
    parser.add_argument(
        "--file_name", type=str, default="fcn_resnet50_best_model(pretrained).pt"
    )
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=7)

    args = parser.parse_args()

    return args


def save_model(model, saved_dir, file_name="fcn_resnet50_best_model(pretrained).pt"):
    # check_point = {"net": model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def validation(epoch, model, data_loader, criterion, device):
    print(f"Start validation #{epoch}")
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):

            images = torch.stack(images)
            masks = torch.stack(masks).long()

            images, masks = images.to(device), masks.to(device)

            # device 할당
            model = model.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_class)

        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [
            {classes: round(IoU, 4)} for IoU, classes in zip(IoU, categories)
        ]

        avrg_loss = total_loss / cnt
        print(
            f"Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, \
                Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}"
        )
        print(f"IoU by class : {IoU_by_class}")

    return mIoU


def train(
    num_epochs,
    model,
    data_loader,
    val_loader,
    criterion,
    optimizer,
    saved_dir,
    file_name,
    print_every,
    val_every,
    device,
    early_stopping,
    patience,
):
    best_mIoU = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(pbar := tqdm(data_loader)):
            pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            images = torch.stack(images)
            masks = torch.stack(masks).long()

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # device 할당
            model = model.to(device)

            # inference
            outputs = model(images)["out"]

            # output mask visualizing
            vsoutput = torch.argmax(outputs[0].squeeze(), dim=0).detach().cpu().numpy()

            # wandb 이미지 저장
            wandb.log(
                {
                    "images": wandb.Image(
                        to_pil_image(images[0]),
                        masks={
                            "predictions": {
                                "mask_data": vsoutput,
                                "class_labels": class_list,
                            },
                            "ground_truth": {
                                "mask_data": masks[0].cpu().numpy(),
                                "class_labels": class_list,
                            },
                        },
                    )
                }
            )

            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # step 주기에 따른 loss 출력
            if (step + 1) % print_every == 0:
                pbar.set_postfix_str(
                    f"Step: [{step+1}/{len(data_loader)}], "
                    + f"Loss: {round(loss.item(),4)}, "
                    + f"mIoU: {round(mIoU,4)}"
                )
            wandb.log(
                {"train/loss": round(loss.item(), 4), "train/mIoU": round(mIoU, 4)}
            )
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            wandb.log({"val/mIoU": round(mIoU, 4)})
            if mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_mIoU = mIoU
                best_epoch = epoch
                save_model(model, saved_dir, file_name=file_name)

            # early stopping 적용 시 patience 동안 성능 개선이 없으면 종료
            if early_stopping:
                if epoch - best_epoch >= val_every * patience:
                    print(f"Early stopped. Saved Model : {best_epoch + 1} epoch")
                    break
                elif epoch != best_epoch:
                    count = (epoch - best_epoch) // val_every
                    print(f"Early stopping counter : {count}/{patience}")


def main(args):
    model, criterion, optimizer = define_network(pretrained=True, learning_rate=args.lr)
    train_loader, val_loader, test_loader = make_dataloader(args.batch_size)

    wandb.init(
        entity="boostcamp-ai-tech-4-cv-17",
        project="Semantic Segmentation",
        name="baseline_augmentation3_lkm"
        # name='baseline_mask_visualize2_test_epoch2_lkm'
    )
    wandb.config.update(args)

    train(
        num_epochs=args.num_epochs,
        model=model,
        data_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        saved_dir=args.saved_dir,
        file_name=args.file_name,
        print_every=args.print_every,
        val_every=args.val_every,
        device=device,
        early_stopping=args.early_stopping,
        patience=args.patience,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
