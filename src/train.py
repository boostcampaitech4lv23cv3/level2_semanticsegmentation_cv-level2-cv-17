import datetime
import os
import warnings
from argparse import ArgumentParser
from time import time

import numpy as np
import torch
from tqdm import tqdm

import wandb
from dataset import make_dataloader
from inference import save_result
from network import define_network
from utils import add_hist, label_accuracy_score

warnings.filterwarnings("ignore")


N_CLASSES = 11
CKPT_DIR = "/opt/ml/input/code/saved/"
OUTPUT_DIR = "/opt/ml/input/code/submission/"
CLASSES = (
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

cvt_ext = lambda name, ext: name.split(".")[0] + ext


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "output_name", type=str, help="model checkpoint and inferenced file name"
    )
    parser.add_argument("-r", "--resume_training", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-e", "--num_epochs", type=int, default=20)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-s", "--early_stopping", action="store_true")
    parser.add_argument("-p", "--patience", type=int, default=7)
    parser.add_argument(
        "-c",
        "--ckpt_criterion",
        type=str,
        default="mIoU",
        help="model checkpoint saving mode. 'Loss' or 'mIoU', case insensitive",
    )
    parser.add_argument(
        "-i",
        "--inference",
        action="store_true",
        help="inference after training. defalt=False",
    )
    parser.add_argument(
        "-a",
        "--amp",
        action="store_true",
        help="apply Automatic Mixed Precision. default=False",
    )
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--val_every", type=int, default=1)

    args = parser.parse_args()

    return args


def save_model(model, file_name):
    # check_point = {"net": model.state_dict()}
    output_path = os.path.join(CKPT_DIR, file_name)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved in {CKPT_DIR}.")


def validation(epoch, model, data_loader, criterion, device):
    print(f"Starting validation #{epoch}...")

    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(tqdm(data_loader)):

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

        acc, acc_cls, mIoU, fwavacc, IoUs = label_accuracy_score(hist)

        avrg_loss = total_loss / cnt

        print(
            f"Validation #{epoch}: "
            + f"[Average Loss: {round(avrg_loss.item(), 4)}], "
            + f"[Accuracy: {round(acc, 4)}], "
            + f"[mIoU: {round(mIoU, 4)}]"
        )
        print(f"{'Class':>13} | {'IoU'}")
        for category, IoU in zip(CLASSES, IoUs):
            print(f"{category:>13} | {IoU:6.4f}")

    return avrg_loss, mIoU


def train(
    data_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    num_epochs,
    output_path,
    device,
    print_every,
    val_every,
    ckpt_crtn,
    early_stopping,
    patience,
    amp,
):
    best_loss = 999999
    best_mIoU = 0
    best_epoch = 0

    ckpt_crtn = ckpt_crtn.lower()
    assert ckpt_crtn in ("loss", "miou")

    # GradScaler prevents gradients to underflow
    scaler = torch.cuda.amp.GradScaler(True)

    for epoch in range(num_epochs):
        start_time = time()
        model.train()

        hist = np.zeros((N_CLASSES, N_CLASSES))
        for step, (images, masks, _) in enumerate(pbar := tqdm(data_loader)):
            pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            images = torch.stack(images)
            masks = torch.stack(masks).long()

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            # device 할당
            model = model.to(device)

            if amp:
                # Exits autocast before backward().
                # Backward passes under autocast are not recommended.
                with torch.cuda.amp.autocast(True):
                    outputs = model(images)["out"]
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=N_CLASSES)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # step 주기에 따른 loss 출력
            if (step + 1) % print_every == 0:
                pbar.set_postfix_str(
                    f"[Step: [{step+1}/{len(data_loader)}]], "
                    + f"[Loss: {round(loss.item(), 4)}], "
                    + f"[mIoU: {round(mIoU, 4)}]"
                )

                wandb.log(
                    {
                        "Step": step + 1,
                        "Epoch": epoch + 1,
                        "Loss": round(loss.item(), 4),
                        "mIoU": round(mIoU, 4),
                    }
                )

        delta = time() - start_time
        elapsed_time = datetime.timedelta(seconds=delta)
        print(f"Elapsed time: {str(elapsed_time)[:7]}")

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, mIoU = validation(
                epoch + 1, model, val_loader, criterion, device
            )
            if ckpt_crtn == "loss" and avrg_loss < best_loss:
                print(f"Lowest loss at epoch: {epoch + 1}")
                best_loss = avrg_loss
                best_epoch = epoch
                save_model(model, output_path)
            elif ckpt_crtn == "miou" and mIoU > best_mIoU:
                print(f"Highest mIoU at epoch: {epoch + 1}")
                best_mIoU = mIoU
                best_epoch = epoch
                save_model(model, output_path)

            # early stopping 적용 시 patience 동안 성능 개선이 없으면 종료
            if early_stopping:
                if epoch - best_epoch >= val_every * patience:
                    print(f"Early stopped. Saved Model: {best_epoch + 1} epoch")
                    break
                elif epoch != best_epoch:
                    count = (epoch - best_epoch) // val_every
                    print(f"Early stopping counter: {count}/{patience}")

    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"pytorch version: {torch.__version__}")
    print(f"CUDA availability: {torch.cuda.is_available()}")
    print(f"Name: {args.output_name}")
    print(f"Resume training: {args.resume_training}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Early stopping: {args.early_stopping}")
    print(f"Patience: {args.patience}")
    print(f"Checkpoint update criterion: {args.ckpt_criterion}")
    print(f"Inference test dataset after training: {args.inference}")
    print(f"Automatic mixed precision: {args.amp}")

    checkpoint_path = os.path.join(CKPT_DIR, cvt_ext(args.output_name, ".pt"))
    output_path = os.path.join(OUTPUT_DIR, cvt_ext(args.output_name, ".csv"))

    model, criterion, optimizer = define_network(
        pretrained=checkpoint_path if args.resume_training else True,
        learning_rate=args.learning_rate,
        device=device,
    )
    train_loader, val_loader, test_loader = make_dataloader(args.batch_size)

    wandb.init(
        entity="boostcamp-ai-tech-4-cv-17",
        project="Semantic Segmentation",
        name=args.output_name,
    )
    wandb.config.update(args)

    model = train(
        data_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        output_path=checkpoint_path,
        device=device,
        print_every=args.print_every,
        val_every=args.val_every,
        ckpt_crtn=args.ckpt_criterion,
        early_stopping=args.early_stopping,
        patience=args.patience,
        amp=args.amp,
    )

    if args.inference:
        save_result(model, device, test_loader, output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
