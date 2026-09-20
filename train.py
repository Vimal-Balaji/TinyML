"""
train.py
End-to-end pipeline:
  1. Train TeacherCNN (float32, full precision) on Speech Commands.
  2. Train StudentDSCNN with:
       - Knowledge distillation from the teacher
       - QAT (quantization-aware training), uniform int8 or mixed precision
       - Optimizer: Adam or SAM (compare both — pass --optimizer)
  3. Convert the QAT model to a truly quantized int8 model and evaluate
     the accuracy drop vs the float32 student — this is the number you
     report per row of the results table.

Usage:
  python train.py --optimizer adam --precision uniform
  python train.py --optimizer sam  --precision mixed
"""

import argparse
import copy
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.optim import Adam

from data import get_dataloaders, NUM_CLASSES, N_MFCC
from models import StudentDSCNN, TinyStudentDSCNN, TeacherCNN, count_params, model_size_bytes
from sam import SAM
from distillation import DistillationLoss
from quant_config import apply_mixed_qconfig, UNIFORM_INT8, MIXED_PRECISION

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Teacher training (plain, float32, no quantization, no distillation)
# ----------------------------------------------------------------------
def train_teacher(train_loader, val_loader, epochs=15, lr=1e-3):
    teacher = TeacherCNN(n_mfcc=N_MFCC, num_classes=NUM_CLASSES).to(DEVICE)
    opt = Adam(teacher.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        teacher.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = ce(teacher(x), y)
            loss.backward()
            opt.step()
        acc = evaluate(teacher, val_loader)
        print(f"[Teacher] epoch {epoch+1}/{epochs}  val_acc={acc:.4f}")
    return teacher


# ----------------------------------------------------------------------
# Student training: QAT + KD, optimizer swappable (Adam or SAM)
# ----------------------------------------------------------------------
def train_student_qat(train_loader, val_loader, teacher, optimizer_name="adam",
                       precision="uniform", model_size="tiny", epochs=20, lr=1e-3, rho=0.05):
    if model_size == "tiny":
        student = TinyStudentDSCNN(n_mfcc=N_MFCC, num_classes=NUM_CLASSES).to(DEVICE)
    else:
        student = StudentDSCNN(n_mfcc=N_MFCC, num_classes=NUM_CLASSES).to(DEVICE)

    student.eval()          # fuse_modules requires eval mode (needs stable BN stats to fold)
    student.fuse_model()
    student.train()         # switch back to train mode for QAT

    policy = UNIFORM_INT8 if precision == "uniform" else MIXED_PRECISION
    apply_mixed_qconfig(student, policy)
    tq.prepare_qat(student, inplace=True)   # inserts fake-quant modules

    teacher.eval()
    distill_loss_fn = DistillationLoss(alpha=0.5, temperature=4.0)

    if optimizer_name == "adam":
        optimizer = Adam(student.parameters(), lr=lr)
    elif optimizer_name == "sam":
        optimizer = SAM(student.parameters(), Adam, rho=rho, lr=lr)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'sam'")

    for epoch in range(epochs):
        student.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(x)

            if optimizer_name == "adam":
                optimizer.zero_grad()
                student_logits = student(x)
                loss = distill_loss_fn(student_logits, teacher_logits, y)
                loss.backward()
                optimizer.step()
            else:  # SAM needs a closure: two forward/backward passes per step
                def closure():
                    optimizer.zero_grad()
                    logits = student(x)
                    l = distill_loss_fn(logits, teacher_logits, y)
                    l.backward()
                    return l

                student_logits = student(x)
                loss = distill_loss_fn(student_logits, teacher_logits, y)
                loss.backward()
                optimizer.step(closure)

        acc = evaluate(student, val_loader)
        print(f"[Student QAT-{precision}-{optimizer_name}] epoch {epoch+1}/{epochs} val_acc={acc:.4f}")

    return student


# ----------------------------------------------------------------------
# Convert QAT model -> real quantized int8 model, and evaluate
# ----------------------------------------------------------------------
def convert_and_evaluate(qat_model, test_loader):
    qat_model.eval()
    qat_model_cpu = copy.deepcopy(qat_model).to("cpu")
    quantized_model = tq.convert(qat_model_cpu, inplace=False)

    fp_acc = evaluate(qat_model, test_loader)                    # fake-quantized (simulated) accuracy
    int_acc = evaluate(quantized_model, test_loader, device="cpu")  # truly quantized int8 accuracy
    print(f"Simulated (fake-quant) test acc: {fp_acc:.4f}")
    print(f"Real int8-converted test acc:    {int_acc:.4f}")
    print(f"Accuracy drop from quantization: {fp_acc - int_acc:.4f}")

    save_path = "student_int8.pth"
    torch.save(quantized_model.state_dict(), save_path)

    import os
    size_bytes = os.path.getsize(save_path)
    print(f"Saved quantized model to {save_path}")
    print(f"File size: {size_bytes} bytes ({size_bytes/1024:.2f} KB)")

    return quantized_model, fp_acc, int_acc


@torch.no_grad()
def evaluate(model, loader, device=None):
    device = device or DEVICE
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["adam", "sam"], default="adam")
    parser.add_argument("--precision", choices=["uniform", "mixed"], default="uniform")
    parser.add_argument("--model_size", choices=["tiny", "full"], default="tiny",
                         help="tiny = ~4KB int8 target (width=22), full = original larger student (width=32)")
    parser.add_argument("--teacher_epochs", type=int, default=15)
    parser.add_argument("--student_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    print("=== Training teacher ===")
    teacher = train_teacher(train_loader, val_loader, epochs=args.teacher_epochs)
    print(f"Teacher params: {count_params(teacher):,}")

    print(f"=== Training student (optimizer={args.optimizer}, precision={args.precision}, size={args.model_size}) ===")
    student_qat = train_student_qat(train_loader, val_loader, teacher,
                                     optimizer_name=args.optimizer,
                                     precision=args.precision,
                                     model_size=args.model_size,
                                     epochs=args.student_epochs)
    n_params = count_params(student_qat)
    print(f"Student params: {n_params:,}")
    print(f"Estimated size @ int8: {model_size_bytes(student_qat, 8):.0f} bytes")
    print(f"Estimated size @ int4: {model_size_bytes(student_qat, 4):.0f} bytes")

    print("=== Converting to true int8 and evaluating ===")
    convert_and_evaluate(student_qat, test_loader)


if __name__ == "__main__":
    main()