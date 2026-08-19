import os
import sys
import argparse
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import read_fasta, write_fasta
from utils.metric_utils import print_detailed_table
from utils.common_utils import setup_seed, Logger

from model import TFourEModel
from dataset import TrainingDataset



def parse_args():
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--folds_dir', type=str, default="data/train_data_new/dataset_split")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--last_layers', type=int, default=4) 
    parser.add_argument('--save_path', type=str, default="./result_manual_split")
    parser.add_argument('--patience', type=int, default=15)
    
    parser.add_argument('--val_neg_size', type=int, default=2000)
    parser.add_argument('--test_neg_total', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=24)
    return parser.parse_args()

def run_fold(fold_idx, args, device, test_ds):
    print(f"\n{'='*20} Fold {fold_idx} / 5 {'='*20}")
    
  
    fold_dir = os.path.join(args.folds_dir, "folds", f"fold_{fold_idx}")
    

    train_pos_path = os.path.join(fold_dir, "train_pos.fasta")
    train_neg_path = os.path.join(fold_dir, "train_neg.fasta") 
    
    val_pos_path = os.path.join(fold_dir, "val_pos.fasta")
    val_neg_path = os.path.join(fold_dir, "val_neg.fasta")    
    
    
    for p in [train_pos_path, train_neg_path, val_pos_path, val_neg_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[Error] Missing file: {p}\nPlease make sure step2_create_folds.py has completed successfully.")

    save_dir = os.path.join(args.save_path, f"fold_{fold_idx}")
    os.makedirs(save_dir, exist_ok=True)
    

    print("Loading model...")
    model = TFourEModel(pretrained_model_path=args.model_path)
    total_layers = model.get_layers()

    for name, param in model.named_parameters():
        param.requires_grad = False 

 
    for i in range(total_layers - args.last_layers, total_layers):
        layer_prefix = f"modelEsm.transformer.blocks.{i}"
        for name, param in model.named_parameters():
            if name.startswith(layer_prefix):
                param.requires_grad = True


    for name, param in model.named_parameters():
        if name.startswith("dnn"):
            param.requires_grad = True
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Unfrozen parameters: {trainable_params / 1e6:.2f}M")
   
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-5
    )

 
    train_ds = TrainingDataset(train_pos_path, train_neg_path)
    val_ds = TrainingDataset(val_pos_path, val_neg_path)
    

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate_fn)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=val_ds.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=test_ds.collate_fn)

 
    best_f1 = -1.0 
    patience_cnt = 0
    best_ckpt = os.path.join(save_dir, "best.pth")
    history = []


    for epoch in range(args.epoch):
        model.train()
        t_loss, steps = 0, 0
        
    
        for x, y in train_loader:
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item(); steps += 1
            
        # Validation
        model.eval()
        preds, truths = [], []
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for x, y in val_loader:
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item()
                val_steps += 1
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                truths.extend(y.cpu().numpy())
        
        avg_train_loss = t_loss / steps if steps > 0 else 0
        avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0
        
        print(f"Ep {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        val_f1 = print_detailed_table(truths, preds, f"Fold {fold_idx} Val Ep {epoch}")
        
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1": val_f1
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print("Early stopping.")
                break
    

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt))
    
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"Fold {fold_idx} Test"):
            y = y.to(device)
            logits = model(x)
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            truths.extend(y.cpu().numpy())
    
    print_detailed_table(truths, preds, f"Fold {fold_idx} FINAL TEST")

    print(f"[Info] Saving plotting data for Fold {fold_idx}...")
    plot_save_path = os.path.join(save_dir, f"plot_data_fold_{fold_idx}.pth")
    
    all_probs = []
    all_labels = []
    all_preds = []
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
           
            probs = torch.softmax(logits, dim=1)[:, 1] 
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
            all_preds.extend(preds.cpu().numpy())
            
    torch.save({
        "y_true": all_labels,
        "y_score": all_probs,
        "y_pred": all_preds,
        "history": history
    }, plot_save_path)
    print(f"[Info] Saved to: {plot_save_path}")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    
    setup_seed(args.random_seed)
    
   
    os.makedirs(args.save_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.save_path, f"train_log_{timestamp}.txt")
    print(f"[Info] Logging to: {log_path}")
    sys.stdout = Logger(log_path, sys.stdout)
    
 
    print(f"Preparing Fixed Test Set from {args.folds_dir}...")
    
  
    test_set_dir = os.path.join(args.folds_dir, "test_set")
    test_pos_path = os.path.join(test_set_dir, "test_pos.fasta")

    test_neg_path = os.path.join(test_set_dir, "test_neg.fasta")
    
    if not os.path.exists(test_pos_path) or not os.path.exists(test_neg_path):
        print(f"[Error] Test set files are missing. Expected paths:\n {test_pos_path}\n {test_neg_path}")
        sys.exit(1)
        
    print(f"[Info] Using Test Positive: {test_pos_path}")
    print(f"[Info] Using Test Negative: {test_neg_path}")
    

    test_ds = TrainingDataset(test_pos_path, test_neg_path)

    for fold_idx in range(1, 6):

        run_fold(fold_idx, args, device, test_ds)
