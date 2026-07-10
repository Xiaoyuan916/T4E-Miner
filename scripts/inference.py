import os
import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

script_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.abspath(os.path.join(script_dir, '..')) 
sys.path.insert(0, project_root)

try:
    
    from model import TFourEModel 
except ImportError as e:
    print(f"Error: failed to import TFourEModel.")
    print(f"Please make sure your 'model.py' file is located under '{project_root}'.")
    print(f"Original error: {e}")
    sys.exit(1)


def parse_fasta(file_path):
   
    sequences = []
    seq_id = None
    current_seq_parts = []
    
    print(f"Parsing FASTA file: {file_path} ...")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  
                
                if line.startswith(">"):
                   
                    if seq_id is not None and current_seq_parts:
                        sequences.append((seq_id, "".join(current_seq_parts)))
                    
                    
                    seq_id = line[1:]   
                    current_seq_parts = []
                else:
                 
                    if seq_id is not None: 
                        current_seq_parts.append(line)
                   
        
     
        if seq_id is not None and current_seq_parts:
            sequences.append((seq_id, "".join(current_seq_parts)))
            
    except Exception as e:
        print(f"Error while reading FASTA file: {e}")
        return [] 

    print(f"Parsing completed. A total of {len(sequences)} sequences were loaded.")
    return sequences

class SimpleFastaDataset(Dataset):

    def __init__(self, fasta_data):

        self.data = fasta_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        return self.data[idx]

def simple_collate_fn(batch):

    return batch

# ----------------------------------------------------------------


def parse_args():

    parser = argparse.ArgumentParser(description="Run model inference and save scores for all sequences.")
    parser.add_argument('--model_path', type=str, required=True, 
                        help="Name or path of the pretrained model, for example 'esmc_300m'.")
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help="Path to the fine-tuned .pth checkpoint file.")
    parser.add_argument('--inference_data', type=str, required=True, 
                        help="Path to the input .fasta file for inference.")
    parser.add_argument('--output_path', type=str, required=True, 
                        help="Directory path for saving inference results, for example '.../inference_result'.")
    parser.add_argument('--batch_size', type=int, default=64, 
                        help="Batch size used during inference.")
    args = parser.parse_args()
    return args

def run_inference():
 
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(args.output_path, exist_ok=True)
    output_score_file = os.path.join(args.output_path, "all_inference_scores.tsv")


    print(f"Loading model from {args.model_path}...")
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    
    model = TFourEModel.from_pretrained(
        pretrained_model_path=args.model_path, 
        state_dict_path=args.checkpoint_path
    )
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")


    print(f"Reading candidate data from {args.inference_data}...")
    

    all_sequences = parse_fasta(args.inference_data) 
    
    if not all_sequences:
        print("Error: no sequences were loaded from the FASTA file. Please check the file path or file content.")
        return


    inference_dataset = SimpleFastaDataset(all_sequences) 
    

    inference_dataloader = DataLoader(
        inference_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=simple_collate_fn,
        drop_last=False, 
        pin_memory=True
    )
    

    print(f"Found {len(inference_dataset)} sequences to process.")


    print(f"Running inference... Writing all scores to {output_score_file}")
    
    total_sequences = 0
    predicted_positives = 0

    try:
        with open(output_score_file, "w") as f_out:
            f_out.write(
                "Sequence_ID\t"
                "Score_Class_0\tScore_Class_1\t"
                "Prob_Class_0\tProb_Class_1\t"
                "Predicted_Class\n"
            )
            
            with torch.no_grad():
                for content in tqdm(inference_dataloader, desc="Inference Progress"):
                    if not content:
                        continue
                    
                    logits = model(content) 
                    probs = F.softmax(logits, dim=1) 
                    predictions = torch.argmax(logits, dim=1) 

                    scores_0 = logits[:, 0].cpu().numpy()
                    scores_1 = logits[:, 1].cpu().numpy()
                    probs_0 = probs[:, 0].cpu().numpy()
                    probs_1 = probs[:, 1].cpu().numpy()

                    for i in range(len(content)):
                        seq_id = content[i][0]
                        pred_class = predictions[i].item()

                        f_out.write(
                            f"{seq_id}\t"
                            f"{scores_0[i]:.6f}\t{scores_1[i]:.6f}\t"
                            f"{probs_0[i]:.6f}\t{probs_1[i]:.6f}\t"
                            f"{pred_class}\n"
                        )
                        
                        if pred_class == 1:
                            predicted_positives += 1
                        total_sequences += 1

        print("\n--- Inference completed ---")
        print(f"Successfully processed {total_sequences} sequences.")
        print(f"{predicted_positives} sequences were predicted as positives (Class 1).")
        print(f"All scores and prediction results were saved to: {output_score_file}")

    except Exception as e:
        print(f"\nAn error occurred during inference or file writing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_inference()