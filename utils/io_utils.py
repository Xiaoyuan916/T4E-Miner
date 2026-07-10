import os

def read_fasta(file_path):
    
    if not os.path.exists(file_path): 
        print(f"[Warning] File not found: {file_path}")
        return []
    
    data = []
    with open(file_path, "r") as f:
        header = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if header is not None:
                    
                    data.append((header, "".join(seq_parts)))
                header = line[1:] 
                seq_parts = []
            else:
                seq_parts.append(line)
        
        
        if header is not None:
            data.append((header, "".join(seq_parts)))
            
    return data

def write_fasta(sequences, file_path):
    
    
    with open(file_path, "w") as f:
        for i, item in enumerate(sequences):
            if isinstance(item, tuple):
                
                head, seq = item
                f.write(f">{head}\n{seq}\n")
            else:
                
                f.write(f">seq_{i}\n{item}\n")
