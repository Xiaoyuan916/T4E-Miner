from esm.models.esmc import ESMC 
import torch
import torch.nn as nn

class TFourEModel(nn.Module):
   
    def __init__(self, pretrained_model_path="esmc_300m"):
        super(TFourEModel, self).__init__()
        
        self.modelEsm = ESMC.from_pretrained(pretrained_model_path)
        
        try:
            
            self.tokenizer = self.modelEsm.tokenizer
        except AttributeError:
            raise RuntimeError("Could not find .tokenizer attribute on loaded ESMC model. Your esm library version might be incompatible.")
        

        self.dnn = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(960, 2) 
        )
        self._device = None
    
    @property
    def device(self):
        
        if self._device is None:
            try:
                self._device = next(self.modelEsm.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        return self._device
    
    def forward(self, data, return_repr=False):
        
        out_result = self._get_representations(data) 
        
        out_put = self.dnn(out_result.float()) 
        
        if return_repr:
            return out_put, out_result
        else:
            return out_put
        

        logits = self.dnn(seq_repr)
        return logits

    def _get_layers(self):
        
        try:
            return len(self.modelEsm.transformer.blocks)
        except AttributeError:
            print("Warning: Could not find self.modelEsm.transformer.blocks. Defaulting layers to 30.")
            return 30
    
    @property
    def layers(self):
        return self.get_layers()
    
    def get_layers(self):
        return self._get_layers() 
    
    def get_last_layer_idx(self):
        
        return self._get_layers() - 1
    
    
    def _get_representations(self, data):
        
        if not data or not isinstance(data, list) or not isinstance(data[0], (tuple, list)):
             raise ValueError(f"Data format error. Expected list of tuples [('name', 'SEQ')], but got: {data[0]}")
        
        sequences = [seq for name, seq in data]

        inputs_dict = self.tokenizer(
            sequences, 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            max_length=1024
        )
        
        tokens = inputs_dict["input_ids"].to(self.device)
            
        outputs = self.modelEsm(tokens)
        
        if hasattr(outputs, "embeddings"):
            out_result = outputs.embeddings[:, 0, :]
        
        else:
            print(f"Error: Output object (type: {type(outputs)}) does not have 'embeddings' attribute.")
            print(f"Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
            raise AttributeError("Could not find expected 'embeddings' tensor in ESMC model.")

        return out_result
    
    def get_representations(self, data):
        return self._get_representations(data)
    
    def get_names(self, data):
        if not data or not isinstance(data, list) or not isinstance(data[0], (tuple, list)):
             raise ValueError(f"Data format error. Expected list of tuples [('name', 'SEQ')], but got: {data[0]}")
        names = [name for name, seq in data]
        return names
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path="esmc_300m", state_dict_path=None):
        model = cls(pretrained_model_path)
        if state_dict_path is not None:
            print(f"Loading state dict from {state_dict_path}")
            model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        return model