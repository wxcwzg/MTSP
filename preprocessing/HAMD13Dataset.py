"""
HAMD-13 dataset loading and preprocessing.
Supports CIDH and PDCH datasets.
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import re
import hashlib
import pickle


class HAMD13Dataset(Dataset):
    """Dataset for HAMD-13 depression assessment."""
    
    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "cidh",
        data_dir: str = "../data",
        sum_labels: bool = False,
        bert_model_name: str = "medbert-base-wwm-chinese",
        model=None,
        device=None
    ):
        """
        Args:
            split: "train", "val", or "test"
            dataset_name: "cidh" or "pdch"
            data_dir: Directory containing data files
            sum_labels: If True, return total score; if False, return subscales
            bert_model_name: Name/path of BERT model
            model: Pre-loaded BERT model (optional)
            device: Device for BERT encoding (optional)
        """
        self.split = split
        self.dataset_name = dataset_name.lower()
        if self.dataset_name =="cidh":
            self.dataset_name = "cidh"
        self.data_dir = data_dir
        self.sum_labels = sum_labels
        self.bert_model_name = bert_model_name
        self.model = model
        self.device = device
        
        # Load data
        self.data = self.load_data()
        
        # Pre-compute BERT encodings (with caching)
        self.encodings = []
        self.attention_masks = []
        self._encode_all()
    
    def load_data(self) -> List[Dict]:
        """Load data from JSON files."""
        if self.dataset_name == "cidh":
            # CIDH dataset (summarized format)
            file_path = os.path.join(self.data_dir, "cidh", f"eval_summary_{self.split}.json")
        elif self.dataset_name == "pdch":
            # PDCH uses pdch_original_train.json format, directory is lowercase 'pdch'
            file_path = os.path.join(self.data_dir, "pdch", f"pdch_summary_{self.split}.json")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def split_transcript_to_utterances(self, transcript: str) -> List[str]:
        """
        Split transcript into utterances.
        Handles both dialogue format and summarized format.
        """
        if not transcript:
            return [""]
        
        # Check if it's summarized format (each line is a subscale description)
        lines = transcript.strip().split('\n')
        if len(lines) > 5 and any(':' in line or '：' in line for line in lines[:5]):
            # Likely summarized format - split by newlines
            utterances = [line.strip() for line in lines if line.strip()]
            return utterances if utterances else [""]
        
        # Dialogue format - use hierarchical splitting
        # First try splitting by newlines
        utterances = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try splitting by speaker tags
            if re.search(r'^(Doctor|Patient)[:：]', line, re.IGNORECASE):
                utterances.append(line)
            else:
                # Split by sentences
                sentences = re.split(r'[。！？.!?]', line)
                for sent in sentences:
                    sent = sent.strip()
                    if sent and len(sent) > 5:  # Filter very short sentences
                        utterances.append(sent)
        
        # If still too few utterances, split by length
        if len(utterances) < 3:
            all_text = ' '.join(utterances) if utterances else transcript
            chunk_size = max(100, len(all_text) // 5)
            utterances = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]
        
        return utterances if utterances else [transcript]
    
    def _get_embedding_cache_path(self) -> str:
        """Generate cache file path for embeddings."""
        # Create cache directory
        cache_dir = os.path.join(self.data_dir, "embedding_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate unique identifier based on dataset config
        cache_key = f"hamd13_{self.dataset_name}_{self.split}_{self.bert_model_name}_{self.sum_labels}"
        # Use hash to avoid long filenames
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        cache_file = os.path.join(cache_dir, f"embeddings_{cache_hash}.pkl")
        
        return cache_file
    
    def _load_embeddings_from_cache(self, cache_path: str) -> bool:
        """Load embeddings from cache file. Returns True if successful."""
        if not os.path.exists(cache_path):
            return False
        
        try:
            print(f"Loading embeddings from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.encodings = cached_data['encodings']
                self.attention_masks = cached_data['attention_masks']
            
            # Verify length matches
            if len(self.encodings) != len(self.data):
                print(f"Warning: Cached embeddings count ({len(self.encodings)}) doesn't match data count ({len(self.data)}). Regenerating...")
                return False
            
            print(f"Successfully loaded {len(self.encodings)} embeddings from cache.")
            return True
        except Exception as e:
            print(f"Error loading cache: {e}. Regenerating embeddings...")
            return False
    
    def _save_embeddings_to_cache(self, cache_path: str):
        """Save embeddings to cache file."""
        try:
            print(f"Saving embeddings to cache: {cache_path}")
            cached_data = {
                'encodings': self.encodings,
                'attention_masks': self.attention_masks
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"Successfully saved {len(self.encodings)} embeddings to cache.")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _encode_all(self, encode_batch_size: int = 64):
        """Pre-compute BERT encodings for all samples using batch processing."""
        if self.dataset_name == "pdch":
            encode_batch_size = 8
        
        # Try to load from cache first (works even if model is None)
        cache_path = self._get_embedding_cache_path()
        if self._load_embeddings_from_cache(cache_path):
            return
        
        # If cache doesn't exist, need model to generate embeddings
        if self.model is None:
            raise ValueError(
                f"Embedding cache not found at {cache_path} and no BERT model provided. "
                f"Please either provide a BERT model or ensure the cache exists."
            )
        
        # Cache miss or invalid - generate embeddings
        from transformers import AutoTokenizer
        from tqdm import tqdm
        
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        
        print(f"Encoding {len(self.data)} samples with BERT (encode_batch_size={encode_batch_size})...")
        
        # Process samples in batches
        for batch_start in tqdm(range(0, len(self.data), encode_batch_size), desc="Processing batches"):
            batch_end = min(batch_start + encode_batch_size, len(self.data))
            batch_items = self.data[batch_start:batch_end]
            
            # Collect all utterances from this batch
            all_utterances = []
            
            for item in batch_items:
                # Get transcript and split into utterances
                transcript = item.get('transcript', '') or item.get('text', '')
                utterances = self.split_transcript_to_utterances(transcript)
                # Filter empty and too short utterances
                utterances = [u.strip() for u in utterances if u and u.strip() and len(u.strip()) >= 5]
                if not utterances:
                    utterances = ["[EMPTY]"]
                all_utterances.append(utterances)
            
            # Flatten all utterances for batch encoding
            flat_utterances = [utt for utt_list in all_utterances for utt in utt_list]
            
            if not flat_utterances:
                # Empty batch - create zero encodings
                for _ in range(len(batch_items)):
                    self.encodings.append(torch.zeros(1, 768))
                    self.attention_masks.append(torch.ones(1, dtype=torch.bool))
                continue
            
            # Batch encode all utterances
            inputs = tokenizer(
                flat_utterances,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    embeddings = outputs.last_hidden_state.mean(dim=1)
            
            embeddings = embeddings.cpu()
            
            # Group embeddings back by sample
            current_idx = 0
            for utt_list in all_utterances:
                num_utts = len(utt_list)
                sample_embeddings = embeddings[current_idx:current_idx + num_utts]
                current_idx += num_utts
                
                seq_len = sample_embeddings.shape[0]
                self.encodings.append(sample_embeddings)
                # All positions are valid (True)
                mask = torch.ones(seq_len, dtype=torch.bool)
                self.attention_masks.append(mask)
        
        print(f"Encoding complete! {len(self.encodings)} samples processed.")
        
        # Save to cache after encoding
        self._save_embeddings_to_cache(cache_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get sample ID
        sample_id = item.get('id') or item.get('sample_id') or f"{self.split}_{idx}"
        
        # Get labels
        if 'scores' in item:
            scores = item['scores']
        elif 'subscales' in item:
            scores = item['subscales']
        else:
            raise ValueError(f"No scores found in item {idx}")
        
        # Ensure scores is a list of 13 values
        if len(scores) != 13:
            scores = scores[:13] + [0] * (13 - len(scores))
        
        labels = torch.tensor(scores, dtype=torch.float32)
        
        # Get encoding
        if self.encodings and idx < len(self.encodings):
            encoding = self.encodings[idx]
            attention_mask = self.attention_masks[idx]
        else:
            # Fallback: return zero encoding
            encoding = torch.zeros(1, 768)
            attention_mask = torch.ones(1, dtype=torch.bool)
        
        # Get raw transcript for debugging
        transcript = item.get('transcript', '') or item.get('text', '')
        utterances = self.split_transcript_to_utterances(transcript)
        raw_utterances = ' | '.join(utterances[:5]) if utterances else '[EMPTY]'
        
        return sample_id, encoding, labels, attention_mask, raw_utterances


def collate_fn(batch):
    """Collate function for DataLoader."""
    sample_ids, encodings, labels, attention_masks, raw_utterances = zip(*batch)
    
    # Pad encodings to same length
    max_len = max(e.shape[0] for e in encodings)
    batch_size = len(encodings)
    embedding_dim = encodings[0].shape[1]
    
    padded_encodings = torch.zeros(batch_size, max_len, embedding_dim)
    padded_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, (enc, orig_mask) in enumerate(zip(encodings, attention_masks)):
        seq_len = enc.shape[0]  # Original sequence length (before padding)
        
        # Copy embeddings
        padded_encodings[i, :seq_len] = enc
        
        # Create attention mask: True for valid positions, False for padding
        padded_masks[i, :seq_len] = True  # Valid positions
        # Positions beyond seq_len remain False (padding)
    
    # Stack labels
    labels = torch.stack(labels)
    
    return list(sample_ids), padded_encodings, labels, padded_masks, list(raw_utterances)


def get_hamd13_dataloader(
    split: str = "train",
    dataset_name: str = "cidh",
    data_dir: str = "../data",
    sum_labels: bool = False,
    bert_model_name: str = "medbert-base-wwm-chinese",
    batch_size: int = 10,
    shuffle: bool = True,
    model=None,
    device=None,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Get DataLoader for HAMD-13 dataset.
    
    Args:
        split: "train", "val", or "test"
        dataset_name: "cidh" or "pdch"
        data_dir: Directory containing data files
        sum_labels: If True, return total score; if False, return subscales
        bert_model_name: Name/path of BERT model
        batch_size: Batch size
        shuffle: Whether to shuffle data
        model: Pre-loaded BERT model
        device: Device for BERT encoding
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
    
    Returns:
        DataLoader for HAMD-13 dataset
    """
    #data_dir = "../data"  # Use relative path
    dataset = HAMD13Dataset(
        split=split,
        dataset_name=dataset_name,
        data_dir=data_dir,
        sum_labels=sum_labels,
        bert_model_name=bert_model_name,
        model=model,
        device=device
    )
    
    # Create generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(42)
    
    def worker_init_fn(worker_id):
        """Initialize worker with deterministic seed."""
        import numpy as np
        np.random.seed(42 + worker_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g if shuffle else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    
    return dataloader

