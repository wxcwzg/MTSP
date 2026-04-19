"""
E-DAIC dataset loading and preprocessing for PHQ-8 depression assessment.
E-DAIC (Extended Distress Analysis Interview Corpus) is an English dataset.
Adapted from SequenceTextualEDAICDataset with BERT encoding support.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import re
import hashlib
import pickle


class EDAICDataset(Dataset):
    """Dataset for E-DAIC PHQ-8 depression assessment with BERT encoding."""
    
    def __init__(
        self,
        split: str = "train",
        data_dir: str = "../data",
        label_file: str = "edaic_labels.csv",
        sum_labels: bool = False,
        bert_model_name: str = "mental/mental-bert-base-uncased",
        model=None,
        device=None,
        use_sliding_window: bool = False,
        window_size: int = 30,
        window_stride: int = 15
    ):
        """
        Args:
            split: "train", "val", or "test"
            data_dir: Directory containing data files
            label_file: CSV file with labels
            sum_labels: If True, return total score; if False, return subscales
            bert_model_name: Name/path of BERT model
            model: Pre-loaded BERT model (optional)
            device: Device for BERT encoding (optional)
            use_sliding_window: If True, create sliding windows from sequences
            window_size: Size of each sliding window
            window_stride: Step size between windows
        """
        self.split = split
        self.data_dir = data_dir
        self.label_file = label_file
        self.sum_labels = sum_labels
        self.bert_model_name = bert_model_name
        self.model = model
        self.device = device
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.window_stride = window_stride
        
        # EDAIC PHQ-8 subscale columns
        self.relevant_columns = [
            "PHQ8_1_NoInterest", "PHQ8_2_Depressed", "PHQ8_3_Sleep", "PHQ8_4_Tired",
            "PHQ8_5_Appetite", "PHQ8_6_Failure", "PHQ8_7_Concentration", "PHQ8_8_Psychomotor"
        ]
        
        # Load data
        self.data = self.load_data()
        
        # Pre-compute BERT encodings (with caching)
        self.encodings = []
        self.attention_masks = []
        self._encode_all()
    
    def load_textual_features(self, participant: str) -> List[str]:
        """
        Loads the textual transcripts of a participant from their CSV file.
        Returns a list of cleaned utterances (strings) or an empty list if file is unavailable.
        """
        # Try multiple possible file names for EDAIC
        possible_patterns = [
            #f"{participant}_Whisper_Transcript.csv",
            f"{participant}_Transcript.csv"
            #f"{participant}_transcript.csv"
        ]
        
        transcript_path = None
        for pattern in possible_patterns:
            candidate_path = os.path.join(self.data_dir, f"{participant}_P", pattern)
            if os.path.exists(candidate_path):
                transcript_path = candidate_path
                break
        
        if transcript_path is None:
            print(f"WARNING: No transcript file found for participant {participant}")
            return []
        
        try:
            # Read the CSV file (EDAIC uses comma separator)
            df = pd.read_csv(transcript_path, header=0, sep=',')
            df.columns = df.columns.str.strip()
            
            # Check if the Text column exists
            if "Text" not in df.columns:
                # Try alternative column names
                if "value" in df.columns:
                    text_column = "value"
                elif "text" in df.columns:
                    text_column = "text"
                else:
                    print(f"ERROR: Missing 'Text' column in transcript file for participant {participant}")
                    return []
            else:
                text_column = "Text"
            
            # Only filter by confidence threshold if not a Whisper transcript
            if "Whisper" not in transcript_path and "Confidence" in df.columns:
                df = df[df["Confidence"] >= 0.9]
            
            # Extract raw utterances
            raw_utterances = df[text_column].tolist()
            
            # Apply basic cleaning for EDAIC
            cleaned_utterances = []
            removed_count = 0
            original_count = len(raw_utterances)
            
            for utterance in raw_utterances:
                if not utterance or not isinstance(utterance, str):
                    removed_count += 1
                    continue
                    
                # Basic cleaning for EDAIC - just remove extra whitespace and very short utterances
                cleaned = ' '.join(str(utterance).split()).strip()
                
                if len(cleaned) >= 3:  # Keep utterances with at least 3 characters
                    cleaned_utterances.append(cleaned)
                else:
                    removed_count += 1
            
            if original_count > 0:
                print(f"Participant {participant}: {original_count} original -> {len(cleaned_utterances)} cleaned utterances ({removed_count} removed)")
            
            if len(cleaned_utterances) == 0:
                print(f"WARNING: No valid utterances found for participant {participant} after cleaning")
                return []
            
            return cleaned_utterances

        except Exception as e:
            # Catch any exception during file reading or processing
            print(f"ERROR processing transcript for participant {participant}: {str(e)}")
            return []
    
    def create_sliding_windows(self, utterances: List[str], scores: List[int], total_score: int) -> List[Dict]:
        """
        Create sliding windows from utterance sequences.
        
        Args:
            utterances: List of utterance strings
            scores: List of 8 subscale scores
            total_score: Total PHQ-8 score
        
        Returns:
            List of data dictionaries, one per window
        """
        if len(utterances) <= self.window_size:
            # If sequence is shorter than window, return as-is
            return [{
                'utterances': utterances,
                'scores': scores,
                'total_score': total_score
            }]
        
        windows = []
        for start_idx in range(0, len(utterances) - self.window_size + 1, self.window_stride):
            end_idx = start_idx + self.window_size
            window_utterances = utterances[start_idx:end_idx]
            windows.append({
                'utterances': window_utterances,
                'scores': scores,
                'total_score': total_score
            })
        
        return windows
    
    def load_data(self) -> List[Dict]:
        """Load data from CSV and transcript files."""
        # Load labels
        label_path = os.path.join(self.data_dir, self.label_file)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        labels_df = pd.read_csv(label_path, header=0)
        labels_df.columns = labels_df.columns.str.strip()
        
        # Filter by split
        if 'split' in labels_df.columns:
            labels_df = labels_df[labels_df['split'] == self.split]
        elif 'set' in labels_df.columns:
            labels_df = labels_df[labels_df['set'] == self.split]
        
        # If "NISQA" is in the path, filter by mos_pred > 2.5
        print(f"Labels DataFrame shape ({self.split}): {labels_df.shape}")
        if "NISQA" in label_path and "mos_pred" in labels_df.columns and self.split == "train":
            # Filter out low-quality samples
            labels_df = labels_df[labels_df["mos_pred"] > 2.5]
            print(f"Filtered Labels DataFrame shape ({self.split}): {labels_df.shape}")
        
        data = []
        total_windows = 0
        
        for _, row in labels_df.iterrows():
            # Get participant ID
            participant_id = str(row.get('Participant'))
            if not participant_id:
                continue
            
            # Load transcript utterances
            utterances = self.load_textual_features(participant_id)
            if not utterances or len(utterances) == 0:
                continue
            
            # Get PHQ-8 subscale scores
            scores = []
            for col in self.relevant_columns:
                score = row.get(col, 0)
                if pd.isna(score):
                    score = 0
                scores.append(float(score))
            
            # Get total score
            if self.sum_labels:
                total_score = sum(scores)
            else:
                total_score = row.get('PHQ8_Total', row.get('phq8_total', sum(scores)))
                if pd.isna(total_score):
                    total_score = sum(scores)
                total_score = float(total_score)
            
            # Apply sliding window if enabled
            if self.use_sliding_window:
                windows = self.create_sliding_windows(utterances, scores, total_score)
                for window in windows:
                    data.append({
                        'participant_id': participant_id,
                        'utterances': window['utterances'],
                        'scores': window['scores'],
                        'total_score': window['total_score']
                    })
                    total_windows += 1
            else:
                data.append({
                    'participant_id': participant_id,
                    'utterances': utterances,
                    'scores': scores,
                    'total_score': total_score
                })
                total_windows += 1
        
        if self.use_sliding_window:
            print(f"Created {total_windows} windows from {len(labels_df)} participants "
                  f"(window_size={self.window_size}, stride={self.window_stride})")
        else:
            print(f"Loaded {total_windows} sequences")
        
        return data
    
    def split_transcript_to_utterances(self, transcript: str) -> List[str]:
        """
        Split transcript into utterances (fallback method).
        Note: This is mainly for backward compatibility. 
        The main loading logic now uses load_textual_features which returns cleaned utterances directly.
        """
        if not transcript:
            return [""]
        
        # Split by common sentence delimiters
        utterances = []
        
        # First try splitting by newlines (if transcript contains multiple lines)
        lines = transcript.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split by sentences
            sentences = re.split(r'[.?!]\s+', line)
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
        cache_key = f"edaic_{self.split}_{self.bert_model_name}_{self.sum_labels}"
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
    
    def _encode_all(self, batch_size: int = 64):
        """Pre-compute BERT encodings for all samples using batch processing."""
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
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        
        print(f"Encoding {len(self.data)} samples with BERT (batch_size={batch_size})...")
        
        # Process samples in batches
        from tqdm import tqdm
        for batch_start in tqdm(range(0, len(self.data), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(self.data))
            batch_items = self.data[batch_start:batch_end]
            
            # Collect all utterances from this batch
            all_utterances = []
            
            for idx, item in enumerate(batch_items):
                # Get utterances directly from data (already cleaned and split)
                utterances = item.get('utterances', [])
                # Filter empty utterances
                utterances = [utt for utt in utterances if utt.strip()]
                if not utterances:
                    # If all utterances were filtered out, use a placeholder
                    # This ensures we have at least one embedding per sample
                    utterances = [""]  # At least one empty utterance
                all_utterances.append(utterances)
            
            # Verify utterance counts match expected structure
            expected_total = sum(len(utt_list) for utt_list in all_utterances)
            
            # Flatten all utterances for batch encoding
            flat_utterances = [utt for utt_list in all_utterances for utt in utt_list]
            
            if not flat_utterances:
                # Empty batch - create zero encodings with valid mask
                for _ in range(len(batch_items)):
                    self.encodings.append(torch.zeros(1, 768))
                    self.attention_masks.append(torch.ones(1, dtype=torch.bool))  # One valid position
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
                # Use [CLS] token or mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    # Fallback: use mean of last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1)
            
            embeddings = embeddings.cpu()  # [num_total_utterances, embed_dim]
            
            # Verify embedding count matches utterance count
            if len(embeddings) != expected_total:
                raise ValueError(
                    f"Embedding count mismatch: generated {len(embeddings)} embeddings, "
                    f"but expected {expected_total} (from {len(all_utterances)} samples)."
                )
            
            # Group embeddings back by sample
            batch_embeddings = []
            current_idx = 0
            for utt_list in all_utterances:
                num_utts = len(utt_list)
                # Verify we have enough embeddings
                if current_idx + num_utts > len(embeddings):
                    raise ValueError(
                        f"Index mismatch: trying to access embeddings[{current_idx}:{current_idx + num_utts}], "
                        f"but only {len(embeddings)} embeddings available. "
                        f"Expected {sum(len(utt_list) for utt_list in all_utterances)} total utterances."
                    )
                sample_embeddings = embeddings[current_idx:current_idx + num_utts]
                batch_embeddings.append(sample_embeddings)
                current_idx += num_utts
            
            # Verify all embeddings were used
            if current_idx != len(embeddings):
                raise ValueError(
                    f"Embedding count mismatch: used {current_idx} embeddings, "
                    f"but {len(embeddings)} embeddings were generated. "
                    f"Expected {sum(len(utt_list) for utt_list in all_utterances)} total utterances."
                )
            
            # Store original embeddings without padding
            # Padding will be done in collate_fn to ensure consistency across batches
            for emb in batch_embeddings:
                seq_len = emb.shape[0]
                # Store original embedding (no padding here)
                self.encodings.append(emb)
                # Create attention mask based on actual sequence length
                # True for valid positions, False for padding (will be padded in collate_fn)
                mask = torch.ones(seq_len, dtype=torch.bool)
                self.attention_masks.append(mask)
        
        # Save to cache after encoding
        self._save_embeddings_to_cache(cache_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get sample ID
        sample_id = item.get('participant_id', f"edaic_{self.split}_{idx}")
        
        # Get labels
        if self.sum_labels:
            # Return total score
            labels = torch.tensor([item['total_score']], dtype=torch.float32)
        else:
            # Return 8 subscale scores
            scores = item['scores']
            if len(scores) != 8:
                scores = scores[:8] + [0] * (8 - len(scores))
            labels = torch.tensor(scores, dtype=torch.float32)
        
        # Get encoding
        if self.encodings and idx < len(self.encodings):
            encoding = self.encodings[idx]  # Original embedding, no padding
            attention_mask = self.attention_masks[idx]  # Mask with actual sequence length
        else:
            # Fallback: return zero encoding
            encoding = torch.zeros(1, 768)
            attention_mask = torch.ones(1, dtype=torch.bool)  # One valid position
        
        # Get raw utterances for debugging
        utterances = item.get('utterances', [])
        raw_utterances = ' | '.join(utterances[:5])  # First 5 utterances for logging
        
        return sample_id, encoding, labels, attention_mask, raw_utterances


def collate_fn_edaic(batch):
    """Collate function for E-DAIC DataLoader."""
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
        # orig_mask should be all True (valid positions), length = seq_len
        padded_masks[i, :seq_len] = True  # Valid positions
        if seq_len < max_len:
            padded_masks[i, seq_len:] = False  # Padding positions
    
    # Stack labels
    labels = torch.stack(labels)
    
    return list(sample_ids), padded_encodings, labels, padded_masks, list(raw_utterances)


def get_edaic_dataloader(
    split: str = "train",
    data_dir: str = "../data",
    label_file: str = "edaic_labels.csv",
    sum_labels: bool = False,
    bert_model_name: str = "mental/mental-bert-base-uncased",
    batch_size: int = 10,
    shuffle: bool = True,
    model=None,
    device=None,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_sliding_window: bool = False,
    window_size: int = 30,
    window_stride: int = 15
) -> DataLoader:
    """
    Get DataLoader for E-DAIC dataset.
    
    Args:
        split: "train", "val", or "test"
        data_dir: Directory containing data files
        label_file: CSV file with labels
        sum_labels: If True, return total score; if False, return subscales
        bert_model_name: Name/path of BERT model
        batch_size: Batch size
        shuffle: Whether to shuffle data
        model: Pre-loaded BERT model
        device: Device for BERT encoding
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        use_sliding_window: If True, create sliding windows from sequences
        window_size: Size of each sliding window
        window_stride: Step size between windows
    
    Returns:
        DataLoader for E-DAIC dataset
    """
    dataset = EDAICDataset(
        split=split,
        data_dir=data_dir,
        label_file=label_file,
        sum_labels=sum_labels,
        bert_model_name=bert_model_name,
        model=model,
        device=device,
        use_sliding_window=use_sliding_window,
        window_size=window_size,
        window_stride=window_stride
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_edaic,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader

