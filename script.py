# data_processor.py
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, pipeline
import torch
from torch.utils.data import Dataset
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nli_training.log'),
        logging.StreamHandler()
    ]
)

def prepare_and_analyze_data(attributes_df, definitions_df, test_size=0.1, val_size=0.1):
    """
    Prepare and analyze data distribution before NLI processing
    """
    logging.info("Starting data preparation and analysis...")
    
    # Merge dataframes
    df = pd.merge(attributes_df, definitions_df, on=['domain', 'concept'])
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split sizes
    val_size_abs = int(len(df) * val_size)
    test_size_abs = int(len(df) * test_size)
    
    # Split data
    test_df = df[:test_size_abs]
    val_df = df[test_size_abs:test_size_abs+val_size_abs]
    train_df = df[test_size_abs+val_size_abs:]
    
    logging.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Initialize tokenizer for length analysis
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    def get_sequence_length(row):
        # Tokenize with NLI format: [CLS] premise [SEP] hypothesis [SEP]
        tokens = tokenizer(
            row['description'],
            row['concept_definition'],
            add_special_tokens=True,
            return_tensors='pt'
        )
        return len(tokens['input_ids'][0])
    
    # Calculate lengths
    logging.info("Calculating sequence lengths...")
    train_lengths = [get_sequence_length(row) for _, row in tqdm(train_df.iterrows(), desc="Train")]
    val_lengths = [get_sequence_length(row) for _, row in tqdm(val_df.iterrows(), desc="Val")]
    test_lengths = [get_sequence_length(row) for _, row in tqdm(test_df.iterrows(), desc="Test")]
    
    # Plot length distribution
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.hist([train_lengths, val_lengths, test_lengths], 
             label=['Train', 'Val', 'Test'], 
             bins=50, alpha=0.7)
    plt.xlabel('Sequence Length (including special tokens)')
    plt.ylabel('Count')
    plt.title('Length Distribution of NLI Sequences')
    plt.legend()
    plt.savefig('length_distribution.png')
    plt.close()
    
    # Log length statistics
    for name, lengths in [('Train', train_lengths), ('Val', val_lengths), ('Test', test_lengths)]:
        logging.info(f"{name} length stats:")
        logging.info(f"  Mean: {np.mean(lengths):.2f}")
        logging.info(f"  Median: {np.median(lengths):.2f}")
        logging.info(f"  95th percentile: {np.percentile(lengths, 95):.2f}")
        logging.info(f"  Max: {max(lengths)}")
    
    return train_df, val_df, test_df

# Update DataProcessor class
class DataProcessor:
    def __init__(self, model_id="answerdotai/ModernBERT-base", batch_size=32):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize zero-shot classifier
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def create_nli_dataset(self, data_df, k_negatives=3):
        """Modified to work with pre-split data"""
        logging.info("Starting NLI dataset creation...")
        
    def process_batch_zeroshot(self, description, candidate_definitions):
        """Process a batch of definitions for a single description using zero-shot"""
        try:
            result = self.classifier(
                description,
                candidate_definitions,
                hypothesis_template="This text means: {}",
                multi_label=False
            )
            # Sort by scores and return indices
            scores = np.array(result['scores'])
            return np.argsort(scores)[::-1]  # Return indices sorted by score
        except Exception as e:
            logging.error(f"Error in zero-shot classification: {e}")
            return []

    def process_chunk(self, chunk_data, all_definitions):
        """Process a chunk of descriptions with zero-shot classification"""
        results = []
        for _, row in chunk_data.iterrows():
            description = row['description']
            correct_def = row['concept_definition']
            
            # Filter out the correct definition from candidates
            candidate_defs = [d for d in all_definitions if d != correct_def]
            
            # Get zero-shot rankings in batches
            ranked_indices = self.process_batch_zeroshot(description, candidate_defs)
            
            results.append({
                'description': description,
                'correct_def': correct_def,
                'hard_negative_indices': ranked_indices[:self.batch_size]  # Take top k as hard negatives
            })
        return results

    def create_nli_dataset(self, attributes_df, definitions_df, k_negatives=3):
        logging.info("Starting NLI dataset creation...")
        
        # Merge dataframes
        df = pd.merge(attributes_df, definitions_df, on=['domain', 'concept'])
        
        # Get all unique definitions
        all_definitions = df['concept_definition'].unique().tolist()
        logging.info(f"Total unique definitions: {len(all_definitions)}")
        
        # Split data into chunks for parallel processing
        chunk_size = max(1, len(df) // (os.cpu_count() * 2))  # Smaller chunks for better parallelization
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        logging.info(f"Processing {len(chunks)} chunks in parallel")
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self.process_chunk, chunk, all_definitions))
            
            # Collect results with progress bar
            results = []
            for future in tqdm(futures, desc="Processing chunks"):
                results.extend(future.result())
        
        nli_data = []
        
        # Create positive and negative pairs
        for result in tqdm(results, desc="Creating NLI pairs"):
            # Add positive pair
            nli_data.append({
                'text': (result['description'], result['correct_def']),  # tuple for paired texts
                'label': 1
            })
            
            # Add hard negative pairs
            for neg_idx in result['hard_negative_indices'][:k_negatives]:
                neg_def = all_definitions[neg_idx]
                nli_data.append({
                    'text': (result['description'], neg_def),  # tuple for paired texts
                    'label': 0
                })
                
        # Convert to DataFrame with expected column names
        df_nli = pd.DataFrame(nli_data)
        df_nli = df_nli.rename(columns={'label': 'labels'})  # match the blog's format
        logging.info(f"Created dataset with {len(df_nli)} pairs")
        
        logging.info(f"Created {len(nli_data)} NLI pairs")
        return pd.DataFrame(nli_data)

# nli_dataset.py
class NLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create proper NLI format with special tokens
        texts_pairs = []
        for _, row in data.iterrows():
            # ModernBERT will automatically add [CLS] at start and [SEP] tokens appropriately
            # when we pass sentence pairs to tokenizer
            texts_pairs.append((row['premise'], row['hypothesis']))
        
        # Tokenize all pairs with proper NLI formatting
        self.encodings = self.tokenizer(
            texts_pairs,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            # Important: This tells tokenizer to handle text pairs
            return_token_type_ids=True,
            is_split_into_words=False,
            add_special_tokens=True  # Ensures [CLS], [SEP] tokens are added correctly
        )
        
        # Each encoding will look like:
        # [CLS] premise [SEP] hypothesis [SEP]
        # with corresponding token_type_ids to differentiate premise and hypothesis
        
        self.labels = torch.tensor(data['labels'].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        return item

# trainer.py
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import evaluate

def train_nli_model(train_data, model_id="answerdotai/ModernBERT-base", output_dir="nli-model"):
    logging.info("Starting model training...")
    
    # Split data
    train_df, eval_df = train_test_split(train_data, test_size=0.1, random_state=42)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    
    # Create datasets
    train_dataset = NLIDataset(train_df, tokenizer)
    eval_dataset = NLIDataset(eval_df, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2
    )
    
    # Metrics
    metric = evaluate.load("f1")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    trainer.train()
    
    # Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logging.info(f"Model saved to {output_dir}")
    return trainer, tokenizer

# inference.py
class NLIPredictor:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_single(self, premise, hypothesis):
        # Proper NLI formatting with special tokens
        inputs = self.tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
            add_special_tokens=True  # Ensures [CLS], [SEP] tokens
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            
        pred_label = int(torch.argmax(predictions))
        return {
            'label': "entailment" if pred_label == 1 else "contradiction",
            'score': float(torch.max(predictions))
        }

    def predict_csv(self, csv_path, premise_col, hypothesis_col, batch_size=32):
        df = pd.read_csv(csv_path)
        results = []
        
        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i:i+batch_size]
            inputs = self.tokenizer(
                batch_df[premise_col].tolist(),
                batch_df[hypothesis_col].tolist(),
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=True,
                add_special_tokens=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                
            batch_results = [{
                'label': "entailment" if int(torch.argmax(pred)) == 1 else "contradiction",
                'score': float(torch.max(pred))
            } for pred in predictions]
            
            results.extend(batch_results)
            
        return pd.DataFrame(results)

# Usage example
if __name__ == "__main__":
    # Load your CSVs
    attributes_df = pd.read_csv("attributes.csv")
    definitions_df = pd.read_csv("definitions.csv")
    
    # First split and analyze data
    train_df, val_df, test_df = prepare_and_analyze_data(attributes_df, definitions_df)
    
    # Create NLI datasets
    processor = DataProcessor()
    train_nli = processor.create_nli_dataset(train_df)
    val_nli = processor.create_nli_dataset(val_df)
    test_nli = processor.create_nli_dataset(test_df)
    
    # Train model
    trainer, tokenizer = train_nli_model(
        train_data=train_nli,
        val_data=val_nli,  # Added validation data
        model_id="answerdotai/ModernBERT-base",
        output_dir="nli-model"
    )
    
    # Example predictions
    predictor = NLIPredictor("nli-model")
    
    # Test set evaluation
    test_results = predictor.predict_csv(
        test_df,
        premise_col='description',
        hypothesis_col='concept_definition'
    )
    
    logging.info(f"Test set results:\n{test_results.describe()}")
