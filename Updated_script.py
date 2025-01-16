import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import torch
from torch.utils.data import Dataset
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import evaluate
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nli_training.log'),
        logging.StreamHandler()
    ]
)

# Standard NLI labels mapping
LABEL_MAP = {
    "entailment": 1,
    "contradiction": 0
}

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

class DataProcessor:
    def __init__(self, model_id="answerdotai/ModernBERT-base", batch_size=32):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def process_batch_zeroshot(self, description: str, candidate_definitions: List[str]) -> List[int]:
        """Process a batch using zero-shot classification"""
        try:
            result = self.classifier(
                description,
                candidate_definitions,
                hypothesis_template="This text means: {}",
                multi_label=False
            )
            scores = np.array(result['scores'])
            return np.argsort(scores)[::-1]
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
                'hard_negative_indices': ranked_indices[:self.batch_size]
            })
        return results

    def create_nli_dataset(self, df: pd.DataFrame, k_negatives: int = 3) -> List[Dict]:
        """Create NLI pairs with labels following standard format"""
        logging.info("Starting NLI dataset creation...")
        
        # Get all unique definitions
        all_definitions = df['concept_definition'].unique().tolist()
        logging.info(f"Total unique definitions: {len(all_definitions)}")
        
        # Split data into chunks for parallel processing
        chunk_size = max(1, len(df) // (os.cpu_count() * 2))
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
        
        # Create NLI pairs
        nli_pairs = []
        for result in tqdm(results, desc="Creating NLI pairs"):
            # Positive pair (entailment)
            nli_pairs.append({
                'premise': result['description'],
                'hypothesis': result['correct_def'],
                'label': 'entailment'
            })
            
            # Hard negative pairs (contradictions)
            for neg_idx in result['hard_negative_indices'][:k_negatives]:
                neg_def = all_definitions[neg_idx]
                nli_pairs.append({
                    'premise': result['description'],
                    'hypothesis': neg_def,
                    'label': 'contradiction'
                })
        
        logging.info(f"Created {len(nli_pairs)} NLI pairs")
        return nli_pairs

class NLIDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Convert string labels to integers using standard mapping
        self.labels = torch.tensor([LABEL_MAP[label] for label in data['label']], dtype=torch.long)
        
        # Create encodings following standard NLI format
        self.encodings = tokenizer(
            data['premise'].tolist(),
            data['hypothesis'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            return_token_type_ids=True,
            add_special_tokens=True  # [CLS] premise [SEP] hypothesis [SEP]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels': self.labels[idx]
        }

def train_nli_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    model_id: str = "answerdotai/ModernBERT-base",
    output_dir: str = "nli-model",
    num_epochs: int = 3
):
    """Train NLI model following standard practices"""
    logging.info("Starting NLI model training...")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(LABEL_MAP),  # Standard binary NLI
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = NLIDataset(train_data, tokenizer)
    eval_dataset = NLIDataset(val_data, tokenizer)
    
    # Standard NLI training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,  # Standard fine-tuning LR
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Standard NLI metric
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=2
    )
    
    # Standard NLI metrics
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Standard NLI metrics
        accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
        f1 = metric_f1.compute(predictions=predictions, references=labels, average='weighted')
        
        return {
            'accuracy': accuracy['accuracy'],
            'f1': f1['f1']
        }
    
    # Initialize trainer with standard settings
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Train and save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logging.info(f"Model saved to {output_dir}")
    return trainer, tokenizer

class NLIPredictor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Reverse label mapping for predictions
        self.id2label = {v: k for k, v in LABEL_MAP.items()}

    def predict(self, premise: str, hypothesis: str) -> Dict:
        """Make NLI prediction following standard format"""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs).item()
            
        return {
            'label': self.id2label[pred_idx],
            'confidence': float(probs[0][pred_idx])
        }

    def predict_batch(self, data: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """Batch prediction for multiple examples"""
        results = []
        
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data.iloc[i:i+batch_size]
            inputs = self.tokenizer(
                batch['premise'].tolist(),
                batch['hypothesis'].tolist(),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred_indices = torch.argmax(probs, dim=1)
                
            batch_results = [{
                'label': self.id2label[idx.item()],
                'confidence': float(probs[i][idx.item()])
            } for i, idx in enumerate(pred_indices)]
            
            results.extend(batch_results)
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Load data
    attributes_df = pd.read_csv("attributes.csv")
    definitions_df = pd.read_csv("definitions.csv")
    
    # Prepare data
    train_df, val_df, test_df = prepare_and_analyze_data(attributes_df, definitions_df)
    
    # Create NLI pairs
    processor = DataProcessor()
    train_pairs = processor.create_nli_dataset(train_df)
    val_pairs = processor.create_nli_dataset(val_df)
    test_pairs = processor.create_nli_dataset(test_df)
    
    # Convert to DataFrames
    train_nli = pd.DataFrame(train_pairs)
    val_nli = pd.DataFrame(val_pairs)
    test_nli = pd.DataFrame(test_pairs)
    
    logging.info("Data distribution:")
    for split_name, split_data in [("Train", train_nli), ("Val", val_nli), ("Test", test_nli)]:
        total = len(split_data)
        entailment = (split_data['label'] == 'entailment').sum()
        contradiction = (split_data['label'] == 'contradiction').sum()
        logging.info(f"{split_name} split distribution:")
        logging.info(f"  Total pairs: {total}")
        logging.info(f"  Entailment: {entailment} ({entailment/total*100:.2f}%)")
        logging.info(f"  Contradiction: {contradiction} ({contradiction/total*100:.2f}%)")
    
    # Train model
    trainer, tokenizer = train_nli_model(
        train_data=train_nli,
        val_data=val_nli,
        model_id="answerdotai/ModernBERT-base",
        output_dir="nli-model"
    )
    
    # Evaluate on test set
    predictor = NLIPredictor("nli-model")
    test_predictions = predictor.predict_batch(test_nli)
    
    # Calculate and log test metrics
    true_labels = [LABEL_MAP[label] for label in test_nli['label']]
    pred_labels = [LABEL_MAP[label] for label in test_predictions['label']]
    
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    test_accuracy = metric_accuracy.compute(predictions=pred_labels, references=true_labels)
    test_f1 = metric_f1.compute(predictions=pred_labels, references=true_labels, average='weighted')
    
    logging.info("Test Set Evaluation:")
    logging.info(f"  Accuracy: {test_accuracy['accuracy']:.4f}")
    logging.info(f"  F1 Score: {test_f1['f1']:.4f}")
    
    # Example predictions
    example_pairs = [
        {
            'premise': test_nli['premise'].iloc[0],
            'hypothesis': test_nli['hypothesis'].iloc[0]
        },
        {
            'premise': test_nli['premise'].iloc[1],
            'hypothesis': test_nli['hypothesis'].iloc[1]
        }
    ]
    
    logging.info("\nExample Predictions:")
    for i, pair in enumerate(example_pairs):
        result = predictor.predict(pair['premise'], pair['hypothesis'])
        logging.info(f"\nExample {i+1}:")
        logging.info(f"Premise: {pair['premise']}")
        logging.info(f"Hypothesis: {pair['hypothesis']}")
        logging.info(f"Predicted: {result['label']} (confidence: {result['confidence']:.4f})")
        logging.info(f"True label: {test_nli['label'].iloc[i]}")
