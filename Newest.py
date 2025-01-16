import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    AutoConfig,
    EarlyStoppingCallback,
    pipeline
)
import torch
from torch.utils.data import Dataset
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import random
import json

# Set up logging
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

def set_seed(seed_val=42):
    """Set random seeds for reproducibility"""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

def prepare_and_analyze_data(attributes_df, definitions_df, test_size=0.1, val_size=0.1):
    """Prepare and analyze data distribution before NLI processing"""
    logging.info("Starting data preparation and analysis...")
    
    # Validate input data
    required_columns = ['domain', 'concept', 'description', 'concept_definition']
    for df, name in [(attributes_df, 'attributes'), (definitions_df, 'definitions')]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {name}_df: {missing_cols}")
    
    # Check for missing values
    for df, name in [(attributes_df, 'attributes'), (definitions_df, 'definitions')]:
        missing_values = df.isnull().sum()
        if missing_values.any():
            logging.warning(f"Found missing values in {name}_df:\n{missing_values[missing_values > 0]}")
            df.dropna(inplace=True)
    
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
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def get_sequence_length(row):
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
    
    # Save length statistics
    length_stats = {}
    for name, lengths in [('Train', train_lengths), ('Val', val_lengths), ('Test', test_lengths)]:
        length_stats[name] = {
            'mean': float(np.mean(lengths)),
            'median': float(np.median(lengths)),
            'p95': float(np.percentile(lengths, 95)),
            'max': float(max(lengths))
        }
        logging.info(f"{name} length stats:")
        for stat, value in length_stats[name].items():
            logging.info(f"  {stat}: {value:.2f}")
    
    # Save statistics to file
    with open('data_statistics.json', 'w') as f:
        json.dump(length_stats, f, indent=2)
    
    return train_df, val_df, test_df

class DataProcessor:
    def __init__(self, model_id="bert-base-uncased", batch_size=32):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_id,
            device=-1  # CPU-based
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
        """Initialize NLI dataset"""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.premises = data['premise'].tolist()
        self.hypotheses = data['hypothesis'].tolist()
        self.labels = [LABEL_MAP[label] for label in data['label']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'premise': self.premises[idx],
            'hypothesis': self.hypotheses[idx],
            'label': self.labels[idx]
        }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Standard metrics
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average='weighted')
    
    # Additional metrics
    report = classification_report(labels, predictions, output_dict=True)
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    
    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1'],
        'specificity': specificity,
        'npv': npv,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }

def train_nli_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    model_id: str = "bert-base-uncased",
    output_dir: str = "nli-model",
    num_epochs: int = 3
):
    """Train NLI model following standard practices"""
    logging.info("Starting NLI model training...")
    
    # Initialize config with dropout
    config = AutoConfig.from_pretrained(model_id)
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    config.num_labels = len(LABEL_MAP)
    config.problem_type = "single_label_classification"
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        config=config
    )
    
    # Create datasets
    train_dataset = NLIDataset(train_data, tokenizer)
    eval_dataset = NLIDataset(val_data, tokenizer)
    
    # Calculate steps
    num_update_steps_per_epoch = len(train_dataset) // 16
    max_steps = num_epochs * num_update_steps_per_epoch
    
    # Create data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Calculate warmup steps (10% of total steps)
    num_train_examples = len(train_dataset)
    num_train_steps = (num_train_examples * num_epochs) // 16
    num_warmup_steps = num_train_steps // 10

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=num_warmup_steps,  # Use explicit warmup steps
        weight_decay=0.01,
        learning_rate=2e-5,
        lr_scheduler_type="linear",  # Explicit scheduler type
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        group_by_length=True,
        disable_tqdm=False,
        report_to="none"  # Disable wandb/tensorboard
    )
    
    # Early stopping callback with more conservative settings
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )

    # Custom callback to save best model metrics
    class MetricsSaverCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            if state.best_metric is None or metrics.get("eval_f1") > state.best_metric:
                state.best_metric = metrics.get("eval_f1")
                # Save metrics
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                # Save epoch number
                with open(os.path.join(args.output_dir, "best_epoch.txt"), "w") as f:
                    f.write(f"Best epoch: {state.epoch}")

    metrics_saver = MetricsSaverCallback()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping, metrics_saver],
        tokenizer=tokenizer
    )
    
    # Train and save
    trainer.train()
    
    # Save model, tokenizer and config
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    
    logging.info(f"Model saved to {output_dir}")
    return trainer, tokenizer

class NLIPredictor:
    def __init__(self, model_path: str):
        """Initialize NLI predictor"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cpu")
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

    def predict_csv(self, csv_path: str, premise_col: str = 'premise', hypothesis_col: str = 'hypothesis', 
                   batch_size: int = 32) -> pd.DataFrame:
        """Predict from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            required_cols = [premise_col, hypothesis_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in CSV: {missing_cols}")
            
            predictions = self.predict_batch(
                df.rename(columns={
                    premise_col: 'premise',
                    hypothesis_col: 'hypothesis'
                }),
                batch_size=batch_size
            )
            
            # Add predictions to original DataFrame
            df['predicted_label'] = predictions['label']
            df['confidence'] = predictions['confidence']
            
            return df
            
        except Exception as e:
            logging.error(f"Error predicting from CSV: {e}")
            raise

def evaluate_per_label_metrics(y_true, y_pred, labels=None):
    """Calculate precision, recall, and F1 for each label"""
    if labels is None:
        labels = {
            1: "entailment",
            0: "contradiction"
        }
    
    # Get detailed classification report
    report = classification_report(
        y_true, 
        y_pred,
        target_names=[labels[0], labels[1]],
        digits=4,
        output_dict=True
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print detailed metrics
    print("\nPer-Label Metrics:")
    print("-" * 50)
    
    for label_id, label_name in labels.items():
        metrics = report[label_name]
        print(f"\n{label_name.upper()}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1-score']:.4f}")
        print(f"Support: {metrics['support']}")
    
    print("\nConfusion Matrix:")
    print("-" * 50)
    print("                 Predicted")
    print("                 Contra.  Entail.")
    print(f"Actual Contra.   {cm[0][0]:<8} {cm[0][1]:<8}")
    print(f"      Entail.   {cm[1][0]:<8} {cm[1][1]:<8}")
    
    return report

if __name__ == "__main__":
    # Set random seeds
    set_seed(42)
    
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
    
    # Log data distribution
    logging.info("Data distribution:")
    for split_name, split_data in [("Train", train_nli), ("Val", val_nli), ("Test", test_nli)]:
        total = len(split_data)
        entailment = (split_data['label'] == 'entailment').sum()
        contradiction = (split_data['label'] == 'contradiction').sum()
        logging.info(f"{split_name} split distribution:")
        logging.info(f"  Total pairs: {total}")
        logging.info(f"  Entailment: {entailment} ({entailment/total*100:.2f}%)")
        logging.info(f"  Contradiction: {contradiction} ({contradiction/total*100:.2f}%)")
    
    # Initialize tokenizer and create data collator
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=512,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Train model
    trainer, tokenizer = train_nli_model(
        train_data=train_nli,
        val_data=val_nli,
        model_id="bert-base-uncased",
        output_dir="nli-model",
        num_epochs=3
    )
    
    # Evaluate on test set
    predictor = NLIPredictor("nli-model")
    test_predictions = predictor.predict_batch(test_nli)
    
    # Calculate and log test metrics
    true_labels = [LABEL_MAP[label] for label in test_nli['label']]
    pred_labels = [LABEL_MAP[label] for label in test_predictions['label']]
    
    # Detailed evaluation
    detailed_metrics = evaluate_per_label_metrics(
        true_labels,
        pred_labels,
        labels={v: k for k, v in LABEL_MAP.items()}
    )
    
    # Save detailed metrics
    with open('detailed_metrics.json', 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
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
