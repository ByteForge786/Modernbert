import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report, precision_recall_fscore_support
import os
from datetime import datetime
import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class ConceptDataset(Dataset):
    def __init__(self, texts, concepts, tokenizer, max_length=128):
        self.texts = texts
        self.concepts = concepts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.concepts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ConceptDataProcessor:
    def __init__(self, attr_csv_path, concept_csv_path):
        """
        Initialize dataset with paths to both CSVs
        """
        self.logger = logging.getLogger(__name__)
        self.load_and_process_data(attr_csv_path, concept_csv_path)

    def load_and_process_data(self, attr_csv_path, concept_csv_path):
        """
        Load and process both CSVs with proper error handling
        """
        try:
            self.logger.info("Loading CSV files...")
            self.attr_df = pd.read_csv(attr_csv_path)
            self.concept_df = pd.read_csv(concept_csv_path)
            
            # Validate required columns
            required_attr_cols = ['attribute_name', 'description', 'concept']
            required_concept_cols = ['concept', 'concept_definition']
            
            if not all(col in self.attr_df.columns for col in required_attr_cols):
                raise ValueError(f"Missing required columns in attributes CSV: {required_attr_cols}")
            if not all(col in self.concept_df.columns for col in required_concept_cols):
                raise ValueError(f"Missing required columns in concepts CSV: {required_concept_cols}")
            
            # Create combined text field
            self.logger.info("Creating combined text from attribute_name and description")
            self.attr_df['combined_text'] = self.attr_df['attribute_name'] + ' - ' + self.attr_df['description']
            
            # Create label encoder for concepts
            self.unique_concepts = sorted(self.concept_df['concept'].unique())
            self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.unique_concepts)}
            self.idx_to_concept = {idx: concept for concept, idx in self.concept_to_idx.items()}
            
            self.num_labels = len(self.unique_concepts)
            self.logger.info(f"Found {self.num_labels} unique concepts")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def create_train_val_test_split(self, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Create stratified split while avoiding data leakage
        """
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("Split proportions must sum to 1")

        try:
            # First split into train and temp
            train_df, temp_df = train_test_split(
                self.attr_df,
                train_size=train_size,
                stratify=self.attr_df['concept'],
                random_state=random_state
            )

            # Then split temp into val and test
            relative_val_size = val_size / (val_size + test_size)
            val_df, test_df = train_test_split(
                temp_df,
                train_size=relative_val_size,
                stratify=temp_df['concept'],
                random_state=random_state
            )

            # Convert concepts to indices
            train_df['label'] = train_df['concept'].map(self.concept_to_idx)
            val_df['label'] = val_df['concept'].map(self.concept_to_idx)
            test_df['label'] = test_df['concept'].map(self.concept_to_idx)

            self.logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df

        except Exception as e:
            self.logger.error(f"Error in split: {str(e)}")
            raise

class ConceptClassifier:
    def __init__(self, model_name="bert-base-uncased", num_labels=None, batch_size=16, num_epochs=3):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_labels = num_labels
        self.best_model_path = None
        self.best_score = 0.0
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )

    def create_datasets(self, train_df, val_df, test_df):
        """
        Create PyTorch datasets for training, validation, and testing
        """
        train_dataset = ConceptDataset(
            train_df['combined_text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer
        )
        
        val_dataset = ConceptDataset(
            val_df['combined_text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer
        )
        
        test_dataset = ConceptDataset(
            test_df['combined_text'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer
        )
        
        return train_dataset, val_dataset, test_dataset

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self, train_dataset, val_dataset):
        """
        Train the model using HuggingFace Trainer
        """
        try:
            # Create checkpoint directory
            checkpoint_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=checkpoint_dir,
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                learning_rate=2e-5,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=100
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics
            )

            self.logger.info("Starting training...")
            trainer.train()

            # Save best model
            best_model_path = os.path.join(checkpoint_dir, "best_model")
            trainer.save_model(best_model_path)
            self.best_model_path = best_model_path

            # Get best metric
            self.best_score = trainer.state.best_metric

            return self.best_model_path, self.best_score

        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise

    def evaluate(self, test_dataset):
        """
        Evaluate model performance with detailed metrics
        """
        trainer = Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics
        )
        
        metrics = trainer.evaluate(test_dataset)
        
        return metrics
        
    def predict(self, texts, batch_size=32):
        """
        Make predictions for new texts
        Args:
            texts (list): List of strings to classify
            batch_size (int): Batch size for prediction
        Returns:
            predictions (list): List of predicted concept labels
            probabilities (list): List of probability distributions over all concepts
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                batch_predictions = torch.argmax(logits, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Convert numeric predictions back to concept labels
        concept_predictions = [self.idx_to_concept[pred] for pred in predictions]
        
        return concept_predictions, probabilities

def main():
    # Set up logging
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize dataset processor
        data_processor = ConceptDataProcessor('attributes.csv', 'concepts.csv')
        
        # Create splits
        train_df, val_df, test_df = data_processor.create_train_val_test_split()
        
        # Initialize and train classifier
        classifier = ConceptClassifier(
            num_labels=data_processor.num_labels,
            batch_size=32,
            num_epochs=5
        )
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = classifier.create_datasets(
            train_df, val_df, test_df
        )
        
        # Train model
        best_model_path, best_score = classifier.train(train_dataset, val_dataset)
        
        logger.info(f"Training completed. Best model saved at: {best_model_path}")
        logger.info(f"Best validation F1 score: {best_score:.4f}")
        
        # Evaluate on test set
        test_metrics = classifier.evaluate(test_dataset)
        logger.info("Test set metrics:")
        logger.info(json.dumps(test_metrics, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
