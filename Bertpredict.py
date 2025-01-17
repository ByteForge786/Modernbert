import argparse
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from sklearn.metrics import classification_report
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ConceptPredictor:
    def __init__(self, model_path, device=None):
        self.logger = logger
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load concept mapping from model config
        self.idx_to_concept = {
            idx: label for idx, label in enumerate(self.model.config.id2label.values())
        }
        
    def predict_single(self, text):
        """Predict concept for a single text input"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred_idx].item()
        
        predicted_concept = self.idx_to_concept[pred_idx]
        
        return {
            'text': text,
            'predicted_concept': predicted_concept,
            'confidence': confidence
        }
    
    def predict_batch(self, texts, batch_size=32):
        """Predict concepts for a batch of texts"""
        predictions = []
        confidences = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                batch_confidences = [probs[i][pred].item() for i, pred in enumerate(preds)]
                batch_predictions = [self.idx_to_concept[pred.item()] for pred in preds]
                
                predictions.extend(batch_predictions)
                confidences.extend(batch_confidences)
        
        return predictions, confidences

def process_csv(predictor, input_csv, output_csv, text_col=None, label_col=None):
    """Process CSV file and generate predictions"""
    logger.info(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # If text column not specified, try to find it
    if text_col is None:
        if 'combined_text' in df.columns:
            text_col = 'combined_text'
        elif 'attribute_name' in df.columns and 'description' in df.columns:
            df['combined_text'] = df['attribute_name'] + ' - ' + df['description']
            text_col = 'combined_text'
        else:
            raise ValueError("Could not determine text column. Please specify using --text_col")
    
    # Make predictions
    predictions, confidences = predictor.predict_batch(df[text_col].tolist())
    
    # Add predictions to dataframe
    df['predicted_concept'] = predictions
    df['confidence'] = confidences
    
    # If labels are available, compute metrics
    if label_col and label_col in df.columns:
        logger.info("Computing classification metrics...")
        report = classification_report(
            df[label_col],
            predictions,
            output_dict=True,
            zero_division=0
        )
        
        # Log detailed metrics
        logger.info("\nClassification Report:")
        for label in sorted(report.keys()):
            if label in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            metrics = report[label]
            logger.info(f"\nLabel: {label}")
            logger.info(f"Precision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            logger.info(f"F1-Score: {metrics['f1-score']:.3f}")
            logger.info(f"Support: {metrics['support']}")
        
        logger.info(f"\nOverall Accuracy: {report['accuracy']:.3f}")
        logger.info(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
    
    # Save results
    logger.info(f"Saving predictions to: {output_csv}")
    df.to_csv(output_csv, index=False)
    
def main():
    parser = argparse.ArgumentParser(description='Predict concepts for text inputs')
    parser.add_argument('--model_path', required=True, help='Path to the saved model')
    parser.add_argument('--input_csv', help='Input CSV file for batch prediction')
    parser.add_argument('--output_csv', help='Output CSV file for batch prediction results')
    parser.add_argument('--text', help='Single text input for prediction')
    parser.add_argument('--text_col', help='Column name containing text in CSV')
    parser.add_argument('--label_col', help='Column name containing true labels in CSV (if available)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ConceptPredictor(args.model_path)
    
    if args.text:
        # Single text prediction
        result = predictor.predict_single(args.text)
        print("\nSingle Text Prediction:")
        print(f"Text: {result['text']}")
        print(f"Predicted Concept: {result['predicted_concept']}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    elif args.input_csv:
        # Batch prediction from CSV
        if not args.output_csv:
            args.output_csv = args.input_csv.rsplit('.', 1)[0] + '_predictions.csv'
        
        process_csv(
            predictor,
            args.input_csv,
            args.output_csv,
            args.text_col,
            args.label_col
        )
    
    else:
        parser.error("Either --text or --input_csv must be provided")

if __name__ == "__main__":
    main()








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
    def __init__(self, texts, concepts, tokenizer, max_length=512):
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
            
            # Validate all attribute concepts exist in concepts.csv
            attr_concepts = set(self.attr_df['concept'].unique())
            concept_defs = set(self.concept_df['concept'].unique())
            
            # Check for concepts in attributes but missing from definitions
            missing_defs = attr_concepts - concept_defs
            if missing_defs:
                self.logger.error(f"Found concepts in attributes without definitions: {missing_defs}")
                raise ValueError("Some concepts in attributes.csv are missing from concepts.csv")
            
            # Check for unused concept definitions
            unused_concepts = concept_defs - attr_concepts
            if unused_concepts:
                self.logger.warning(f"Found unused concept definitions: {unused_concepts}")
            
            # Create label encoder for concepts (using all defined concepts)
            self.unique_concepts = sorted(self.concept_df['concept'].unique())
            # Save labels to label.txt
            with open('label.txt', 'w') as f:
                f.write('\n'.join(self.unique_concepts))
            self.logger.info(f"Saved {len(self.unique_concepts)} labels to label.txt")
            
            self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.unique_concepts)}
            self.idx_to_concept = {idx: concept for concept, idx in self.concept_to_idx.items()}
            
            self.num_labels = len(self.unique_concepts)
            self.logger.info(f"Found {self.num_labels} unique concepts")
            
            # Analyze class distribution
            concept_counts = self.attr_df['concept'].value_counts()
            min_samples = concept_counts.min()
            max_samples = concept_counts.max()
            mean_samples = concept_counts.mean()
            median_samples = concept_counts.median()
            
            # Log detailed statistics
            self.logger.info("\nPer-Label Statistics:")
            self.logger.info(f"Total number of samples: {len(self.attr_df)}")
            self.logger.info(f"Number of unique concepts: {self.num_labels}")
            self.logger.info(f"Average samples per concept: {mean_samples:.2f}")
            self.logger.info(f"Median samples per concept: {median_samples:.2f}")
            self.logger.info(f"Min samples: {min_samples} (Concept: '{concept_counts.index[-1]}')")
            self.logger.info(f"Max samples: {max_samples} (Concept: '{concept_counts.index[0]}')")
            
            # Log detailed class distribution with percentages
            self.logger.info("\nDetailed Class Distribution:")
            total_samples = len(self.attr_df)
            for concept, count in concept_counts.items():
                percentage = (count / total_samples) * 100
                self.logger.info(f"Concept '{concept}': {count} samples ({percentage:.1f}%)")
                
                # Get some example attributes for this concept
                examples = self.attr_df[self.attr_df['concept'] == concept]['combined_text'].head(2)
                self.logger.info(f"Example attributes:")
                for ex in examples:
                    self.logger.info(f"  - {ex}")
                self.logger.info("")  # Empty line for readability
            
            # Check for significant imbalance
            if max_samples / min_samples > 10:
                self.logger.warning(f"\nSignificant class imbalance detected:")
                self.logger.warning(f"- Ratio of max/min samples: {max_samples/min_samples:.2f}")
                self.logger.warning(f"- Most frequent concept has {max_samples} samples")
                self.logger.warning(f"- Least frequent concept has {min_samples} samples")
                
                # Suggest potential solutions
                self.logger.warning("\nConsider using one of these techniques to handle imbalance:")
                self.logger.warning("1. Class weights in loss function")
                self.logger.warning("2. Oversampling minority classes")
                self.logger.warning("3. Undersampling majority classes")
                self.logger.warning("4. Using techniques like SMOTE for synthetic samples")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def create_train_val_test_split(self, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Create stratified split while ensuring label representation in all splits
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

            # Log split statistics
            self.logger.info("\nSplit Statistics:")
            self.logger.info(f"Total samples: {len(self.attr_df)}")
            self.logger.info(f"Train set: {len(train_df)} samples ({train_size*100:.1f}%)")
            self.logger.info(f"Validation set: {len(val_df)} samples ({val_size*100:.1f}%)")
            self.logger.info(f"Test set: {len(test_df)} samples ({test_size*100:.1f}%)")

            # Verify label distribution in each split
            self.logger.info("\nLabel Distribution Across Splits:")
            for concept in self.unique_concepts:
                train_count = len(train_df[train_df['concept'] == concept])
                val_count = len(val_df[val_df['concept'] == concept])
                test_count = len(test_df[test_df['concept'] == concept])
                total_count = train_count + val_count + test_count

                self.logger.info(f"\nConcept: {concept}")
                self.logger.info(f"  Total: {total_count}")
                self.logger.info(f"  Train: {train_count} ({train_count/total_count*100:.1f}%)")
                self.logger.info(f"  Val: {val_count} ({val_count/total_count*100:.1f}%)")
                self.logger.info(f"  Test: {test_count} ({test_count/total_count*100:.1f}%)")

                # Verify minimum samples in validation and test
                if val_count < 5:
                    self.logger.warning(f"  Warning: Very few samples ({val_count}) in validation for concept '{concept}'")
                if test_count < 5:
                    self.logger.warning(f"  Warning: Very few samples ({test_count}) in test for concept '{concept}'")

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
        
        # Initialize SentenceTransformer and classification head
        self.sentence_transformer = SentenceTransformer(model_name)
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.sentence_transformer.get_sentence_embedding_dimension(), 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, num_labels)
        )
        
        # Combined model
        class CombinedModel(nn.Module):
            def __init__(self, sentence_transformer, classifier):
                super().__init__()
                self.sentence_transformer = sentence_transformer
                self.classifier = classifier
                
            def forward(self, input_ids, attention_mask):
                embeddings = self.sentence_transformer({'input_ids': input_ids, 'attention_mask': attention_mask})
                return self.classifier(embeddings)

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
        Compute metrics for evaluation with proper handling of zero division
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Debug: analyze predictions
        unique_labels = np.unique(labels)
        unique_preds = np.unique(predictions)
        
        self.logger.info("\nDebug Information:")
        self.logger.info(f"Unique true labels in this batch: {unique_labels}")
        self.logger.info(f"Unique predicted labels in this batch: {unique_preds}")
        
        # Find labels that exist but were never predicted
        missing_preds = set(unique_labels) - set(unique_preds)
        if missing_preds:
            self.logger.warning(f"Labels that were never predicted: {missing_preds}")
            
        # Count predictions per class
        pred_counts = np.bincount(predictions, minlength=len(self.idx_to_concept))
        true_counts = np.bincount(labels, minlength=len(self.idx_to_concept))
        
        self.logger.info("\nPrediction Distribution:")
        for idx in range(len(self.idx_to_concept)):
            concept = self.idx_to_concept[idx]
            self.logger.info(f"Concept {concept}:")
            self.logger.info(f"  True count: {true_counts[idx]}")
            self.logger.info(f"  Predicted count: {pred_counts[idx]}")

        # Get detailed classification report
        report = classification_report(
            labels, 
            predictions, 
            output_dict=True,
            zero_division=0  # Explicitly handle zero division
        )

        # Calculate macro metrics with zero_division parameter
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, 
            predictions, 
            average='macro',
            zero_division=0  # Handle zero division cases
        )

        # Add per-class metrics to identify problematic classes
        per_class_metrics = {}
        for label in np.unique(labels):
            label_name = str(label)
            if label_name in report:
                per_class_metrics[f'class_{label}_precision'] = report[label_name]['precision']
                per_class_metrics[f'class_{label}_recall'] = report[label_name]['recall']
                per_class_metrics[f'class_{label}_f1'] = report[label_name]['f1-score']
                per_class_metrics[f'class_{label}_support'] = report[label_name]['support']

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            **per_class_metrics  # Include per-class metrics
        }

        # Log class distribution
        class_distribution = np.bincount(labels)
        self.logger.info(f"Class distribution in evaluation: {class_distribution}")
        pred_distribution = np.bincount(predictions)
        self.logger.info(f"Prediction distribution: {pred_distribution}")

        return metrics

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
                max_length=512,
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
