def predict_batch(self, texts, batch_size=32):
        """
        Make predictions for a batch of texts
        Args:
            texts (list): List of strings to classify
            batch_size (int): Batch size for prediction
        Returns:
            predictions (list): List of predicted concept labels
            confidences (list): List of highest confidence scores
            all_probs (list): List of probability distributions over all concepts
        """
        self.model.eval()
        predictions = []
        confidences = []
        all_probs = []
        
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
                batch_confidences = torch.max(probs, dim=-1).values
                
                predictions.extend([self.idx_to_concept[pred.item()] for pred in batch_predictions])
                confidences.extend(batch_confidences.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
        
        return predictions, confidences, all_probs

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
    predictions, confidences, all_probabilities = predictor.predict_batch(df[text_col].tolist())
    
    # Add predictions and confidence to dataframe
    df['predicted_concept'] = predictions
    df['confidence'] = confidences
    
    # Add top 3 predictions with their confidences
    top_k = 3
    top_predictions = []
    top_confidences = []
    
    for probs in all_probabilities:
        # Get top k indices and their probabilities
        top_k_indices = np.argsort(probs)[-top_k:][::-1]
        top_k_probs = np.array(probs)[top_k_indices]
        
        # Convert indices to concept labels
        top_k_labels = [predictor.idx_to_concept[idx] for idx in top_k_indices]
        
        # Format predictions and confidences
        top_predictions.append(' | '.join(top_k_labels))
        top_confidences.append(' | '.join([f'{prob:.4f}' for prob in top_k_probs]))
    
    df['top_3_predictions'] = top_predictions
    df['top_3_confidences'] = top_confidences
    
    # If original labels are available, compute metrics
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
            logger.info(f"\nConcept: {label}")
            logger.info(f"Precision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            logger.info(f"F1-Score: {metrics['f1-score']:.3f}")
            logger.info(f"Support: {metrics['support']}")
            
            # Additional statistics
            correct_predictions = len(df[(df[label_col] == label) & (df['predicted_concept'] == label)])
            total_predictions = len(df[df['predicted_concept'] == label])
            total_actual = len(df[df[label_col] == label])
            
            logger.info(f"Correct Predictions: {correct_predictions}")
            logger.info(f"Total Predictions: {total_predictions}")
            logger.info(f"Actual Occurrences: {total_actual}")
        
        logger.info(f"\nOverall Accuracy: {report['accuracy']:.3f}")
        logger.info(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
        
        # Save metrics to a separate file
        metrics_file = output_csv.rsplit('.', 1)[0] + '_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Detailed metrics saved to: {metrics_file}")
    
    # Save results
    logger.info(f"Saving predictions to: {output_csv}")
    df.to_csv(output_csv, index=False)





model_config = self.model.config
            model_config.id2label = {str(idx): label for idx, label in self.idx_to_concept.items()}
            model_config.label2id = {label: idx for idx, label in self.idx_to_concept.items()}
            model_config.best_metric = trainer.state.best_metric
            model_config.save_pretrained(best_model_path)
            
            # Save training metrics history
            metrics_path = os.path.join(best_model_path, "training_metrics.json")
            training_history = {
                'log_history': trainer.state.log_history,
                'best_metric': trainer.state.best_metric,
                'best_model_checkpoint': trainer.state.best_model_checkpoint,
                'best_metric_value': trainer.state.best_metric_value,
                'label_mapping': {
                    'id2label': model_config.id2label,
                    'label2id': model_config.label2id
                }
            }
            with open(metrics_path, 'w') as f:
                json.dump(training_history, f, indent=2)
            
            self.best_model_path = best_model_path







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










class ConceptClassifier:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=16, num_epochs=3):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.best_model_path = None
        self.best_score = 0.0
        
        # Define the fixed split mapping for NLI cache files
        self.split_mapping = {
            '109041164578947912': 'train',
            '204910560809856016': 'val',
            '296292887499757834': 'test'
        }

    def load_from_nli_cache(self, nli_cache_dir):
        """
        Load pre-computed NLI pairs from cache directory
        """
        try:
            self.logger.info(f"Attempting to load NLI cache from {nli_cache_dir}")
            train_pairs = []
            train_labels = []
            val_pairs = []
            val_labels = []
            test_pairs = []
            test_labels = []
            
            # Load each split based on the mapping
            for file_id, split_name in self.split_mapping.items():
                cache_file = os.path.join(nli_cache_dir, f"{file_id}.pkl")
                if not os.path.exists(cache_file):
                    raise FileNotFoundError(f"Cache file not found: {cache_file}")
                
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                if split_name == 'train':
                    train_pairs = cache_data['pairs']
                    train_labels = cache_data['labels']
                elif split_name == 'val':
                    val_pairs = cache_data['pairs']
                    val_labels = cache_data['labels']
                elif split_name == 'test':
                    test_pairs = cache_data['pairs']
                    test_labels = cache_data['labels']
            
            return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels
            
        except Exception as e:
            self.logger.error(f"Error loading NLI cache: {str(e)}")
            raise

    def prepare_training_data(self, train_df=None, val_df=None, concept_df=None, nli_cache_dir=None):
        """
        Prepare training data either from NLI cache or by generating new pairs
        """
        # Try loading from NLI cache first if directory is provided
        if nli_cache_dir:
            try:
                return self.load_from_nli_cache(nli_cache_dir)
            except Exception as e:
                self.logger.warning(f"Failed to load from NLI cache: {str(e)}")
                if train_df is None or val_df is None or concept_df is None:
                    raise ValueError("Raw data not provided as fallback")
        
        # Fall back to generating pairs if NLI cache loading fails or isn't requested
        if not all([train_df is not None, val_df is not None, concept_df is not None]):
            raise ValueError("Missing required dataframes for pair generation")
            
        # Check if processed pairs exist
        if os.path.exists('processed_pairs.pkl'):
            self.logger.info("Loading pre-processed pairs from disk")
            with open('processed_pairs.pkl', 'rb') as f:
                pairs_data = pickle.load(f)
            return (pairs_data['train_pairs'], pairs_data['train_labels'],
                   pairs_data['val_pairs'], pairs_data['val_labels'],
                   pairs_data.get('test_pairs', []), pairs_data.get('test_labels', []))

        # Rest of the existing prepare_training_data implementation...
        def create_pairs(row, concepts_dict):
            desc = row['combined_text']
            true_concept = row['concept']
            pairs = []
            labels = []
            
            pairs.append((desc, concepts_dict[true_concept]))
            labels.append(1)
            
            other_concepts = [c for c in concepts_dict.keys() if c != true_concept]
            selected_negative = np.random.choice(other_concepts, size=2, replace=False)
            for neg_concept in selected_negative:
                pairs.append((desc, concepts_dict[neg_concept]))
                labels.append(0)
            
            return pairs, labels

        concepts_dict = dict(zip(concept_df['concept'], concept_df['concept_definition']))
        
        with ThreadPoolExecutor() as executor:
            train_results = list(executor.map(
                lambda row: create_pairs(row, concepts_dict),
                [row for _, row in train_df.iterrows()]
            ))
            
            val_results = list(executor.map(
                lambda row: create_pairs(row, concepts_dict),
                [row for _, row in val_df.iterrows()]
            ))

        train_pairs = [pair for result in train_results for pair in result[0]]
        train_labels = [label for result in train_results for label in result[1]]
        val_pairs = [pair for result in val_results for pair in result[0]]
        val_labels = [label for result in val_results for label in result[1]]

        # Save pairs to disk
        pairs_data = {
            'train_pairs': train_pairs,
            'train_labels': train_labels,
            'val_pairs': val_pairs,
            'val_labels': val_labels
        }
        
        save_path = 'processed_pairs.pkl'
        self.logger.info(f"Saving processed pairs to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(pairs_data, f)
            
        return train_pairs, train_labels, val_pairs, val_labels, [], []






class ConceptClassifier:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=16, num_epochs=3):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.best_model_path = None
        self.best_score = 0.0
        
        # Define the fixed split mapping for NLI cache files
        self.split_mapping = {
            '109041164578947912': 'train',
            '204910560809856016': 'val',
            '296292887499757834': 'test'
        }

    def load_from_nli_cache(self, nli_cache_dir):
        """
        Load pre-computed NLI pairs from cache directory
        """
        try:
            self.logger.info(f"Attempting to load NLI cache from {nli_cache_dir}")
            train_pairs = []
            train_labels = []
            val_pairs = []
            val_labels = []
            test_pairs = []
            test_labels = []
            
            # Load each split based on the mapping
            for file_id, split_name in self.split_mapping.items():
                cache_file = os.path.join(nli_cache_dir, f"{file_id}.pkl")
                if not os.path.exists(cache_file):
                    raise FileNotFoundError(f"Cache file not found: {cache_file}")
                
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                if split_name == 'train':
                    train_pairs = cache_data['pairs']
                    train_labels = cache_data['labels']
                elif split_name == 'val':
                    val_pairs = cache_data['pairs']
                    val_labels = cache_data['labels']
                elif split_name == 'test':
                    test_pairs = cache_data['pairs']
                    test_labels = cache_data['labels']
            
            return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels
            
        except Exception as e:
            self.logger.error(f"Error loading NLI cache: {str(e)}")
            raise

    def prepare_training_data(self, train_df=None, val_df=None, concept_df=None, nli_cache_dir=None):
        """
        Prepare training data either from NLI cache or by generating new pairs
        """
        # Try loading from NLI cache first if directory is provided
        if nli_cache_dir:
            try:
                return self.load_from_nli_cache(nli_cache_dir)
            except Exception as e:
                self.logger.warning(f"Failed to load from NLI cache: {str(e)}")
                if train_df is None or val_df is None or concept_df is None:
                    raise ValueError("Raw data not provided as fallback")
        
        # Fall back to generating pairs if NLI cache loading fails or isn't requested
        if not all([train_df is not None, val_df is not None, concept_df is not None]):
            raise ValueError("Missing required dataframes for pair generation")
            
        # Check if processed pairs exist
        if os.path.exists('processed_pairs.pkl'):
            self.logger.info("Loading pre-processed pairs from disk")
            with open('processed_pairs.pkl', 'rb') as f:
                pairs_data = pickle.load(f)
            return (pairs_data['train_pairs'], pairs_data['train_labels'],
                   pairs_data['val_pairs'], pairs_data['val_labels'],
                   pairs_data.get('test_pairs', []), pairs_data.get('test_labels', []))

        # Rest of the existing prepare_training_data implementation...
        def create_pairs(row, concepts_dict):
            desc = row['combined_text']
            true_concept = row['concept']
            pairs = []
            labels = []
            
            pairs.append((desc, concepts_dict[true_concept]))
            labels.append(1)
            
            other_concepts = [c for c in concepts_dict.keys() if c != true_concept]
            selected_negative = np.random.choice(other_concepts, size=2, replace=False)
            for neg_concept in selected_negative:
                pairs.append((desc, concepts_dict[neg_concept]))
                labels.append(0)
            
            return pairs, labels

        concepts_dict = dict(zip(concept_df['concept'], concept_df['concept_definition']))
        
        with ThreadPoolExecutor() as executor:
            train_results = list(executor.map(
                lambda row: create_pairs(row, concepts_dict),
                [row for _, row in train_df.iterrows()]
            ))
            
            val_results = list(executor.map(
                lambda row: create_pairs(row, concepts_dict),
                [row for _, row in val_df.iterrows()]
            ))

        train_pairs = [pair for result in train_results for pair in result[0]]
        train_labels = [label for result in train_results for label in result[1]]
        val_pairs = [pair for result in val_results for pair in result[0]]
        val_labels = [label for result in val_results for label in result[1]]

        # Save pairs to disk
        pairs_data = {
            'train_pairs': train_pairs,
            'train_labels': train_labels,
            'val_pairs': val_pairs,
            'val_labels': val_labels
        }
        
        save_path = 'processed_pairs.pkl'
        self.logger.info(f"Saving processed pairs to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(pairs_data, f)
            
        return train_pairs, train_labels, val_pairs, val_labels, [], []








def compute_metrics(eval_pred):
    """Compute metrics for evaluation without downloading"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy manually
    accuracy = (predictions == labels).mean()
    
    # Calculate confusion matrix values
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate precision, recall, f1
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate specificity and negative predictive value
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'npv': float(npv),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }







# 1. Add constant at the top after imports
MAX_SEQUENCE_LENGTH = 512  # Maximum sequence length for BERT models

# 2. Update prepare_and_analyze_data function's get_sequence_length:
def get_sequence_length(row):
    tokens = tokenizer(
        row['description'],
        row['concept_definition'],
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors='pt'
    )
    return len(tokens['input_ids'][0])

# 3. Update NLIDataset class:
class NLIDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = MAX_SEQUENCE_LENGTH):
        """Initialize NLI dataset"""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.premises = data['premise'].tolist()
        self.hypotheses = data['hypothesis'].tolist()
        self.labels = [LABEL_MAP[label] for label in data['label']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            text=self.premises[idx],
            text_pair=self.hypotheses[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None
        )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': self.labels[idx]
        }

# 4. Update data_collator in train_nli_model function:
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding='max_length',
    max_length=MAX_SEQUENCE_LENGTH,
    return_tensors="pt"
)

# 5. Update NLIPredictor class predict methods:
def predict(self, premise: str, hypothesis: str) -> Dict:
    """Make NLI prediction following standard format"""
    inputs = self.tokenizer(
        premise,
        hypothesis,
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt"
    )

def predict_batch(self, data: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Batch prediction for multiple examples"""
    results = []
    
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data.iloc[i:i+batch_size]
        inputs = self.tokenizer(
            batch['premise'].tolist(),
            batch['hypothesis'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        )











# 1. Add this constant at the top of the file, after the imports:
MAX_SEQUENCE_LENGTH = 512  # Maximum sequence length for BERT models

# 2. The only needed change in NLIDataset class is to use the constant:
class NLIDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = MAX_SEQUENCE_LENGTH):
        """Initialize NLI dataset"""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.premises = data['premise'].tolist()
        self.hypotheses = data['hypothesis'].tolist()
        self.labels = [LABEL_MAP[label] for label in data['label']]
    
    # Rest of the class remains exactly the same

# 3. Update DataCollator initialization in train_nli_model function:
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    max_length=MAX_SEQUENCE_LENGTH,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

# 4. Update the predict methods in NLIPredictor class to use the constant:
def predict(self, premise: str, hypothesis: str) -> Dict:
    """Make NLI prediction following standard format"""
    inputs = self.tokenizer(
        premise,
        hypothesis,
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,  # Use constant here
        return_tensors="pt"
    )

def predict_batch(self, data: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    # Only change the max_length parameter in the tokenizer call
    inputs = self.tokenizer(
        batch['premise'].tolist(),
        batch['hypothesis'].tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,  # Use constant here
        return_tensors="pt"
    )





class DataProcessor:
    def __init__(self, model_id="bert-base-uncased", batch_size=32, cache_dir="nli_cache"):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_id,
            device=-1  # CPU-based
        )
        # Add cache directory
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def create_nli_dataset(self, df: pd.DataFrame, k_negatives: int = 3) -> List[Dict]:
        """Create NLI pairs with labels following standard format"""
        # Generate cache filename based on dataframe hash and k_negatives
        df_hash = pd.util.hash_pandas_object(df).sum()
        cache_file = os.path.join(self.cache_dir, f"nli_pairs_{df_hash}_{k_negatives}.json")
        
        if os.path.exists(cache_file):
            logging.info(f"Loading cached NLI pairs from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        logging.info("Starting NLI dataset creation...")
        
        # Existing code remains exactly the same...
        
        # Save to cache before returning
        with open(cache_file, 'w') as f:
            json.dump(nli_pairs, f)
            
        return nli_pairs





def prepare_and_analyze_data(attributes_df, definitions_df, test_size=0.1, val_size=0.1):
    """Prepare and analyze data distribution before NLI processing"""
    logging.info("Starting data preparation and analysis...")
    
    # Validate input data - Fix the column validation
    attributes_required = ['domain', 'concept', 'description', 'attribute_name']
    definitions_required = ['domain', 'concept', 'concept_definition']
    
    # Check attributes DataFrame
    missing_attrs = [col for col in attributes_required if col not in attributes_df.columns]
    if missing_attrs:
        raise ValueError(f"Missing required columns in attributes_df: {missing_attrs}")
    
    # Check definitions DataFrame
    missing_defs = [col for col in definitions_required if col not in definitions_df.columns]
    if missing_defs:
        raise ValueError(f"Missing required columns in definitions_df: {missing_defs}")
    
    # Check for missing values
    for df, name in [(attributes_df, 'attributes'), (definitions_df, 'definitions')]:
        missing_values = df.isnull().sum()
        if missing_values.any():
            logging.warning(f"Found missing values in {name}_df:\n{missing_values[missing_values > 0]}")
            df.dropna(inplace=True)
    
    # Combine attribute_name with description and add augmentations
    def augment_description(row):
        # Original format: "{attribute_name} - {description}"
        base = f"{row['attribute_name']} - {row['description']}"
        
        # Augmentations
        variations = [
            base,  # Original
            f"This {row['attribute_name']}: {row['description']}", # Variation 1
            f"The {row['attribute_name']} is described as: {row['description']}", # Variation 2
            f"Regarding {row['attribute_name']}, {row['description']}" # Variation 3
        ]
        return random.choice(variations)

    # Apply augmentation to create enhanced description
    attributes_df['enhanced_description'] = attributes_df.apply(augment_description, axis=1)
    
    # Merge dataframes using enhanced_description
    df = pd.merge(
        attributes_df[['domain', 'concept', 'enhanced_description']], 
        definitions_df[['domain', 'concept', 'concept_definition']], 
        on=['domain', 'concept']
    )
    
    # Rename enhanced_description to description for compatibility with rest of the code
    df = df.rename(columns={'enhanced_description': 'description'})
    
    # Rest of the function remains the same...
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)




def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def train_nli_model(train_data, val_data=None, model_id="answerdotai/ModernBERT-base", output_dir="nli-model"):
    """
    Standard NLI fine-tuning
    """
    logging.info("Starting NLI model training...")
    
    # Define label mappings
    label2id = {"contradiction": 0, "entailment": 1}
    id2label = {0: "contradiction", 1: "entailment"}
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=2,
        label2id=label2id,
        id2label=id2label
    )
    
    # Create datasets
    train_dataset = NLIDataset(train_data, tokenizer)
    eval_dataset = NLIDataset(val_data, tokenizer) if val_data is not None else None
    
    # Standard NLI training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Optional: Print detailed metrics after training
    if eval_dataset:
        eval_results = trainer.evaluate()
        logging.info(f"Final evaluation accuracy: {eval_results['eval_accuracy']:.4f}")
    
    return trainer, tokenizer


# Training configuration best practices
training_args = TrainingArguments(
    output_dir="nli_model",
    # Learning rate setup
    learning_rate=2e-5,  # Lower than standard fine-tuning
    warmup_ratio=0.1,    # Important for NLI
    
    # Training dynamics
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # Effective batch size = 32
    
    # Evaluation strategy
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    
    # Early stopping
    load_best_model_at_end=True,
    metric_for_best_model="entail_f1",  # Focus on entailment performance
    greater_is_better=True,
    
    # Prevent overfitting
    weight_decay=0.01,
    
    # Logging
    logging_dir="./logs",
    logging_steps=10
)

# Early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

# Model initialization with proper config
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    label2id={"contradiction": 0, "entailment": 1},
    id2label={0: "contradiction", 1: "entailment"},
)

# Training with proper callbacks
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_nli_metrics,
    callbacks=[early_stopping],
    # Add class weights if dataset is imbalanced
    class_weights=torch.tensor([1.0, 1.2])  # Slight emphasis on entailment
)
