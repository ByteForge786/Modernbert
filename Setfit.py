import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from setfit import SetFitModel, SetFitTrainer
import torch
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
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class ConceptDataset:
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
            self.unique_concepts = self.concept_df['concept'].unique()
            self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.unique_concepts)}
            self.idx_to_concept = {idx: concept for concept, idx in self.concept_to_idx.items()}
            
            self.logger.info(f"Found {len(self.unique_concepts)} unique concepts")
            
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

            self.logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df

        except Exception as e:
            self.logger.error(f"Error in split: {str(e)}")
            raise

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

    def prepare_training_data(self, train_df, val_df, concept_df):
        # Check if processed pairs exist
        if os.path.exists('processed_pairs.pkl'):
            self.logger.info("Loading pre-processed pairs from disk")
            with open('processed_pairs.pkl', 'rb') as f:
                pairs_data = pickle.load(f)
            return (pairs_data['train_pairs'], pairs_data['train_labels'],
                   pairs_data['val_pairs'], pairs_data['val_labels'])
        """
        Prepare data for SetFit training with parallel processing
        """
        def create_pairs(row, concepts_dict):
            desc = row['combined_text']  # Using combined attribute_name and description
            true_concept = row['concept']
            pairs = []
            labels = []
            
            # Create positive pair
            pairs.append((desc, concepts_dict[true_concept]))
            labels.append(1)
            
            # Create negative pairs
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
            
        return train_pairs, train_labels, val_pairs, val_labels

    def train(self, train_pairs, train_labels, val_pairs, val_labels):
        """
        Train the model with checkpointing and logging
        """
        try:
            self.logger.info("Preparing training data...")
            train_pairs, train_labels, val_pairs, val_labels = self.prepare_training_data(
                train_df, val_df, concept_df
            )

            self.logger.info("Initializing model...")
            model = SetFitModel.from_pretrained(
                self.model_name,
                head_params={
                    "dropout": 0.2,
                    "num_layers": 2,
                    "hidden_size": 768,
                    "use_differentiable_head": True
                }
            )
            
            # Create checkpoint directory
            checkpoint_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            trainer = SetFitTrainer(
                model=model,
                train_dataset=(train_pairs, train_labels),
                eval_dataset=(val_pairs, val_labels),
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                column_mapping={"text_a": 0, "text_b": 1}
            )

            # Training with checkpointing
            for epoch in range(self.num_epochs):
                self.logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                trainer.train()
                
                # Evaluate
                metrics = self.evaluate(trainer.model, val_pairs, val_labels)
                
                # Save checkpoint if better
                if metrics['macro_f1'] > self.best_score:
                    self.best_score = metrics['macro_f1']
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}")
                    trainer.model.save_pretrained(checkpoint_path)
                    self.best_model_path = checkpoint_path
                    self.logger.info(f"New best model saved with F1: {self.best_score:.4f}")

                self.logger.info(f"Epoch {epoch + 1} metrics: {metrics}")

            return self.best_model_path, self.best_score

        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise

    def evaluate(self, model, eval_pairs, eval_labels):
        """
        Evaluate model performance with detailed metrics
        """
        predictions = model.predict(eval_pairs)
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            eval_labels, predictions, average='macro'
        )
        
        # Get per-class metrics
        class_report = classification_report(
            eval_labels, predictions, output_dict=True
        )
        
        metrics = {
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'per_class': class_report
        }
        
        return metrics

def main():
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize classifier
        classifier = ConceptClassifier(batch_size=32, num_epochs=5)
        
        try:
            # First try to use NLI cache
            train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = \
                classifier.prepare_training_data(nli_cache_dir='nli_cache')
            logger.info("Successfully loaded pre-computed NLI pairs")
        except Exception as e:
            logger.warning(f"Failed to load NLI cache: {e}")
            logger.info("Falling back to generating pairs from raw data...")
            
            # Initialize dataset for fallback
            dataset = ConceptDataset('attributes.csv', 'concepts.csv')
            train_df, val_df, test_df = dataset.create_train_val_test_split()
            
            # Generate pairs using fallback method
            train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = \
                classifier.prepare_training_data(
                    train_df=train_df,
                    val_df=val_df,
                    concept_df=dataset.concept_df
                )
        
        # Train the model using pre-computed pairs
        best_model_path, best_score = classifier.train(train_pairs, train_labels, val_pairs, val_labels)
        
        logger.info(f"Training completed. Best model saved at: {best_model_path}")
        logger.info(f"Best validation F1 score: {best_score:.4f}")
        
        # Load best model and evaluate on test set
        best_model = SetFitModel.from_pretrained(best_model_path)
        test_pairs, test_labels, _, _ = classifier.prepare_training_data(
            test_df, pd.DataFrame(), dataset.concept_df
        )
        
        test_metrics = classifier.evaluate(best_model, test_pairs, test_labels)
        logger.info("Test set metrics:")
        logger.info(json.dumps(test_metrics, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
