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
