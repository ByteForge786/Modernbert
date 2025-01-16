from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict_single(model_path: str, premise: str, hypothesis: str):
    # Load saved model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    
    # Tokenize
    inputs = tokenizer(
        premise,
        hypothesis,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = float(probs[0][prediction])
    
    # Map prediction to label
    id2label = {1: "entailment", 0: "contradiction"}
    return {
        "label": id2label[prediction],
        "confidence": confidence
    }

# Usage
result = predict_single(
    model_path="path/to/saved/model",
    premise="The cat is sleeping.",
    hypothesis="The animal is at rest."
)
print(f"Prediction: {result['label']} (confidence: {result['confidence']:.4f})")





import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_from_csv(
    model_path: str,
    csv_path: str,
    premise_col: str = 'premise',
    hypothesis_col: str = 'hypothesis',
    batch_size: int = 32
):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Validate columns
    if premise_col not in df.columns or hypothesis_col not in df.columns:
        raise ValueError(f"Required columns {premise_col} and/or {hypothesis_col} not found in CSV")
    
    # Initialize results
    predictions = []
    confidences = []
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch[premise_col].tolist(),
            batch[hypothesis_col].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Get confidence scores
            batch_confidences = [float(probs[i][pred]) for i, pred in enumerate(preds)]
            
            # Convert to labels
            id2label = {1: "entailment", 0: "contradiction"}
            batch_predictions = [id2label[pred.item()] for pred in preds]
            
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
    
    # Add predictions to dataframe
    df['predicted_label'] = predictions
    df['confidence'] = confidences
    
    return df

# Usage
results_df = predict_from_csv(
    model_path="path/to/saved/model",
    csv_path="test_data.csv",
    premise_col="text1",
    hypothesis_col="text2"
)

# Save results
results_df.to_csv("predictions.csv", index=False)
