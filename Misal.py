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
