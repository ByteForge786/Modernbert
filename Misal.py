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
