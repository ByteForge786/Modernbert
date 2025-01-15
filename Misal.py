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
