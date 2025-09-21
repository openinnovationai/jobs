from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, TrainerCallback
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

class TrainingUtils:
    def __init__(self, 
                 backbone_model: str, 
                 dataset_name: str, 
                 dataset_task: str,
                ):
        self.__backbone_model = backbone_model
        self.__dataset_name = dataset_name
        self.__dataset_task = dataset_task
        self.__tokenizer = None
        self.__dataset = None
        self.__model = None
        self.__peft_model = None
        self.__trainer = None
        
    def get_tokenizer(self):
        if self.__tokenizer is None:
            self.__tokenizer = AutoTokenizer.from_pretrained(self.__backbone_model)
            # Add padding token if it doesn't exist
            if self.__tokenizer.pad_token is None:
                self.__tokenizer.pad_token = self.__tokenizer.eos_token
        return self.__tokenizer

    def get_dataset(self):
        if self.__dataset is None:
            # Load the dataset
            dataset = load_dataset(self.__dataset_name, self.__dataset_task)
            
            # Print dataset info for debugging
            print("Dataset structure:")
            print(f"Train dataset: {dataset['train']}")
            print(f"Sample: {dataset['train'][0]}")
            
            # Tokenize the dataset
            encoded_dataset = dataset.map(
                self.tokenize_fn, 
                batched=True,
                remove_columns=dataset["train"].column_names  # Remove original columns
            )
            
            self.__dataset = encoded_dataset
        return self.__dataset
    
    def tokenize_fn(self, examples):
        tokenizer = self.get_tokenizer()
        
        # Debug: print what we're receiving
        print(f"Tokenizing batch of size: {len(examples['sentence'])}")
        print(f"First sentence: {examples['sentence'][0]}")
        
        # Tokenize the sentences
        tokenized = tokenizer(
            examples["sentence"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        
        # Add labels - make sure they're the right type
        tokenized["labels"] = examples["label"]
            
        return tokenized

    def get_model(self):
        if self.__model is None:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            self.__model = AutoModelForSequenceClassification.from_pretrained(
                self.__backbone_model,
                num_labels=2,
                quantization_config=quantization_config,
                # Removed device_map="auto" for distributed training compatibility
            )
        
        return self.__model

    def get_peft_model(self):
        if self.__peft_model is None:
            model = self.get_model()
            
            # Enable gradient checkpointing
            model.gradient_checkpointing_enable()
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            
            # Apply LoRA configuration
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query", "value"],  # Updated for BERT
                lora_dropout=0.05,  
                bias="none",
                task_type="SEQ_CLS",
            )
            
            self.__peft_model = get_peft_model(model, lora_config)
            
            # Print trainable parameters info
            self.__peft_model.print_trainable_parameters()

        return self.__peft_model

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy.compute(predictions=preds, references=labels)
        print(f"accuracy: {acc}")
        return acc

    def get_trainer(self, 
                     output_dir: str = "./output",
                     learning_rate: float = 2e-4,
                     per_device_train_batch_size: int = 16,
                     per_device_eval_batch_size: int = 16,
                     num_train_epochs: int = 4,
                     logging_steps: int = 10
                    ) -> Trainer:
        model = self.get_peft_model()
        encoded_dataset = self.get_dataset()
        tokenizer = self.get_tokenizer()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            save_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            warmup_steps=100,
            weight_decay=0.01,
            dataloader_drop_last=False, 
            dataloader_num_workers=0,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
        )

        return trainer
