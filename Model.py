import torch
from transformers import  AutoModelForCausalLM, TrainingArguments, Trainer, GPT2Tokenizer
from datasets import load_dataset


def tokenize_function(examples):
    full_prompt = [
        f"{inp}{out}<|eos|>" for inp, out in zip(examples["input"], examples["output"])
    ]
    prompt_only = examples["input"]

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'eos_token': '<|eos|>'})

    tokenized_full = tokenizer(
        full_prompt,
        truncation=True,
        padding='max_length',
        max_length=512
    )

    tokenized_prompt = tokenizer(
        prompt_only,
        truncation=True,
        padding='max_length',
        max_length=512
    )

    labels = []
    for full_ids, prompt_ids in zip(tokenized_full["input_ids"], tokenized_prompt["input_ids"]):
        label = [-100 if token == prompt_token else token
                 for token, prompt_token in zip(full_ids, prompt_ids)]
        labels.append(label)

    tokenized_full["labels"] = labels
    return tokenized_full


class Model:
    def __init__(self, modelDir='gpt-model/model'):
        model_dir = modelDir

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'eos_token': '<|eos|>'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def inquire(self, client_full_name, client_age, purchase_history, discount_offer):
        prompt = ("""
        Instruction: Generate a promotional email for the given client based on their purchase history and discount offer.
        Input: Client name: """ + client_full_name + """
        Client age: """ + client_age + """ y.o.
        Purchase history: """ + purchase_history + """
        Discount offer: """ + discount_offer + """\n"""
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        output = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=300,
            temperature=0.90,
            top_p=0.95,
            do_sample=True,
        )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.strip()

    def train(self, dataset_path="dataset.json", numberOfEpochs=15):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        dataset = load_dataset("json", data_files=dataset_path)
        dataset = dataset["train"].train_test_split(test_size=0.2)

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'eos_token': '<|eos|>'})

        model = AutoModelForCausalLM.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids('<|pad|>')
        tokenizer.pad_token = '<|pad|>'
        tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'eos_token': '<|eos|>'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir='gpt-model',
            num_train_epochs=numberOfEpochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir='./gpt-model-logs',
            logging_steps=10,
            save_total_limit=1,
            save_strategy="epoch",
            eval_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
        )

        trainer.train()

        model_output_dir = 'gpt-model/model'
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
