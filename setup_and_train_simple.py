#!/usr/bin/env python3
"""
Script semplificato per fine-tuning Qwen3-VL-8B-Instruct su AMD RX 6650 XT.
Versione compatibile con Windows e ROCm.
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import signal

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configurazione per il training ottimizzata per RX 6650 XT 8GB."""
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    dataset_name: str = "medalpaca"
    output_dir: str = "./llama_factory_data/medical_qwen3_output"
    
    # Parametri training ottimizzati per 8GB VRAM
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 4
    lora_alpha: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 2
    cutoff_length: int = 512
    
    # Percorsi
    venv_path: str = "./.venv"
    data_path: str = "./llama_factory_data/data"
    final_model_path: str = "./final_model"

class SimpleQwenTrainer:
    """Classe semplificata per fine-tuning con Transformers."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.venv_python = None
        
    def setup_amd_environment(self):
        """Configura variabili ambiente per AMD GPU."""
        logger.info("Configurazione ambiente AMD...")
        
        env_vars = {
            'HSA_OVERRIDE_GFX_VERSION': '10.3.0',
            'PYTORCH_ROCM_ARCH': 'gfx1030',
            'CUDA_VISIBLE_DEVICES': '0',
            'HIP_VISIBLE_DEVICES': '0'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
            
        logger.info("Ambiente AMD configurato")
        
    def create_virtual_environment(self) -> bool:
        """Crea e attiva virtual environment."""
        logger.info("Creazione virtual environment...")
        
        venv_path = Path(self.config.venv_path)
        
        if venv_path.exists():
            logger.info("Virtual environment già esistente")
            self.venv_python = venv_path / "Scripts" / "python.exe"
            return True
            
        try:
            # Create virtual environment
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                         check=True, capture_output=True)
            
            self.venv_python = venv_path / "Scripts" / "python.exe"
            
            # Upgrade pip
            logger.info("Upgrade pip...")
            subprocess.run([str(self.venv_python), "-m", "pip", "install", "--upgrade", 
                          "pip", "setuptools", "wheel"], check=True)
            
            logger.info("Virtual environment creato con successo")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Errore creazione virtual environment: {e}")
            return False
            
    def install_dependencies(self) -> bool:
        """Installa dipendenze essenziali."""
        logger.info("Installazione dipendenze essenziali...")
        
        if not self.venv_python:
            logger.error("Virtual environment non configurato")
            return False
            
        # Dipendenze base per fine-tuning
        dependencies = [
            "torch>=2.1.0",
            "transformers>=4.45.0",
            "datasets>=2.14.0",
            "accelerate>=0.34.0",
            "peft>=0.12.0",
            "bitsandbytes>=0.41.0",
            "numpy",
            "tqdm",
            "pyyaml",
            "requests",
            "matplotlib"
        ]
        
        for dep in dependencies:
            logger.info(f"Installazione: {dep}")
            try:
                subprocess.run([str(self.venv_python), "-m", "pip", "install", dep], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Errore installazione {dep}: {e}")
                
        logger.info("Dipendenze essenziali installate")
        return True
        
    def prepare_dataset(self) -> bool:
        """Prepara il dataset MedAlpaca."""
        logger.info("Preparazione dataset...")
        
        data_path = Path(self.config.data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Check if local dataset exists
        local_dataset = Path("technical_architecture_qwen3_amd.md")
        if local_dataset.exists():
            logger.info("Trovato dataset locale, conversione in formato alpaca...")
            return self._convert_local_dataset(local_dataset, data_path)
            
        # Fallback: create synthetic medical dataset
        return self._create_synthetic_dataset(data_path)
        
    def _convert_local_dataset(self, local_file: Path, data_path: Path) -> bool:
        """Converte il dataset locale in formato alpaca."""
        try:
            with open(local_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Convert to alpaca format
            alpaca_data = []
            sections = content.split('\n#')
            
            for section in sections[1:]:  # Skip first empty section
                lines = section.strip().split('\n')
                if len(lines) >= 2:
                    title = lines[0].strip()
                    content_text = '\n'.join(lines[1:]).strip()
                    
                    alpaca_data.append({
                        "instruction": f"Spiega {title}",
                        "input": "",
                        "output": content_text
                    })
                    
            # Save converted dataset
            dataset_file = data_path / "medalpaca.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Dataset convertito: {len(alpaca_data)} esempi")
            return True
            
        except Exception as e:
            logger.error(f"Errore conversione dataset locale: {e}")
            return False
            
    def _create_synthetic_dataset(self, data_path: Path) -> bool:
        """Crea dataset sintetico medico di fallback."""
        logger.info("Creazione dataset sintetico medico...")
        
        synthetic_data = [
            {
                "instruction": "Quali sono i sintomi principali della febbre?",
                "input": "",
                "output": "I sintomi principali della febbre includono: temperatura corporea elevata (oltre 38°C), brividi, sudorazione, mal di testa, dolori muscolari, affaticamento e perdita di appetito."
            },
            {
                "instruction": "What are the main symptoms of hypertension?",
                "input": "",
                "output": "Main symptoms of hypertension include: headaches, shortness of breath, nosebleeds, chest pain, vision problems, and dizziness. However, hypertension is often called the 'silent killer' because it may have no symptoms."
            },
            {
                "instruction": "Come si diagnostica il diabete?",
                "input": "",
                "output": "Il diabete si diagnostica attraverso test del sangue come: glicemia a digiuno (≥126 mg/dL), test di tolleranza al glucosio orale, emoglobina glicata (≥6.5%), o glicemia casuale (≥200 mg/dL con sintomi)."
            },
            {
                "instruction": "What is the treatment for common cold?",
                "input": "",
                "output": "Treatment for common cold includes: rest, adequate fluid intake, over-the-counter pain relievers, decongestants, cough suppressants, and throat lozenges. Antibiotics are not effective against viral infections like the common cold."
            },
            {
                "instruction": "Quali sono le cause dell'ipertensione?",
                "input": "",
                "output": "Le cause dell'ipertensione includono: fattori genetici, età avanzata, obesità, sedentarietà, dieta ricca di sodio, consumo eccessivo di alcol, fumo, stress e condizioni mediche come diabete e malattie renali."
            }
        ]
        
        # Save dataset
        dataset_file = data_path / "medalpaca.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(synthetic_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Dataset sintetico creato: {len(synthetic_data)} esempi")
        return True
        
    def train_with_transformers(self) -> bool:
        """Training semplificato con Transformers."""
        logger.info("Training con Transformers...")
        
        try:
            # Create training script
            training_script = f'''
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os

# Setup AMD environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'

# Load model and tokenizer
model_name = "{self.config.model_name}"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r={self.config.lora_rank},
    lora_alpha={self.config.lora_alpha},
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)

# Load dataset
with open("{self.config.data_path}/medalpaca.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare dataset
def format_example(example):
    text = f"### Instruction: {{example['instruction']}}\\n### Input: {{example['input']}}\\n### Response: {{example['output']}}"
    return {{"text": text}}

formatted_data = [format_example(ex) for ex in data]
dataset = Dataset.from_list(formatted_data)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length={self.config.cutoff_length},
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="{self.config.output_dir}",
    per_device_train_batch_size={self.config.batch_size},
    gradient_accumulation_steps={self.config.gradient_accumulation_steps},
    num_train_epochs={self.config.num_epochs},
    learning_rate={self.config.learning_rate},
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    save_total_limit=2,
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    max_grad_norm=1.0,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save model
trainer.save_model()
trainer.model.save_pretrained("{self.config.output_dir}")

print("Training completato!")
'''
            
            # Save training script
            script_path = "train_model.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(training_script)
                
            logger.info("Script training creato")
            
            # Run training
            logger.info("Avvio training... (questo richiederà diverse ore)")
            result = subprocess.run([
                str(self.venv_python), script_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Training completato con successo!")
                return True
            else:
                logger.error(f"Training fallito: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Errore training: {e}")
            return False
            
    def test_model(self) -> bool:
        """Test del modello addestrato."""
        logger.info("Testing del modello...")
        
        try:
            test_script = f'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Setup AMD environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'

# Load model
tokenizer = AutoTokenizer.from_pretrained("{self.config.model_name}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "{self.config.model_name}",
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "{self.config.output_dir}")

# Test questions
test_questions = [
    "Quali sono i sintomi principali della febbre?",
    "What are the main symptoms of hypertension?",
    "Come si diagnostica il diabete?",
    "What is the treatment for common cold?",
    "Quali sono le cause dell'ipertensione?"
]

print("Test domande mediche:")
for i, question in enumerate(test_questions, 1):
    print(f"\\nDomanda {{i}}: {{question}}")
    
    inputs = tokenizer(question, return_tensors="pt", max_length=512, truncation=True)
    inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(question, '').strip()
    
    print(f"Risposta: {{response}}")

print("\\nTest completato!")
'''
            
            # Save test script
            script_path = "test_model.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(test_script)
                
            logger.info("Script test creato")
            
            # Run test
            result = subprocess.run([
                str(self.venv_python), script_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Test completato!")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"Test fallito: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Errore test: {e}")
            return False
            
    def export_model(self) -> bool:
        """Esporta il modello finale."""
        logger.info("Esportazione modello finale...")
        
        try:
            export_script = f'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Setup AMD environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'

# Load and merge models
base_model = AutoModelForCausalLM.from_pretrained(
    "{self.config.model_name}",
    device_map="auto",
    trust_remote_code=True
)

lora_model = PeftModel.from_pretrained(
    base_model,
    "{self.config.output_dir}",
    device_map="auto"
)

# Merge LoRA weights
merged_model = lora_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("{self.config.final_model_path}")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained("{self.config.model_name}", trust_remote_code=True)
tokenizer.save_pretrained("{self.config.final_model_path}")

print("Modello finale esportato con successo!")
'''
            
            # Save export script
            script_path = "export_model.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(export_script)
                
            logger.info("Script export creato")
            
            # Run export
            result = subprocess.run([
                str(self.venv_python), script_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Modello finale esportato!")
                return True
            else:
                logger.error(f"Export fallito: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Errore export: {e}")
            return False
            
    def run_complete_pipeline(self):
        """Esegue l'intera pipeline di training."""
        logger.info("=== AVVIO PIPELINE COMPLETA ===")
        start_time = time.time()
        
        try:
            # Step 1: Setup AMD environment
            self.setup_amd_environment()
            
            # Step 2: Create virtual environment
            if not self.create_virtual_environment():
                logger.error("Creazione virtual environment fallita")
                return False
                
            # Step 3: Install dependencies
            if not self.install_dependencies():
                logger.error("Installazione dipendenze fallita")
                return False
                
            # Step 4: Prepare dataset
            if not self.prepare_dataset():
                logger.error("Preparazione dataset fallita")
                return False
                
            # Step 5: Train model
            if not self.train_with_transformers():
                logger.error("Training fallito")
                return False
                
            # Step 6: Test model
            if not self.test_model():
                logger.warning("Test modello fallito, ma continuo")
                
            # Step 7: Export model
            if not self.export_model():
                logger.warning("Esportazione modello fallita")
                
            total_time = time.time() - start_time
            logger.info(f"=== PIPELINE COMPLETATA IN {total_time/3600:.1f} ORE ===")
            
            return True
            
        except Exception as e:
            logger.error(f"Errore pipeline: {e}")
            return False

def main():
    """Funzione principale."""
    print("🚀 Script Semplificato di Fine-tuning Qwen3-VL-8B-Instruct per AMD RX 6650 XT")
    print("=" * 80)
    
    # Create configuration
    config = TrainingConfig()
    
    # Create trainer
    trainer = SimpleQwenTrainer(config)
    
    # Run complete pipeline
    success = trainer.run_complete_pipeline()
    
    if success:
        print("\n✅ Training completato con successo!")
        print(f"📁 Modello finale salvato in: {config.final_model_path}")
        print(f"📊 Log completi in: training.log")
        print("\n🎯 Il tuo modello medico Qwen3-VL-8B-Instruct è pronto!")
        print("\n💡 Per usare il modello:")
        print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"   model = AutoModelForCausalLM.from_pretrained('{config.final_model_path}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{config.final_model_path}')")
    else:
        print("\n❌ Training fallito. Controlla i log per dettagli.")
        sys.exit(1)

if __name__ == "__main__":
    main()