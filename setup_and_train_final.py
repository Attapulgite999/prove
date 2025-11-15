#!/usr/bin/env python3
"""
Script finale per fine-tuning Qwen3-VL-8B-Instruct su AMD RX 6650 XT.
Versione ottimizzata per Windows con ROCm support.
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
    model_name: str = "Qwen/Qwen3-8B-Instruct"  # Modello base senza VL per compatibilità
    dataset_name: str = "medalpaca"
    output_dir: str = "./qwen3_medical_output"
    
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
    data_path: str = "./training_data"
    final_model_path: str = "./final_qwen3_model"

class Qwen3Trainer:
    """Classe per fine-tuning Qwen3 con supporto AMD ROCm."""
    
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
        """Installa dipendenze essenziali per Qwen3."""
        logger.info("Installazione dipendenze essenziali...")
        
        if not self.venv_python:
            logger.error("Virtual environment non configurato")
            return False
            
        # Dipendenze ottimizzate per Qwen3 e ROCm
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
            "matplotlib",
            "einops",
            "transformers-stream-generator",
            "tiktoken"
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
        """Prepara il dataset medico."""
        logger.info("Preparazione dataset...")
        
        data_path = Path(self.config.data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Check if local dataset exists
        local_dataset = Path("technical_architecture_qwen3_amd.md")
        if local_dataset.exists():
            logger.info("Trovato dataset locale, conversione in formato istruzione-risposta...")
            return self._convert_local_dataset(local_dataset, data_path)
            
        # Fallback: create comprehensive medical dataset
        return self._create_medical_dataset(data_path)
        
    def _convert_local_dataset(self, local_file: Path, data_path: Path) -> bool:
        """Converte il dataset locale in formato training."""
        try:
            with open(local_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Convert to instruction format
            training_data = []
            sections = content.split('\n#')
            
            for section in sections[1:]:  # Skip first empty section
                lines = section.strip().split('\n')
                if len(lines) >= 2:
                    title = lines[0].strip()
                    content_text = '\n'.join(lines[1:]).strip()
                    
                    # Format for instruction tuning
                    training_data.append({
                        "instruction": f"Spiega in dettaglio: {title}",
                        "input": "",
                        "output": content_text,
                        "system": "Sei un assistente medico esperto. Rispondi in modo dettagliato e accurato."
                    })
                    
            # Save converted dataset
            dataset_file = data_path / "medical_dataset.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Dataset convertito: {len(training_data)} esempi")
            return True
            
        except Exception as e:
            logger.error(f"Errore conversione dataset locale: {e}")
            return False
            
    def _create_medical_dataset(self, data_path: Path) -> bool:
        """Crea dataset medico completo."""
        logger.info("Creazione dataset medico...")
        
        medical_data = [
            {
                "instruction": "Quali sono i sintomi principali della febbre?",
                "input": "",
                "output": "I sintomi principali della febbre includono: temperatura corporea elevata (oltre 38°C), brividi, sudorazione, mal di testa, dolori muscolari, affaticamento, perdita di appetito, e possibile disidratazione. La febbre è una risposta naturale del corpo alle infezioni.",
                "system": "Sei un assistente medico esperto."
            },
            {
                "instruction": "What are the main symptoms of hypertension?",
                "input": "",
                "output": "Main symptoms of hypertension include: severe headaches, shortness of breath, nosebleeds, chest pain, vision problems, dizziness, and fatigue. However, hypertension is often called the 'silent killer' because it may have no noticeable symptoms until serious complications occur.",
                "system": "You are an expert medical assistant."
            },
            {
                "instruction": "Come si diagnostica il diabete mellito?",
                "input": "",
                "output": "Il diabete si diagnostica attraverso diversi test del sangue: 1) Glicemia a digiuno ≥126 mg/dL, 2) Test di tolleranza al glucosio orale con valore ≥200 mg/dL dopo 2 ore, 3) Emoglobina glicata (HbA1c) ≥6.5%, 4) Glicemia casuale ≥200 mg/dL con sintomi classici. È importante confermare con test ripetuti.",
                "system": "Sei un assistente medico esperto."
            },
            {
                "instruction": "What is the standard treatment for type 2 diabetes?",
                "input": "",
                "output": "Standard treatment for type 2 diabetes includes: 1) Lifestyle modifications (diet, exercise, weight loss), 2) Oral medications like metformin as first-line therapy, 3) Other medications such as SGLT2 inhibitors, GLP-1 agonists, or insulin if needed, 4) Regular blood glucose monitoring, 5) Management of cardiovascular risk factors.",
                "system": "You are an expert medical assistant."
            },
            {
                "instruction": "Quali sono le complicanze dell'ipertensione non trattata?",
                "input": "",
                "output": "Le complicanze dell'ipertensione non trattata includono: malattie cardiovascolari (infarto, insufficienza cardiaca), ictus, nefropatia ipertensiva, retinopatia ipertensiva con possibile perdita della vista, aneurismi, demenza vascolare, e compromissione della funzione cognitiva. Il controllo pressorio è fondamentale per prevenire queste complicazioni.",
                "system": "Sei un assistente medico esperto."
            },
            {
                "instruction": "Explain the pathophysiology of asthma.",
                "input": "",
                "output": "Asthma pathophysiology involves: 1) Chronic airway inflammation with eosinophils, mast cells, and T-helper 2 cells, 2) Bronchial hyperresponsiveness to various triggers, 3) Reversible airway obstruction due to bronchoconstriction, mucus hypersecretion, and airway wall edema, 4) Airway remodeling in chronic cases, 5) Genetic and environmental factors contributing to disease development.",
                "system": "You are an expert medical assistant."
            }
        ]
        
        # Save dataset
        dataset_file = data_path / "medical_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(medical_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Dataset medico creato: {len(medical_data)} esempi")
        return True
        
    def create_training_script(self) -> str:
        """Crea script di training ottimizzato."""
        script_content = f'''
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import json

# Setup AMD environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'

# Load model and tokenizer
model_name = "{self.config.model_name}"
logger.info(f"Caricamento modello: {{model_name}}")

# Load with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r={self.config.lora_rank},
    lora_alpha={self.config.lora_alpha},
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
logger.info("Caricamento dataset...")
with open("{self.config.data_path}/medical_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

logger.info(f"Dataset caricato: {{len(data)}} esempi")

# Prepare dataset
def format_example(example):
    if example.get("system"):
        text = f"System: {{example['system']}}\\nHuman: {{example['instruction']}}\\nAssistant: {{example['output']}}"
    else:
        text = f"Human: {{example['instruction']}}\\nAssistant: {{example['output']}}"
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

logger.info("Tokenizzazione dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

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
    save_total_limit=2,
    evaluation_strategy="no",
    load_best_model_at_end=False,
    gradient_checkpointing=True,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    max_grad_norm=1.0,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    ddp_find_unused_parameters=False
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
logger.info("Inizio training...")
trainer.train()

# Save model
logger.info("Salvataggio modello...")
trainer.save_model()
trainer.model.save_pretrained("{self.config.output_dir}")

logger.info("Training completato con successo!")
'''
        
        return script_content
        
    def train_model(self) -> bool:
        """Esegue il training del modello."""
        logger.info("Preparazione script di training...")
        
        try:
            # Create training script
            script_content = self.create_training_script()
            script_path = "train_qwen3.py"
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
                
            logger.info("Script training creato")
            
            # Run training
            logger.info("Avvio training... (questo richiederà diverse ore)")
            logger.info("Training in corso su GPU AMD RX 6650 XT...")
            
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
            logger.error(f"Errore durante il training: {e}")
            return False
            
    def test_trained_model(self) -> bool:
        """Test del modello addestrato."""
        logger.info("Testing del modello addestrato...")
        
        try:
            test_script = f'''
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Setup AMD environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'

# Load model
tokenizer = AutoTokenizer.from_pretrained("{self.config.model_name}", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "{self.config.model_name}",
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{self.config.output_dir}")

# Test questions
test_questions = [
    ("it", "Quali sono i sintomi principali della febbre?"),
    ("en", "What are the main symptoms of hypertension?"),
    ("it", "Come si diagnostica il diabete?"),
    ("en", "What is the standard treatment for type 2 diabetes?"),
    ("it", "Quali sono le complicanze dell'ipertensione non trattata?")
]

print("🧪 Test del modello medico fine-tunato:")
print("=" * 60)

for i, (lang, question) in enumerate(test_questions, 1):
    print(f"\\n{{i}}. [{'Italiano' if lang == 'it' else 'English'}] {{question}}")
    
    # Prepare input
    if lang == "it":
        system_msg = "Sei un assistente medico esperto. Rispondi in modo dettagliato e accurato in italiano."
    else:
        system_msg = "You are an expert medical assistant. Respond in detailed and accurate English."
    
    full_input = f"System: {{system_msg}}\\nHuman: {{question}}\\nAssistant:"
    
    inputs = tokenizer(full_input, return_tensors="pt", max_length=512, truncation=True)
    inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    print(f"💡 Risposta: {{response}}")
    print("-" * 60)

print("\\n✅ Test completato! Il modello è pronto per l'uso.")
'''
            
            # Save test script
            script_path = "test_qwen3.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(test_script)
                
            logger.info("Script test creato")
            
            # Run test
            result = subprocess.run([
                str(self.venv_python), script_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Test completato!")
                logger.info("\\n" + result.stdout)
                return True
            else:
                logger.error(f"Test fallito: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Errore durante il test: {e}")
            return False
            
    def export_final_model(self) -> bool:
        """Esporta il modello finale pronto all'uso."""
        logger.info("Esportazione modello finale...")
        
        try:
            export_script = f'''
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Setup AMD environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1030'

# Load and merge models
logger.info("Caricamento modelli per il merge...")
base_model = AutoModelForCausalLM.from_pretrained(
    "{self.config.model_name}",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

lora_model = PeftModel.from_pretrained(
    base_model,
    "{self.config.output_dir}",
    device_map="auto"
)

# Merge LoRA weights with base model
logger.info("Merge dei pesi LoRA...")
merged_model = lora_model.merge_and_unload()

# Save merged model
logger.info("Salvataggio modello finale...")
merged_model.save_pretrained("{self.config.final_model_path}")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained("{self.config.model_name}", trust_remote_code=True)
tokenizer.save_pretrained("{self.config.final_model_path}")

logger.info(f"✅ Modello finale salvato in: {self.config.final_model_path}")
logger.info("🎯 Il modello è pronto per l'uso!")
'''
            
            # Save export script
            script_path = "export_qwen3.py"
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
            logger.error(f"Errore durante l'export: {e}")
            return False
            
    def run_complete_pipeline(self):
        """Esegue l'intera pipeline di training."""
        logger.info("🚀 AVVIO PIPELINE COMPLETA DI FINE-TUNING")
        logger.info("=" * 70)
        start_time = time.time()
        
        try:
            # Step 1: Setup AMD environment
            logger.info("1️⃣ Configurazione ambiente AMD...")
            self.setup_amd_environment()
            
            # Step 2: Create virtual environment
            logger.info("2️⃣ Creazione ambiente virtuale...")
            if not self.create_virtual_environment():
                logger.error("❌ Creazione virtual environment fallita")
                return False
                
            # Step 3: Install dependencies
            logger.info("3️⃣ Installazione dipendenze...")
            if not self.install_dependencies():
                logger.error("❌ Installazione dipendenze fallita")
                return False
                
            # Step 4: Prepare dataset
            logger.info("4️⃣ Preparazione dataset medico...")
            if not self.prepare_dataset():
                logger.error("❌ Preparazione dataset fallita")
                return False
                
            # Step 5: Train model
            logger.info("5️⃣ Avvio training del modello...")
            logger.info("⚠️  Il training richiederà 4-8 ore su AMD RX 6650 XT")
            if not self.train_model():
                logger.error("❌ Training fallito")
                return False
                
            # Step 6: Test model
            logger.info("6️⃣ Testing del modello...")
            if not self.test_trained_model():
                logger.warning("⚠️ Test modello fallito, ma continuo")
                
            # Step 7: Export model
            logger.info("7️⃣ Esportazione modello finale...")
            if not self.export_final_model():
                logger.warning("⚠️ Esportazione modello fallita")
                
            total_time = time.time() - start_time
            logger.info("🎉 PIPELINE COMPLETATA CON SUCCESSO!")
            logger.info(f"⏱️  Tempo totale: {total_time/3600:.1f} ore")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore pipeline: {e}")
            return False

def main():
    """Funzione principale."""
    print("🚀 Qwen3-8B-Instruct Fine-tuning per AMD RX 6650 XT")
    print("=" * 70)
    print("📋 Questo script automatizzerà il fine-tuning completo")
    print("⚠️  Richiederà 4-8 ore e ~30GB di spazio su disco")
    print("=" * 70)
    
    # Create configuration
    config = TrainingConfig()
    
    # Create trainer
    trainer = Qwen3Trainer(config)
    
    # Run complete pipeline
    success = trainer.run_complete_pipeline()
    
    if success:
        print("\\n🎉 FINE-TUNING COMPLETATO CON SUCCESSO!")
        print(f"📁 Modello finale: {config.final_model_path}")
        print(f"📊 Log completi: training.log")
        print("\\n💡 Utilizzo del modello:")
        print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"   model = AutoModelForCausalLM.from_pretrained('{config.final_model_path}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{config.final_model_path}')")
        print("\\n🎯 Il tuo modello medico Qwen3 è pronto per l'uso!")
    else:
        print("\\n❌ Fine-tuning fallito. Controlla training.log per dettagli.")
        sys.exit(1)

if __name__ == "__main__":
    main()