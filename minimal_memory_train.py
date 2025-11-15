#!/usr/bin/env python3
"""
Script per training con utilizzo di memoria minimale
Usa direttamente le API di LLaMA Factory con ottimizzazioni estreme
"""
import os
import sys
import logging
import json
import subprocess
import gc
import torch
from pathlib import Path

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Percorso LLaMA Factory
LLAMA_FACTORY_PATH = "./llama_factory_data/LLaMA-Factory"

def clear_gpu_memory():
    """Pulisce la memoria GPU prima del caricamento del modello."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def setup_llama_factory():
    """Clona e installa LLaMA Factory."""
    logger.info("Setup LLaMA Factory...")
    if not Path(LLAMA_FACTORY_PATH).exists():
        logger.info("Clonazione LLaMA Factory...")
        subprocess.run(["git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git", LLAMA_FACTORY_PATH], check=True)
    
    # Aggiungi il percorso di LLaMA Factory a sys.path per assicurare l'import
    sys.path.insert(0, LLAMA_FACTORY_PATH)
    
    logger.info("Installazione LLaMA Factory (modalità editable)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", LLAMA_FACTORY_PATH], check=True)
    logger.info("LLaMA Factory installato con successo.")

def prepare_dataset():
    """Prepara il dataset per Colab."""
    logger.info("Preparazione dataset per Colab...")
    
    # Crea directory
    data_path = Path("./llama_factory_data/data")
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Controlla se esiste il file tecnico
    local_dataset = Path("technical_architecture_qwen3_amd.md")
    if local_dataset.exists():
        logger.info("Trovato dataset locale, conversione in formato alpaca...")
        
        with open(local_dataset, 'r', encoding='utf-8') as f:
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
                
        # Salva solo un sottoinsieme per ridurre la memoria
        alpaca_data = alpaca_data[:1] # Solo 1 esempio per risparmiare memoria
        
        # Save converted dataset
        dataset_file = data_path / "medalpaca.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
            
        # Update dataset_info.json
        dataset_info = {
            "medalpaca": {
                "file_name": "medalpaca.json",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                }
            }
        }
        
        info_file = data_path / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
            
        logger.info(f"Dataset convertito: {len(alpaca_data)} esempi")
        return True
    
    logger.error("Nessun dataset trovato")
    return False

def train_with_minimal_memory():
    """Esegue il training con impostazioni di memoria minimale."""
    logger.info("Avvio training con utilizzo di memoria minimale...")
    
    # Importa direttamente le funzioni necessarie da LLaMA Factory
    sys.path.insert(0, os.path.join(LLAMA_FACTORY_PATH, "src"))
    
    # Usa un comando CLI con opzioni specifiche per la memoria
    cmd = [
        sys.executable, 
        "-c",
        f"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from llamafactory.train.tuner import run_exp
import sys
import yaml

# Configurazione estrema per la memoria
config = {{
    "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "q_proj,v_proj",
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "dataset": "medalpaca",
    "template": "qwen3_vl_nothink",
    "cutoff_len": 16,
    "max_samples": 1,
    "overwrite_cache": True,
    "preprocessing_num_workers": 0,
    "output_dir": "./llama_factory_data/medical_qwen3_output_minimal",
    "logging_steps": 20,
    "save_steps": 20,
    "eval_steps": 20,
    "load_best_model_at_end": False,
    "save_strategy": "steps",
    "learning_rate": 1e-3,
    "num_train_epochs": 1,
    "max_grad_norm": 0.1,
    "save_total_limit": 1,
    "save_on_each_node": True,
    "fp16": False,
    "bf16": False,
    "remove_unused_columns": True,
    "weight_decay": 0.01,
    "warmup_ratio": 0.01,
    "lr_scheduler_type": "linear",
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 128,  # Aumentato ulteriormente
    "gradient_checkpointing": True,
    "tf32": False,
    "report_to": [],
    "run_name": "medical_qwen3_minimal",
    "plot_loss": False,
    "trust_remote_code": True,
    "quantization_bit": 8, # 8-bit quantization
    "quantization_method": "bitsandbytes",
    "use_unsloth": True,
    "flash_attn": "fa2",
    "ddp_timeout": 18000,
    "dataloader_num_workers": 0,
    "dataloader_pin_memory": False,
    "optim": "adamw_8bit",
    "dataset_dir": "./llama_factory_data/data",
    "torch_empty_cache_steps": 1,
    "ddp_find_unused_parameters": False,
    "upcast_layernorm": True,
    "low_cpu_mem_usage": True,
}}

# Salva la configurazione in un file temporaneo
with open('temp_config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

# Imposta il percorso per LLaMA Factory
import sys
sys.path.insert(0, '{LLAMA_FACTORY_PATH}/src')

from llamafactory.train.tuner import run_exp
from llamafactory.hparams.parser import get_train_args
from transformers import HfArgumentParser
from llamafactory.hparams import ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments

# Esegui direttamente l'esperienza di training
try:
    run_exp(config)
    print("Training completato con successo")
except Exception as e:
    print(f"Errore durante il training: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    ]
    
    # Imposta la variabile di ambiente per la gestione della memoria
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['PYTHONPATH'] = LLAMA_FACTORY_PATH + ':' + env.get('PYTHONPATH', '')
    
    try:
        # Esegui il comando
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Training completato con successo")
            return True
        else:
            logger.error(f"Training fallito: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione del training: {e}")
        return False

def main():
    """Funzione principale."""
    logger.info("=== AVVIO TRAINING QWEN3 CON UTILIZZO MINIMALE DI MEMORIA ===")
    
    try:
        # Step 1: Setup LLaMA Factory
        setup_llama_factory()
        
        # Step 2: Pulizia memoria GPU
        logger.info("Pulizia memoria GPU...")
        clear_gpu_memory()
        
        # Step 3: Prepara dataset
        if not prepare_dataset():
            logger.error("Preparazione dataset fallita")
            return False
        
        # Step 4: Pulizia memoria GPU prima del training
        logger.info("Pulizia memoria GPU prima del training...")
        clear_gpu_memory()
        
        # Step 5: Avvia training con impostazioni estreme per la memoria
        success = train_with_minimal_memory()
        
        if success:
            logger.info("=== TRAINING COMPLETATO CON SUCCESSO ===")
            logger.info("Modello salvato in ./llama_factory_data/medical_qwen3_output_minimal")
            return True
        else:
            logger.error("Training fallito")
            return False
            
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Training completato con successo!")
        print("📁 Modello salvato in: ./llama_factory_data/medical_qwen3_output_minimal")
    else:
        print("\n❌ Training fallito. Controlla i log per dettagli.")
        sys.exit(1)