#!/usr/bin/env python3
"""
Script altamente ottimizzato per il training Qwen3 su Colab
Versione con ottimizzazioni estreme per la memoria
"""
import os
import sys
import logging
import yaml
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

def create_training_config():
    """Crea configurazione training altamente ottimizzata per la memoria."""
    config = {
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "q_proj,v_proj",  # Minimi parametri LoRA
        "lora_rank": 4,  # Ridotto ulteriormente
        "lora_alpha": 8,  # Aggiustato di conseguenza
        "lora_dropout": 0.1,
        "dataset": "medalpaca",
        "template": "qwen3_vl_nothink",
        "cutoff_len": 16,  # Ridotto al minimo
        "max_samples": 1,  # Solo 1 esempio per test
        "overwrite_cache": True,
        "preprocessing_num_workers": 0,
        "output_dir": "/content/drive/MyDrive/colab_training/medical_qwen3_output",
        "logging_steps": 10,  # Ridotto per meno registrazione
        "save_steps": 20,
        "eval_steps": 20,
        "load_best_model_at_end": False,
        "save_strategy": "steps",
        "learning_rate": 1e-3,  # Aumentato per convergenza rapida
        "num_train_epochs": 1,
        "max_grad_norm": 0.1,  # Ridotto drasticamente
        "save_total_limit": 1,
        "save_on_each_node": True,
        "fp16": False,  # Disabilitato per risparmiare memoria
        "bf16": False,  # Disabilitato per risparmiare memoria
        "remove_unused_columns": True,
        "weight_decay": 0.01,
        "warmup_ratio": 0.01,  # Ridotto
        "lr_scheduler_type": "linear",
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 64,  # Aumentato drasticamente
        "gradient_checkpointing": True,
        "tf32": False,
        "report_to": [],
        "run_name": "medical_qwen3_lora_extreme",
        "plot_loss": False,
        "trust_remote_code": True,
        "quantization_bit": 8,  # Aumentato da 4 a 8 per migliore stabilità
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
    }
    
    config_path = "training_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
    logger.info(f"Configurazione training salvata in {config_path}")
    return config_path

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

def convert_to_gguf():
    """Converte il modello finale in formato GGUF per LM Studio."""
    logger.info("Conversione del modello in formato GGUF per LM Studio...")
    
    try:
        # Installa llama-cpp-python se non presente
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], check=True)
        
        # Scarica lo script di conversione da LLaMA-Factory
        convert_script = Path("./convert_llama_to_gguf.py")
        if not convert_script.exists():
            # Creiamo uno script di conversione personalizzato
            with open(convert_script, 'w', encoding='utf-8') as f:
                f.write("""#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse

def convert_to_gguf(model_path, output_path, quantization_type="Q4_K_M"):
    \"\"\"Converte un modello Hugging Face in formato GGUF.\"\"\"
    print(f"Caricamento modello da: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Conversione in formato GGUF con quantizzazione {quantization_type}...")
    # Usa llama.cpp per la conversione
    import subprocess
    cmd = [
        "python", "-m", "llama_cpp", "convert", 
        "--outfile", f"{output_path}/model.gguf",
        "--outtype", quantization_type,
        model_path
    ]
    subprocess.run(cmd)
    
    print(f"Modello convertito salvato in: {output_path}/model.gguf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--quantization", type=str, default="Q4_K_M", help="Quantization type")
    args = parser.parse_args()
    
    convert_to_gguf(args.model_path, args.output_path, args.quantization)
""")
        
        # Esegui la conversione
        model_path = "/content/drive/MyDrive/colab_training/medical_qwen3_output"
        output_path = "/content/drive/MyDrive/colab_training/medical_qwen3_gguf"
        
        # Crea la directory di output se non esiste
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Esegui lo script di conversione
        subprocess.run([
            sys.executable, 
            str(convert_script),
            "--model_path", model_path,
            "--output_path", output_path
        ], check=True)
        
        logger.info("Modello convertito in formato GGUF con successo!")
        logger.info(f"File GGUF disponibile in: {output_path}/model.gguf")
        return True
        
    except Exception as e:
        logger.error(f"Errore durante la conversione in GGUF: {e}")
        logger.info("Puoi comunque usare il modello in formato Hugging Face con LM Studio")
        return False

def main():
    """Funzione principale per Colab."""
    logger.info("=== AVVIO TRAINING QWEN3 SU COLAB (VERSIONE ESTREMA) ===")
    
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
            
        # Step 4: Crea configurazione training
        config_path = create_training_config()
        
        # Step 5: Pulizia memoria GPU prima del training
        logger.info("Pulizia memoria GPU prima del training...")
        clear_gpu_memory()
        
        # Step 6: Avvia training
        logger.info("Avvio training con LLaMA Factory...")
        logger.info("Questo potrebbe richiedere 10-30 minuti con GPU...")
        
        # Imposta la variabile di ambiente per la gestione della memoria
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['PYTHONPATH'] = LLAMA_FACTORY_PATH + ':' + env.get('PYTHONPATH', '')
        
        # Esegui training utilizzando il modulo Python direttamente
        exit_code = subprocess.run([
            sys.executable, 
            "-m", 
            "llamafactory.cli", 
            "train", 
            config_path
        ], env=env).returncode
        
        if exit_code == 0:
            logger.info("=== TRAINING COMPLETATO CON SUCCESSO ===")
            logger.info("Il modello è stato salvato in /content/drive/MyDrive/colab_training/medical_qwen3_output")
            
            # Step 7: Conversione in GGUF per LM Studio
            logger.info("Avvio conversione in formato GGUF per LM Studio...")
            convert_success = convert_to_gguf()
            
            if convert_success:
                logger.info("=== MODELLO PRONTO PER LM STUDIO ===")
                logger.info("File GGUF disponibile in: /content/drive/MyDrive/colab_training/medical_qwen3_gguf/model.gguf")
            else:
                logger.info("Conversione fallita, ma il modello in formato Hugging Face è comunque utilizzabile con LM Studio")
                
            return True
        else:
            logger.error(f"Training fallito con codice di uscita: {exit_code}")
            return False
            
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Training completato con successo!")
        print("📁 Modello salvato in: /content/drive/MyDrive/colab_training/medical_qwen3_output")
        print("📊 Log disponibili in: training.log")
        print("🎯 Modello convertito in GGUF per LM Studio (se la conversione è riuscita)")
    else:
        print("\n❌ Training fallito. Controlla i log per dettagli.")
        sys.exit(1)