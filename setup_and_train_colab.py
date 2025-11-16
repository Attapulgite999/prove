#!/usr/bin/env python3
"""
Script semplificato per training Qwen su Colab per applicazioni mediche
Versione ottimizzata per Google Colab con GPU
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
import time
from functools import wraps
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import threading

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Percorso LLaMA Factory
LLAMA_FACTORY_PATH = "./llama_factory_data/LLaMA-Factory"

def retry(exceptions, tries=4, delay=3, backoff=2, logger=None):
    """
    Decorator per ritentare una funzione in caso di eccezione.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = f"{str(e)}, ritento in {mdelay} secondi..."
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

@retry(exceptions=(OSError, RuntimeError), tries=3, delay=5, backoff=2, logger=logger)
def load_model_and_tokenizer(model_name):
    """Carica modello e tokenizer con gestione degli errori."""
    logger.info(f"Tentativo di caricamento di: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    logger.info(f"Caricato con successo: {model_name}")
    return model, tokenizer

def clear_gpu_memory():
    """Pulisce la memoria GPU prima del caricamento del modello."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def setup_llama_factory():
    """Clona e installa LLaMA Factory."""
    logger.info("Setup LLaMA Factory...")

    # Verifica se la directory esiste e contiene i file necessari per l'installazione
    llama_factory_dir = Path(LLAMA_FACTORY_PATH)
    setup_py = llama_factory_dir / "setup.py"
    pyproject_toml = llama_factory_dir / "pyproject.toml"

    if not llama_factory_dir.exists() or not (setup_py.exists() or pyproject_toml.exists()):
        logger.info("LLaMA Factory non trovato o incompleto. Clonazione da GitHub...")
        if llama_factory_dir.exists():
            logger.info("Rimozione directory esistente incompleta...")
            import shutil
            shutil.rmtree(llama_factory_dir)
        subprocess.run(["git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git", LLAMA_FACTORY_PATH], check=True)
    else:
        logger.info("LLaMA Factory già presente e valido.")

    sys.path.insert(0, str(llama_factory_dir.resolve()))

    logger.info("Installazione LLaMA Factory (modalità editable)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", LLAMA_FACTORY_PATH], check=True)

    import importlib.metadata
    try:
        importlib.metadata.version("unsloth")
        logger.info("unsloth rilevato.")
    except importlib.metadata.PackageNotFoundError:
        logger.info("Installazione unsloth...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "unsloth"], check=True)

    logger.info("LLaMA Factory installato con successo.")

def ensure_dependencies():
    """Installa le dipendenze necessarie."""
    subprocess.run([sys.executable, "-m", "pip", "install", "datasets>=2.16.0,<=4.0.0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "trl>=0.8.6,<=0.9.6"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "bitsandbytes"], check=True)
    try:
        import pynvml
    except ImportError:
        logger.info("Installazione pynvml per monitoraggio GPU...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pynvml"], check=True)

def keep_alive(stop_event):
    """Stampa un messaggio periodicamente per mantenere attiva la sessione Colab."""
    while not stop_event.is_set():
        logger.info("Keep-alive: la sessione è attiva...")
        time.sleep(600) # 10 minuti

def print_gpu_usage():
    """Stampa l'utilizzo attuale della GPU."""
    if not torch.cuda.is_available():
        return
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        logger.info(f"GPU Memory: {info.used // 1024**2}MB / {info.total // 1024**2}MB ({info.used/info.total*100:.2f}%) | GPU Utilization: {util.gpu}%")
        pynvml.nvmlShutdown()
    except Exception:
        pass

class ResourceMonitorCallback(TrainerCallback):
    """Un callback per monitorare le risorse durante il training."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % args.logging_steps == 0:
            print_gpu_usage()

def create_training_config():
    """Crea configurazione training ottimizzata."""
    config = {
        "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "q_proj,v_proj",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "dataset": "medalpaca",
        "template": "qwen2_5",
        "tokenizer_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "cutoff_len": 256,
        "max_samples": 500,
        "overwrite_cache": True,
        "preprocessing_num_workers": 2,
        "output_dir": "/content/drive/MyDrive/colab_training/medical_qwen_output",
        "logging_steps": 10,
        "save_steps": 50,
        "eval_steps": 50,
        "load_best_model_at_end": True,
        "save_strategy": "steps",
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "max_grad_norm": 1.0,
        "save_total_limit": 2,
        "fp16": True,
        "bf16": False,
        "remove_unused_columns": True,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "gradient_checkpointing": True,
        "report_to": [],
        "run_name": "medical_qwen_2.5_lora",
        "plot_loss": False,
        "trust_remote_code": True,
        "quantization_bit": 4,
        "use_unsloth": False,
        "dataset_dir": "./llama_factory_data/data",
    }

    config_path = "training_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Configurazione training salvata in {config_path}")
    return config_path

def prepare_dataset():
    """Prepara il dataset per Colab, scaricandolo se non esiste."""
    logger.info("Preparazione dataset per Colab...")
    data_path = Path("./llama_factory_data/data")
    data_path.mkdir(parents=True, exist_ok=True)
    dataset_file = data_path / "medalpaca.json"
    info_file = data_path / "dataset_info.json"

    if dataset_file.exists():
        logger.info("Dataset medalpaca.json trovato.")
        return True

    try:
        logger.info("Dataset non trovato localmente. Tentativo di download da Hugging Face...")
        from datasets import load_dataset

        # Scarica un dataset medico di esempio
        dataset = load_dataset("medalpaca/medical_meadow_medqa")

        # Prendi un sottoinsieme e formatta come alpaca
        alpaca_data = []
        for item in dataset["train"].select(range(500)): # Usiamo 500 esempi
            alpaca_data.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"]
            })

        # Salva il dataset convertito
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

        # Crea il file dataset_info.json
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
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)

        logger.info(f"Dataset scaricato e preparato con {len(alpaca_data)} esempi.")
        return True

    except Exception as e:
        logger.error(f"Impossibile scaricare o preparare il dataset: {e}", exc_info=True)
        return False

def check_model_availability(model_name):
    """Verifica la disponibilità del modello prima del training."""
    logger.info(f"Verifica disponibilità del modello: {model_name}")
    try:
        # Prova a scaricare solo il tokenizer per verificare la connessione
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("Modello disponibile e accessibile.")
        return True
    except Exception as e:
        logger.error(f"Modello non disponibile: {e}")
        return False

def run_training_with_retry(config_path, output_dir, max_retries=3):
    """Esegue il training con retry in caso di errori di caricamento tokenizer."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Tentativo di training {attempt + 1}/{max_retries}")

            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(LLAMA_FACTORY_PATH).resolve()) + os.pathsep + env.get('PYTHONPATH', '')
            env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            env['TOKENIZERS_PARALLELISM'] = 'false'
            env['HF_DATASETS_DOWNLOAD_NUM_WORKERS'] = '1'
            env['HF_DATASETS_DOWNLOAD_NUM_PROC'] = '1'

            train_cmd = [
                sys.executable, "-m", "llamafactory.cli", "train", config_path
            ]

            latest_checkpoint = None
            if os.path.isdir(output_dir):
                checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda d: int(d.split("-")[-1]))
                    latest_checkpoint = os.path.join(output_dir, latest_checkpoint)

            if latest_checkpoint:
                logger.info(f"Trovato checkpoint, riprendo da: {latest_checkpoint}")
                train_cmd.extend(["--resume_from_checkpoint", latest_checkpoint])

            result = subprocess.run(train_cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Training completato con successo!")
                return True
            else:
                logger.error(f"Training fallito (tentativo {attempt + 1}): {result.stderr}")
                if attempt < max_retries - 1:
                    logger.info("Ritento tra 30 secondi...")
                    time.sleep(30)
                else:
                    return False

        except Exception as e:
            logger.error(f"Errore durante il tentativo {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info("Ritento tra 30 secondi...")
                time.sleep(30)
            else:
                return False

    return False

def merge_lora_adapters(model_path, output_dir):
    """Unisce gli adapter LoRA al modello base per la conversione."""
    logger.info("Unione degli adapter LoRA al modello base...")
    export_dir = os.path.join(output_dir, "merged")
    try:
        export_config = {
            "model_name_or_path": model_path,
            "adapter_name_or_path": output_dir,
            "export_dir": export_dir,
            "export_size": 2,
            "export_device": "cpu",
            "export_legacy_format": False,
        }

        export_config_path = "export_config.yaml"
        with open(export_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(export_config, f)

        subprocess.run([
            sys.executable, "-m", "llamafactory.cli", "export", export_config_path
        ], check=True)

        logger.info(f"Adapter LoRA uniti con successo in: {export_dir}")
        return export_dir

    except Exception as e:
        logger.error(f"Errore durante l'unione degli adapter LoRA: {e}")
        return output_dir

def convert_to_gguf(model_path, output_dir):
    """Converte il modello finale in formato GGUF per LM Studio."""
    logger.info("Conversione del modello in formato GGUF per LM Studio...")

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], check=True)

        gguf_output_dir = os.path.join(os.path.dirname(output_dir), "gguf_model")
        Path(gguf_output_dir).mkdir(parents=True, exist_ok=True)
        gguf_file_path = os.path.join(gguf_output_dir, "model.gguf")

        # Usa lo script di conversione di llama.cpp
        # Assicurati che llama.cpp sia installato e nel path
        # In Colab, questo di solito funziona dopo pip install
        # Cloniamo llama.cpp per essere sicuri
        if not Path("llama.cpp").exists():
             subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)

        convert_script = Path("llama.cpp/convert-hf-to-gguf.py")

        cmd = [
            sys.executable, str(convert_script), model_path,
            "--outfile", gguf_file_path,
            "--outtype", "q4_k_m"
        ]
        subprocess.run(cmd, check=True)

        logger.info(f"Modello convertito in GGUF con successo! File disponibile in: {gguf_file_path}")
        return True

    except Exception as e:
        logger.error(f"Errore durante la conversione in GGUF: {e}")
        return False

def main():
    """Funzione principale per Colab."""
    logger.info("=== AVVIO TRAINING QWEN PER APPLICAZIONI MEDICHE SU COLAB ===")

    stop_keep_alive = threading.Event()
    keep_alive_thread = threading.Thread(target=keep_alive, args=(stop_keep_alive,))
    keep_alive_thread.daemon = True
    keep_alive_thread.start()

    try:
        # Step 1: Setup
        setup_llama_factory()
        ensure_dependencies()

        # Step 2: Carica configurazione
        config_path = create_training_config()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_name = config["model_name_or_path"]
        output_dir = config["output_dir"]

        # Step 3: Verifica modello e prepara dataset
        if not check_model_availability(model_name):
            logger.error("Impossibile accedere al modello. Controlla la connessione internet e riprova.")
            return False

        clear_gpu_memory()
        if not prepare_dataset():
            return False

        # Step 4: Avvia training con retry
        logger.info(f"Avvio training per il modello: {model_name}...")

        if run_training_with_retry(config_path, output_dir, max_retries=3):
            logger.info("=== TRAINING COMPLETATO CON SUCCESSO ===")

            # Step 5: Unisci adapter e converti in GGUF
            merged_path = merge_lora_adapters(model_name, output_dir)
            convert_to_gguf(merged_path, output_dir)

            return True
        else:
            logger.error("Training fallito dopo tutti i tentativi.")
            return False

    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {e}", exc_info=True)
        return False
    finally:
        stop_keep_alive.set()
        logger.info("Thread keep-alive terminato.")

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Processo completato con successo!")
    else:
        print("\n❌ Processo fallito. Controlla i log per dettagli.")
        sys.exit(1)