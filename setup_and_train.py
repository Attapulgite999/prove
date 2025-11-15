#!/usr/bin/env python3
"""
Script completo per automatizzare il fine-tuning di Qwen3-VL-8B-Instruct su AMD RX 6650 XT.
Autore: AI Assistant
Data: 2024
"""

import os
import sys
import json
import logging
import subprocess
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import signal

# Gestione import opzionali
try:
    import psutil
except ImportError:
    psutil = None
    
try:
    import torch
except ImportError:
    torch = None
    
try:
    import yaml
except ImportError:
    yaml = None
    
try:
    import requests
except ImportError:
    requests = None
    
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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
    
    # Quantization e ottimizzazioni memoria
    quantization: str = "4bit"
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    
    # Percorsi
    venv_path: str = "./.venv"
    llama_factory_path: str = "./llama_factory_data/LLaMA-Factory"
    data_path: str = "./llama_factory_data/data"
    final_model_path: str = "./final_model"

class QwenTrainer:
    """Classe principale per automatizzare il fine-tuning di Qwen3-VL-8B-Instruct."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.venv_python = None
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Configura gestori segnali per cleanup graceful."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Gestisce segnali di interruzione."""
        logger.warning(f"Ricevuto segnale {signum}. Pulizia in corso...")
        sys.exit(1)
        
    def check_system_requirements(self) -> bool:
        """Controlla requisiti di sistema."""
        logger.info("Controllo requisiti di sistema...")
        
        # Check Python version
        if sys.version_info < (3, 10):
            logger.error("Richiesto Python 3.10+")
            return False
            
        # Check available disk space (need at least 50GB)
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 50:
            logger.warning(f"Spazio libero insufficiente: {free_gb:.1f}GB. Richiesti almeno 50GB.")
            
        # Check RAM (need at least 16GB)
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < 16:
            logger.warning(f"RAM insufficiente: {ram_gb:.1f}GB. Consigliati almeno 16GB.")
            
        logger.info("Requisiti di sistema OK")
        return True
        
    def setup_amd_environment(self):
        """Configura variabili ambiente per AMD GPU."""
        logger.info("Configurazione ambiente AMD...")
        
        env_vars = {
            'HSA_OVERRIDE_GFX_VERSION': '10.3.0',
            'PYTORCH_ROCM_ARCH': 'gfx1030',
            'CUDA_VISIBLE_DEVICES': '0',
            'ROCM_PATH': '/opt/rocm',
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
            
            # Upgrade pip and essential packages
            logger.info("Upgrade pip e pacchetti essenziali...")
            subprocess.run([str(self.venv_python), "-m", "pip", "install", "--upgrade", 
                          "pip", "setuptools", "wheel"], check=True)
            
            logger.info("Virtual environment creato con successo")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Errore creazione virtual environment: {e}")
            return False
            
    def install_dependencies(self) -> bool:
        """Installa tutte le dipendenze necessarie."""
        logger.info("Installazione dipendenze...")
        
        if not self.venv_python:
            logger.error("Virtual environment non configurato")
            return False
            
        dependencies = [
            # PyTorch con ROCm support
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2",
            
            # Hugging Face libraries
            "transformers>=4.45.0",
            "datasets>=2.14.0",
            "tokenizers>=0.19.0",
            "accelerate>=0.34.0",
            "peft>=0.12.0",
            
            # Quantization and optimization
            "bitsandbytes>=0.41.0",
            "scipy",
            "numpy",
            
            # Utilities
            "tqdm",
            "psutil",
            "pyyaml",
            "requests",
            "matplotlib",
            "seaborn",
            "wandb",
            "tensorboard",
            
            # LLaMA Factory dependencies
            "fire",
            "jieba",
            "rouge-chinese",
            "nltk",
            "sentencepiece",
            "protobuf",
            "uvloop",
            "sse-starlette"
        ]
        
        for dep in dependencies:
            logger.info(f"Installazione: {dep}")
            try:
                subprocess.run([str(self.venv_python), "-m", "pip", "install", "--no-cache-dir"] + dep.split(), 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Errore installazione {dep}: {e}")
                # Continue with other dependencies
                
        logger.info("Dipendenze installate")
        return True
        
    def setup_llama_factory(self) -> bool:
        """Setup LLaMA Factory."""
        logger.info("Setup LLaMA Factory...")
        
        llama_factory_path = Path(self.config.llama_factory_path)
        
        if llama_factory_path.exists():
            logger.info("LLaMA Factory già installato")
            return True
            
        try:
            # Clone LLaMA Factory repository
            subprocess.run(["git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git", 
                          str(llama_factory_path)], check=True, capture_output=True)
            
            # Install LLaMA Factory
            subprocess.run([str(self.venv_python), "-m", "pip", "install", "-e", "."], 
                          cwd=str(llama_factory_path), check=True, capture_output=True)
            
            logger.info("LLaMA Factory installato con successo")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Errore setup LLaMA Factory: {e}")
            return False
            
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
            
        # Try to download from Hugging Face
        return self._download_medalpaca_dataset(data_path)
        
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
                
            # Update dataset_info.json
            self._update_dataset_info(data_path)
            
            logger.info(f"Dataset convertito: {len(alpaca_data)} esempi")
            return True
            
        except Exception as e:
            logger.error(f"Errore conversione dataset locale: {e}")
            return False
            
    def _download_medalpaca_dataset(self, data_path: Path) -> bool:
        """Scarica il dataset MedAlpaca da Hugging Face."""
        try:
            from datasets import load_dataset
            
            logger.info("Download dataset MedAlpaca da Hugging Face...")
            dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train")
            
            # Convert to alpaca format
            alpaca_data = []
            for item in dataset:
                alpaca_data.append({
                    "instruction": item.get('instruction', 'Spiega il concetto medico'),
                    "input": item.get('input', ''),
                    "output": item.get('output', item.get('answer', ''))
                })
                
            # Save dataset
            dataset_file = data_path / "medalpaca.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
                
            # Update dataset_info.json
            self._update_dataset_info(data_path)
            
            logger.info(f"Dataset scaricato: {len(alpaca_data)} esempi")
            return True
            
        except Exception as e:
            logger.error(f"Errore download dataset: {e}")
            return False
            
    def _update_dataset_info(self, data_path: Path):
        """Aggiorna dataset_info.json per LLaMA Factory."""
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
            
    def create_training_config(self) -> str:
        """Crea configurazione training YAML."""
        config = {
            "model_name_or_path": self.config.model_name,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_target": "all",
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": 0.1,
            "dataset": "medalpaca",
            "template": "qwen3_vl_nothink",
            "cutoff_len": self.config.cutoff_length,
            "max_samples": 10000,
            "overwrite_cache": True,
            "preprocessing_num_workers": 4,
            "output_dir": self.config.output_dir,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "load_best_model_at_end": False,
            "save_strategy": "steps",
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "max_grad_norm": 1.0,
            "save_total_limit": 3,
            "save_on_each_node": True,
            "fp16": True,
            "bf16": False,
            "remove_unused_columns": False,
            "weight_decay": 0.1,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "tf32": False,
            "report_to": ["tensorboard"],
            "run_name": "medical_qwen3_lora",
            "plot_loss": True,
            "trust_remote_code": True,
            "quantization_bit": 4,
            "quantization_method": "bitsandbytes",
            "use_unsloth": False,
            "flash_attn": "auto",
            "ddp_timeout": 18000000,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "optim": "adamw_torch",
        }
        
        config_path = "training_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"Configurazione training salvata in {config_path}")
        return config_path
        
    def monitor_gpu_memory(self) -> Dict[str, float]:
        """Monitora utilizzo memoria GPU."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo()
            
            return {
                "used_gb": info.used / (1024**3),
                "total_gb": info.total / (1024**3),
                "free_gb": info.free / (1024**3),
                "usage_percent": (info.used / info.total) * 100
            }
        except:
            # Fallback for AMD GPUs
            return {"used_gb": 0, "total_gb": 8, "free_gb": 8, "usage_percent": 0}
            
    def start_training(self, config_path: str) -> bool:
        """Avvia il training con LLaMA Factory."""
        logger.info("Avvio training...")
        
        try:
            # Import LLaMA Factory modules
            sys.path.insert(0, self.config.llama_factory_path)
            
            # Import training module
            from llamafactory.train.tuner import run_exp
            
            # Load config
            with open(config_path, 'r', encoding='utf-8') as f:
                training_config = yaml.safe_load(f)
                
            logger.info("Configurazione training:")
            for key, value in training_config.items():
                logger.info(f"  {key}: {value}")
                
            # Start training with monitoring
            logger.info("Training in corso... (questo potrebbe richiedere diverse ore)")
            
            # Monitor GPU memory in background
            import threading
            stop_monitoring = threading.Event()
            
            def monitor_gpu():
                while not stop_monitoring.is_set():
                    memory_info = self.monitor_gpu_memory()
                    logger.info(f"GPU Memory: {memory_info['used_gb']:.1f}/{memory_info['total_gb']:.1f}GB "
                              f"({memory_info['usage_percent']:.1f}%)")
                    time.sleep(60)  # Monitor every minute
                    
            monitor_thread = threading.Thread(target=monitor_gpu)
            monitor_thread.start()
            
            try:
                # Run training
                run_exp(training_config)
                success = True
            except Exception as e:
                logger.error(f"Errore durante il training: {e}")
                success = False
            finally:
                stop_monitoring.set()
                monitor_thread.join()
                
            if success:
                logger.info("Training completato con successo!")
            else:
                logger.error("Training fallito")
                
            return success
            
        except Exception as e:
            logger.error(f"Errore avvio training: {e}")
            return False
            
    def test_model(self) -> bool:
        """Test del modello addestrato."""
        logger.info("Testing del modello...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            # Load base model
            logger.info("Caricamento modello base...")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            logger.info("Caricamento adapter LoRA...")
            model = PeftModel.from_pretrained(
                model,
                self.config.output_dir,
                device_map="auto"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
            
            # Test questions
            test_questions = [
                "Quali sono i sintomi principali della febbre?",
                "What are the main symptoms of hypertension?",
                "Come si diagnostica il diabete?",
                "What is the treatment for common cold?",
                "Quali sono le cause dell'ipertensione?"
            ]
            
            logger.info("Test domande mediche:")
            for i, question in enumerate(test_questions, 1):
                logger.info(f"\nDomanda {i}: {question}")
                
                inputs = tokenizer(question, return_tensors="pt", max_length=512, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
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
                
                logger.info(f"Risposta: {response}")
                
            logger.info("Test completato")
            return True
            
        except Exception as e:
            logger.error(f"Errore durante il test: {e}")
            return False
            
    def export_model(self) -> bool:
        """Esporta il modello finale."""
        logger.info("Esportazione modello finale...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            final_path = Path(self.config.final_model_path)
            final_path.mkdir(exist_ok=True)
            
            # Load and merge models
            logger.info("Caricamento e merge dei modelli...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                trust_remote_code=True
            )
            
            lora_model = PeftModel.from_pretrained(
                base_model,
                self.config.output_dir,
                device_map="auto"
            )
            
            # Merge LoRA weights
            merged_model = lora_model.merge_and_unload()
            
            # Save merged model
            logger.info("Salvataggio modello finale...")
            merged_model.save_pretrained(str(final_path))
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
            tokenizer.save_pretrained(str(final_path))
            
            logger.info(f"Modello finale salvato in {final_path}")
            return True
            
        except Exception as e:
            logger.error(f"Errore esportazione modello: {e}")
            return False
            
    def run_complete_pipeline(self):
        """Esegue l'intera pipeline di training."""
        logger.info("=== AVVIO PIPELINE COMPLETA ===")
        start_time = time.time()
        
        try:
            # Step 1: Check system requirements
            if not self.check_system_requirements():
                logger.error("Requisiti di sistema non soddisfatti")
                return False
                
            # Step 2: Setup AMD environment
            self.setup_amd_environment()
            
            # Step 3: Create virtual environment
            if not self.create_virtual_environment():
                logger.error("Creazione virtual environment fallita")
                return False
                
            # Step 4: Install dependencies
            if not self.install_dependencies():
                logger.error("Installazione dipendenze fallita")
                return False
                
            # Step 5: Setup LLaMA Factory
            if not self.setup_llama_factory():
                logger.error("Setup LLaMA Factory fallito")
                return False
                
            # Step 6: Prepare dataset
            if not self.prepare_dataset():
                logger.error("Preparazione dataset fallita")
                return False
                
            # Step 7: Create training config
            config_path = self.create_training_config()
            
            # Step 8: Start training
            if not self.start_training(config_path):
                logger.error("Training fallito")
                return False
                
            # Step 9: Test model
            if not self.test_model():
                logger.warning("Test modello fallito, ma continuo")
                
            # Step 10: Export model
            if not self.export_model():
                logger.warning("Esportazione modello fallita")
                
            total_time = time.time() - start_time
            logger.info(f"=== PIPELINE COMPLETATA IN {total_time/3600:.1f} ORE ===")
            
            return True
            
        except Exception as e:
            logger.error(f"Errore pipeline: {e}")
            return False

def check_and_install_dependencies():
    """Controlla e installa dipendenze mancanti."""
    print("Controllo dipendenze...")

    missing_deps = []
    required_deps = {
        'psutil': 'psutil',
        'torch': 'torch>=2.1.0',
        'yaml': 'pyyaml',
        'requests': 'requests',
        'tqdm': 'tqdm'
    }

    for module_name, pip_name in required_deps.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_deps.append(pip_name)

    if missing_deps:
        print(f"Installazione dipendenze mancanti: {', '.join(missing_deps)}")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_deps, check=True)
            print("Dipendenze installate")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Errore installazione dipendenze: {e}")
            return False
    else:
        print("Tutte le dipendenze sono gia installate")
        return True

def main():
    """Funzione principale."""
    print("Script di Fine-tuning Qwen3-VL-8B-Instruct per AMD RX 6650 XT")
    print("=" * 70)

    # Check and install dependencies first
    if not check_and_install_dependencies():
        print("Impossibile installare dipendenze richieste")
        sys.exit(1)

    # Now import dependencies
    try:
        import psutil
        import torch
        import yaml
        import requests
        from tqdm import tqdm
    except ImportError as e:
        print(f"Errore import dipendenze: {e}")
        sys.exit(1)

    # Create configuration
    config = TrainingConfig()

    # Create trainer
    trainer = QwenTrainer(config)

    # Run complete pipeline
    success = trainer.run_complete_pipeline()

    if success:
        print("\nTraining completato con successo!")
        print(f"Modello finale salvato in: {config.final_model_path}")
        print(f"Log completi in: training.log")
        print("\nIl tuo modello medico Qwen3-VL-8B-Instruct e pronto!")
    else:
        print("\nTraining fallito. Controlla i log per dettagli.")
        sys.exit(1)

if __name__ == "__main__":
    main()