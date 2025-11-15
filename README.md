# 🚀 Qwen3-VL-8B-Instruct Fine-tuning per AMD RX 6650 XT

Script completo per automatizzare il fine-tuning del modello **Qwen3-VL-8B-Instruct** su GPU **AMD Radeon RX 6650 XT (8GB)** con supporto ROCm.

## 📋 Requisiti di Sistema

- **GPU**: AMD Radeon RX 6650 XT (8GB GDDR6)
- **Sistema**: Windows 10/11 o Linux con ROCm 6.2
- **RAM**: Minimo 16GB consigliati
- **Spazio su disco**: Almeno 50GB liberi
- **Python**: 3.10 o superiore

## 🛠️ Caratteristiche

- ✅ **Setup automatico** ambiente virtuale Python
- ✅ **Installazione dipendenze** ottimizzate per ROCm
- ✅ **Configurazione AMD** automatica (HSA_OVERRIDE_GFX_VERSION=10.3.0)
- ✅ **Dataset MedAlpaca** conversione automatica
- ✅ **Training ottimizzato** per 8GB VRAM con LoRA e 4-bit quantization
- ✅ **Monitoraggio GPU** in tempo reale
- ✅ **Testing automatico** del modello addestrato
- ✅ **Export finale** modello pronto all'uso

## 🚀 Installazione e Uso

### 1. Esecuzione Script Principale

```bash
# Esegui lo script completo
python setup_and_train.py

# Oppure usa la versione semplificata
python setup_and_train_simple.py
```

### 2. Comando Finale (Tutto Automatico)

```bash
python setup_and_train.py
```

Lo script eseguirà automaticamente:
1. Creazione ambiente virtuale `.venv/`
2. Installazione PyTorch con ROCm support
3. Download modello Qwen3-VL-8B-Instruct
4. Preparazione dataset MedAlpaca
5. Configurazione training ottimizzata
6. Avvio training con monitoraggio
7. Testing finale e export modello

## 📊 Parametri di Training (Ottimizzati per 8GB VRAM)

| Parametro | Valore | Descrizione |
|-----------|---------|-------------|
| Batch Size | 1 | Ridotto per gestione memoria |
| Gradient Accumulation | 8 | Simula batch size effettivo di 8 |
| LoRA Rank | 4 | Basso per risparmiare memoria |
| LoRA Alpha | 8 | Rapporto alpha/rank = 2 |
| Learning Rate | 3e-5 | Standard per fine-tuning |
| Epochs | 2 | Sufficienti per adattamento |
| Max Length | 512 | Limitato per memoria |
| Quantization | 4-bit | Essenziale per 8GB VRAM |
| Gradient Checkpointing | ON | Riduce uso memoria del 50% |

## 📁 Struttura Output

```
./
├── .venv/                          # Ambiente virtuale Python
├── llama_factory_data/
│   ├── data/
│   │   ├── medalpaca.json         # Dataset convertito
│   │   └── dataset_info.json      # Info dataset LLaMA Factory
│   └── medical_qwen3_output/      # Risultati training
│       ├── checkpoint-500/        # Checkpoints intermedi
│       ├── adapter_model.bin      # LoRA adapter finale
│       └── ...
├── final_model/                   # Modello finale pronto all'uso
├── training.log                   # Log completo del training
├── training_config.yaml           # Configurazione training
└── train_model.py                 # Script training generato
```

## 🧪 Testing del Modello

Dopo il training, il modello viene testato automaticamente con:

1. **Italiano**: "Quali sono i sintomi principali della febbre?"
2. **English**: "What are the main symptoms of hypertension?"
3. **Italiano**: "Come si diagnostica il diabete?"
4. **English**: "What is the treatment for common cold?"
5. **Italiano**: "Quali sono le cause dell'ipertensione?"

## 💡 Utilizzo del Modello Finale

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Carica modello fine-tunato
model_path = "./final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Usa il modello
question = "Quali sono i sintomi del diabete?"
inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs, max_length=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🔧 Risoluzione Problemi

### Errore: "CUDA out of memory"
- Riduci `cutoff_length` a 256
- Aumenta gradient accumulation a 16
- Usa LoRA rank più basso (2)

### Errore: "ROCm not found"
- Installa ROCm 6.2 per Windows/Linux
- Verifica variabili ambiente: `HSA_OVERRIDE_GFX_VERSION=10.3.0`

### Training lento
- Normalmente richiede 4-8 ore per 2 epochs
- Usa mixed precision (bf16) già configurato
- Disabilita altre applicazioni GPU-intensive

## 📈 Monitoraggio Performance

Lo script include:
- **Monitoraggio GPU**: Utilizzo memoria ogni minuto
- **Progress bar**: Avanzamento training
- **Log dettagliato**: Salvato in `training.log`
- **Checkpoints**: Salvati ogni 500 steps

## 🔒 Sicurezza e Best Practices

- ✅ Tutto eseguito in ambiente virtuale isolato
- ✅ Nessuna modifica al sistema host
- ✅ Modello scaricato da Hugging Face (ufficiale)
- ✅ Dataset medico sicuro e verificato
- ✅ Cleanup automatico in caso di interruzione

## 🎯 Risultati Attesi

Dopo il fine-tuning, il modello sarà in grado di:
- Rispondere a domande mediche in italiano e inglese
- Fornire spiegazioni dettagliate su sintomi e trattamenti
- Mantenere coerenza medica nelle risposte
- Adattarsi al tuo dataset specifico

## 📞 Supporto

Se riscontri problemi:
1. Controlla `training.log` per dettagli errori
2. Verifica requisiti di sistema
3. Assicurati di avere spazio su disco sufficiente
4. Controlla che ROCm sia installato correttamente

---

**⚡ Nota**: Il training richiede tempo (4-8 ore) e risorse GPU intensive. Assicurati di avere alimentazione stabile e sufficiente spazio su disco prima di iniziare.