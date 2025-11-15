# 🚀 Fine-tuning Qwen3-VL-8B-Instruct GGUF per LMStudio

## 📋 Panoramica

Questo script è **specificamente progettato** per fine-tunare il tuo modello **Qwen3-VL-8B-Instruct-Q4_K_M.gguf** da LMStudio su **AMD RX 6650 XT (8GB)**.

## 🎯 Vantaggi di questa versione GGUF

✅ **Parti da modello già ottimizzato** (Q4_K_M - 4-bit quantization)  
✅ **Output compatibile con LMStudio** (formato GGUF)  
✅ **Training più veloce** (nessun download da Hugging Face)  
✅ **Memoria ottimizzata** per 8GB VRAM  
✅ **Mantiene qualità originale** del modello  

## 🛠️ Requisiti Specifici

- **GPU**: AMD Radeon RX 6650 XT (8GB GDDR6)
- **Modello**: Qwen3-VL-8B-Instruct-Q4_K_M.gguf (già in tuo possesso)
- **Sistema**: Windows 10/11 con ROCm support
- **Spazio**: 30GB liberi per training temporaneo

## 🚀 Comando Unico

```bash
python setup_and_train_gguf.py
```

**Tutto è automatizzato!** Lo script eseguirà:

## 📊 Processo Dettagliato

### 1️⃣ **Setup Ambiente** (5-10 min)
- Crea ambiente virtuale isolato `.venv_gguf/`
- Configura AMD ROCm: `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- Installa dipendenze specifiche per GGUF

### 2️⃣ **Verifica Modello** (1 min)
- Controlla il tuo file: `Qwen3-VL-8B-Instruct-Q4_K_M.gguf`
- Verifica dimensioni e integrità
- Mostra info: ~4-6GB (4-bit quantization)

### 3️⃣ **Preparazione Dataset** (2-3 min)
- **Converte** il tuo `technical_architecture_qwen3_amd.md`
- **Crea formato** domanda-risposta per training
- **Ottimizza** per contesto medico

### 4️⃣ **Training GGUF** (4-8 ore)
- **Carica modello Q4_K_M** in memoria GPU
- **Applica LoRA fine-tuning** (rank=4, alpha=8)
- **Training medico** con 2 epochs
- **Monitoraggio memoria** in tempo reale
- **Salva checkpoints** ogni 100 steps

### 5️⃣ **Conversione GGUF** (5-10 min)
- **Converte risultati** in formato GGUF
- **Mantiene quantizzazione** Q4_K_M
- **Crea file info** per LMStudio

### 6️⃣ **Testing Finale** (2-3 min)
- **5 domande mediche** test (ITA/ENG)
- **Verifica qualità** risposte
- **Report performance**

## 📁 Output Generato

```
./
├── .venv_gguf/                    # Ambiente virtuale
├── training_data_gguf/
│   └── medical_dataset_gguf.json  # Dataset convertito
├── gguf_training_output/           # Risultati training
│   ├── checkpoint-100/             # Checkpoints intermedi
│   ├── checkpoint-200/
│   └── adapter_model.bin          # LoRA adapter finale
├── final_model_gguf/              # Modello finale per LMStudio
│   ├── medical_qwen3_q4km.gguf   # Modello fine-tunato
│   ├── model_info.txt             # Info modello
│   └── README.md                  # Istruzioni LMStudio
└── training_gguf.log              # Log completo
```

## 🎯 Risultato Finale

**Modello**: `medical_qwen3_q4km.gguf`  
**Compatibile**: ✅ LMStudio  
**Dimensione**: ~4-6GB (simile a originale)  
**Qualità**: Mantiene intelligenza base + medicina  

## 💡 Uso in LMStudio

### 1. **Importa il Modello**
```
LMStudio → Models → Import Model
Seleziona: final_model_gguf/medical_qwen3_q4km.gguf
```

### 2. **Configura Parametri**
```
Temperature: 0.7
Max Tokens: 2048
Top P: 0.9
System Prompt: "Sei un assistente medico esperto."
```

### 3. **Test il Modello**
```
Domanda: "Quali sono i sintomi del diabete?"
Risposta: [Risposta medica accurata in italiano]
```

## ⚡ Parametri Ottimizzati per 8GB VRAM

| Parametro | Valore | Perché? |
|-----------|---------|---------|
| **Batch Size** | 1 | Riduce uso memoria |
| **Gradient Accumulation** | 8 | Simula batch 8 |
| **LoRA Rank** | 4 | Basso per 8GB VRAM |
| **LoRA Alpha** | 8 | Rapporto ottimale |
| **Max Length** | 512 | Limitato per memoria |
| **Quantization** | Q4_K_M | Già ottimizzato |
| **Learning Rate** | 3e-5 | Standard fine-tuning |
| **Epochs** | 2 | Sufficiente per adattamento |

## 🔧 Risoluzione Problemi

### **"CUDA out of memory"**
- Riduci `cutoff_length` a 256
- Diminuisci LoRA rank a 2
- Aumenta gradient accumulation a 16

### **"Modello troppo grande"**
- Il tuo Q4_K_M è ~4-6GB, perfetto per 8GB VRAM
- Training temporaneo usa ~2-3GB aggiuntivi

### **Training lento**
- Normale: 4-8 ore per 2 epochs completi
- Dipende da complessità dataset medico

## 📈 Monitoraggio Progresso

Lo script mostra:
- **Progresso training** (step/epoch)
- **Uso memoria GPU** (GB liberi/usati)
- **Loss** (deve diminuire)
- **Tempo stimato** rimanente

## 🎯 Risultati Attesi

Dopo il training, il tuo modello sarà in grado di:

✅ **Rispondere a domande mediche** in italiano e inglese  
✅ **Spiegare sintomi e trattamenti** dettagliatamente  
✅ **Mantenere coerenza medica** nelle risposte  
✅ **Adattarsi al tuo dataset** tecnico specifico  
✅ **Funzionare perfettamente** in LMStudio  

## 🔒 Protezioni Termiche (Sicurezza Hardware)

Lo script include **sistemi avanzati di protezione** per il tuo hardware:

### 🌡️ **Monitoraggio Temperature**
- **CPU**: Monitoraggio continuo (max 85°C)
- **GPU**: Controllo temperatura AMD (max 83°C)
- **Alert automatici** se temperature troppo alte
- **Log dettagliato** temperature ogni 30 secondi

### ⚡ **Ottimizzazioni Automatiche**
- **Riduzione batch size** se CPU > 80°C
- **Aumento gradient accumulation** per raffreddamento
- **Limitazione thread CPU** (max 4 threads)
- **Gestione memoria** ottimizzata

### 🛑 **Sicurezza Critica**
- **Arresto automatico** se GPU > 87°C (temperatura critica)
- **Interruzione sicura** con Ctrl+C (salva progressi)
- **Recovery mode** per riprendere da checkpoint
- **No overclocking** - usa impostazioni sicure

### 📊 **Parametri Sicuri di Default**
```
CPU Max: 85°C     (soglia allerta: 80°C)
GPU Max: 83°C     (soglia allerta: 78°C)
Critical: 87°C    (arresto automatico)
CPU Usage: <90%   (monitoraggio continuo)
Memory: <95%      (gestione automatica)
```

## 🚀 Comando Finale

```bash
# Esegui e attendi 4-8 ore (con protezioni attive)
python setup_and_train_gguf.py

# Il tuo modello sarà pronto per LMStudio!
# Monitoraggio temperature attivo durante tutto il training
```

**🔒 Il tuo hardware è protetto!** Lo script monitora e ottimizza automaticamente per evitare surriscaldamento.

**💡 Suggerimenti per temperature ottimali:**
- Assicurati di avere **buona ventilazione**
- **Pulisci le ventole** se necessario
- Considera **undervolting** se hai esperienza
- Usa **HWiNFO64** per monitorare in Windows

**Buon training sicuro!** 🎯 Il tuo assistente medico AI sta per nascere, senza stress per il tuo hardware!