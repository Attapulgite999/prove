# 🚀 Fine-tuning Qwen2.5 con Axolotl

Questo progetto implementa un approccio più stabile per il fine-tuning del modello Qwen2.5-7B-Instruct utilizzando Axolotl invece di LLaMA Factory.

## 📋 Prerequisiti

- Google Colab con GPU (raccomandato T4 o superiore)
- Google Drive per salvare i risultati
- Connessione internet stabile

## 📁 Struttura del Progetto

```
axolotl_training/
├── config/
│   └── qwen_axolotl.yaml    # Configurazione Axolotl
├── data/
│   ├── medalpaca.json       # Dataset di training
│   └── dataset_info.json    # Metadati del dataset
├── colab_axolotl.ipynb      # Notebook Colab principale
└── README.md               # Questa guida
```

## 🚀 Come Utilizzare

### 1. Apri il Notebook su Colab

1. Vai su [Google Colab](https://colab.research.google.com/)
2. Carica il file `colab_axolotl.ipynb`
3. Abilita la GPU: Runtime → Change runtime type → T4 GPU

### 2. Esegui le Celle in Ordine

1. **Cella 0**: Verifica ambiente (GPU, RAM, CPU)
2. **Cella 1**: Setup ambiente e clonazione repository
3. **Cella 2**: Installazione Axolotl e dipendenze
4. **Cella 3**: Avvio del training
5. **Cella 4**: Conversione in GGUF (dopo il training)
6. **Cella 5**: Download del modello finale

### 3. Monitora il Training

Il training mostrerà:
- Progress bar con loss e step
- Utilizzo GPU e memoria
- Tempi stimati

## ⚙️ Configurazione

Il file `config/qwen_axolotl.yaml` contiene tutti i parametri di training:

- **Modello**: Qwen/Qwen2.5-7B-Instruct
- **Tecnica**: LoRA (r=8, alpha=16)
- **Dataset**: 500 esempi medici (medalpaca)
- **Epoche**: 3
- **Batch size**: 1 (con gradient accumulation 32)
- **Sequence length**: 256 token
- **Learning rate**: 5e-5

## 📊 Risultati Attesi

- **Durata**: ~2-4 ore su T4 GPU
- **Utilizzo memoria**: ~12-14GB
- **Output**: Modello LoRA + GGUF per LM Studio

## 🔧 Troubleshooting

### Errore di Memoria GPU
- Riduci `micro_batch_size` a 1
- Aumenta `gradient_accumulation_steps`

### Errore di Timeout Colab
- Il notebook include un keep-alive thread
- Monitora la sessione regolarmente

### Problemi con il Dataset
- Verifica che i file `medalpaca.json` esistano
- Controlla il formato JSON (instruction, input, output)

## 📈 Ottimizzazioni Future

Una volta che il training base funziona, puoi aggiungere:
- **Unsloth**: Per accelerare il training
- **Flash Attention**: Per prestazioni migliori
- **Dataset più grande**: Per risultati migliori

## 🤝 Supporto

Se riscontri problemi:
1. Controlla i log di errore
2. Verifica la configurazione YAML
3. Assicurati che la GPU sia abilitata

Buon fine-tuning! 🎉