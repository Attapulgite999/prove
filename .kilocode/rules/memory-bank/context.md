# Contesto

## Stato attuale
Il progetto è in fase attiva di sviluppo per l'addestramento e l'ottimizzazione del modello Qwen3 utilizzando LLaMA Factory. Sono stati creati diversi script di setup e addestramento (setup_and_train.py, setup_and_train_gguf.py, setup_and_train_final.py, setup_and_train_simple.py, setup_and_train_colab.py). Il repository LLaMA Factory è stato integrato nella cartella llama_factory_data/. Sono stati risolti problemi di compatibilità con Windows passando a Google Colab per l'addestramento GPU. È stato risolto il problema di importazione del modulo 'llamafactory' nello script setup_and_train_colab.py, correggendo il percorso e il metodo di esecuzione del training. Sono stati implementati i suggerimenti di Google Gemini per ottimizzare i parametri di addestramento, inclusi cutoff_len, max_samples, fp16, torch_empty_cache_steps, e la gestione della conversione LoRA-GGUF. Nello script `setup_and_train_colab.py`, il modello `Qwen/Qwen3-VL-8B-Instruct` è stato sostituito con `Qwen/Qwen2.5-7B-Instruct` per ottimizzare l'addestramento per applicazioni mediche e per rimuovere la necessità di un token Hugging Face.

**RISOLTO (16/11/2025)**: È stato corretto il problema di installazione di LLaMA Factory in ambiente Colab. Il repository GitHub pubblico non includeva correttamente il sottomodulo LLaMA Factory, causando errori durante `pip install -e`. È stata modificata la funzione `setup_llama_factory()` per verificare la presenza dei file necessari (`setup.py` o `pyproject.toml`) e riclonare il repository se incompleto.

**RISOLTO (16/11/2025)**: È stato risolto il problema della cache di Google Colab che impediva il caricamento delle versioni aggiornate del codice. È stato modificato il notebook `colab.ipynb` per forzare sempre una clonazione fresca del repository rimuovendo prima la directory esistente.

**RISOLTO (16/11/2025)**: È stato risolto il problema di incompatibilità della libreria `trl`. È stata aggiunta la dipendenza `trl>=0.8.6,<=0.9.6` al file `requirements.txt` per garantire la compatibilità con LLaMA Factory.

**RISOLTO (16/11/2025)**: È stato risolto definitivamente l'ImportError di `PreTrainedModel` tra `peft` e `transformers`. È stato modificato il file `requirements.txt` di LLaMA Factory per specificare versioni compatibili: `transformers>=4.40.0,<4.45.0` e `peft>=0.10.0,<0.12.0`, garantendo che LLaMA Factory installi automaticamente le versioni corrette delle sue dipendenze.

**RISOLTO (16/11/2025)**: È stato disabilitato l'uso di `unsloth` per evitare conflitti di dipendenza con `llamafactory`. La libreria `unsloth` è stata commentata nel codice di installazione e il parametro `use_unsloth` è impostato su `False` nella configurazione del training.

**RISOLTO (16/11/2025)**: È stato aggiunto un comando di disinstallazione esplicita di `unsloth` e `unsloth-zoo` nel notebook Colab prima dell'installazione delle dipendenze, per eliminare definitivamente i conflitti di versione che impedivano l'avvio del training.

**RISOLTO (16/11/2025)**: È stato risolto definitivamente il conflitto di dipendenze modificando direttamente il file `requirements.txt` di LLaMA Factory per specificare versioni compatibili di `peft` e `transformers`, garantendo che il framework installi automaticamente le versioni corrette senza conflitti.

**RISOLTO (16/11/2025)**: È stato corretto un errore di sintassi nel file `requirements.txt` di LLaMA Factory che causava l'errore "Invalid requirement", rimuovendo una riga mal formattata.

**RISOLTO (17/11/2025)**: È stato risolto l'errore ModuleNotFoundError: No module named 'torchvision' durante l'esecuzione di accelerate launch aggiungendo l'installazione esplicita di timm nel notebook Colab.

È stato creato il notebook Jupyter `colab.ipynb` ottimizzato per l'esecuzione su Google Colab, che include:
- Verifica automatica della connessione internet, GPU, RAM e CPU
- Montaggio automatico di Google Drive per il salvataggio dei risultati
- Clonazione del repository GitHub pubblico `https://github.com/Attapulgite999/prove.git`
- Installazione guidata delle dipendenze Python necessarie
- Esecuzione dello script di training con gestione degli errori
- Download automatico del modello GGUF finale per LM Studio
- Documentazione integrata con spiegazioni dettagliate di ogni passo

## Focus di lavoro
Dopo aver riscontrato problemi di stabilità con LLaMA Factory dovuti a conflitti di dipendenza, è stato creato un nuovo progetto basato su **Axolotl**, un framework più robusto per il fine-tuning di modelli linguistici.

### Nuovo Progetto Axolotl
- **Directory**: `axolotl_training/`
- **Framework**: Axolotl (più stabile di LLaMA Factory)
- **Modello**: Qwen/Qwen2.5-7B-Instruct
- **Tecnica**: LoRA fine-tuning
- **Dataset**: Dati medici (medalpaca)
- **Output**: Modello ottimizzato + GGUF per LM Studio

### File del Progetto
- `axolotl_training/colab_axolotl.ipynb`: Notebook Colab principale
- `axolotl_training/config/qwen_axolotl.yaml`: Configurazione YAML per Axolotl
- `axolotl_training/data/`: Dataset di training
- `axolotl_training/README.md`: Guida all'utilizzo

### Script Legacy (LLaMA Factory)
I precedenti script rimangono disponibili per riferimento:
- setup_and_train.py: Script principale per l'addestramento (Windows)
- setup_and_train_gguf.py: Script per l'ottimizzazione GGUF
- setup_and_train_final.py: Versione finale dell'implementazione
- setup_and_train_simple.py: Versione semplificata per test rapidi
- setup_and_train_colab.py: Versione ottimizzata per Google Colab con GPU

## Prossimi passi
1. Testare l'esecuzione completa del notebook Colab con il repository pubblico
2. Verificare il corretto funzionamento del training end-to-end
3. Ottimizzare ulteriormente le prestazioni su GPU Colab
4. Implementare monitoraggio avanzato dell'addestramento
5. Creare documentazione tecnica dettagliata per Colab
6. Preparare il modello per la distribuzione e download da Colab

## Note importanti
- Il progetto utilizza principalmente Python con framework di machine learning
- Sono presenti dati di addestramento nella cartella data/ e llama_factory_data/data/
- L'architettura tecnica è documentata in technical_architecture_qwen3_amd.md
- È stato risolto il problema di installazione PyTorch ROCm su Windows passando a Colab
- È stato risolto l'errore WinError 1224 durante il training
- Il modello Qwen2.5-7B-Instruct viene scaricato automaticamente da Hugging Face durante il training
- I file del Memory Bank vengono aggiornati regolarmente per riflettere i progressi
- Ogni sessione di lavoro dovrebbe includere un aggiornamento dello stato attuale in questo file
- Risolto problema di importazione del modulo 'llamafactory' in ambiente Colab
- Il modello Qwen2.5-7B-Instruct scaricato da Hugging Face è compatibile con LM Studio
- Implementati i suggerimenti di Google Gemini per migliorare l'efficacia dell'addestramento
- Aggiunta funzione per unire gli adapter LoRA al modello base prima della conversione GGUF
- Aumentato il numero di esempi di addestramento e la lunghezza delle sequenze
- Ottimizzate le impostazioni di precisione mista (fp16) per prestazioni migliori
- Risolto problema con dipendenza mancante di unsloth durante l'installazione di LLaMA Factory
- Creato repository pubblico su GitHub: https://github.com/Attapulgite999/prove
- Implementato notebook Jupyter `colab.ipynb` per esecuzione guidata su Google Colab
- Configurato workflow completo: verifica connessione → setup ambiente → training → download GGUF
