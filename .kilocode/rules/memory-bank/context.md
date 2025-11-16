# Contesto

## Stato attuale
Il progetto è in fase attiva di sviluppo per l'addestramento e l'ottimizzazione del modello Qwen3 utilizzando LLaMA Factory. Sono stati creati diversi script di setup e addestramento (setup_and_train.py, setup_and_train_gguf.py, setup_and_train_final.py, setup_and_train_simple.py, setup_and_train_colab.py). Il repository LLaMA Factory è stato integrato nella cartella llama_factory_data/. Sono stati risolti problemi di compatibilità con Windows passando a Google Colab per l'addestramento GPU. È stato risolto il problema di importazione del modulo 'llamafactory' nello script setup_and_train_colab.py, correggendo il percorso e il metodo di esecuzione del training. Sono stati implementati i suggerimenti di Google Gemini per ottimizzare i parametri di addestramento, inclusi cutoff_len, max_samples, fp16, torch_empty_cache_steps, e la gestione della conversione LoRA-GGUF. Nello script `setup_and_train_colab.py`, il modello `Qwen/Qwen3-VL-8B-Instruct` è stato sostituito con `Qwen/Qwen2.5-7B-Instruct` per ottimizzare l'addestramento per applicazioni mediche e per rimuovere la necessità di un token Hugging Face.

È stato creato il notebook Jupyter `colab.ipynb` ottimizzato per l'esecuzione su Google Colab, che include:
- Verifica automatica della connessione internet, GPU, RAM e CPU
- Montaggio automatico di Google Drive per il salvataggio dei risultati
- Clonazione del repository GitHub pubblico `https://github.com/Attapulgite999/prove.git`
- Installazione guidata delle dipendenze Python necessarie
- Esecuzione dello script di training con gestione degli errori
- Download automatico del modello GGUF finale per LM Studio
- Documentazione integrata con spiegazioni dettagliate di ogni passo

## Focus di lavoro
L'obiettivo attuale è completare l'implementazione degli script di addestramento e ottimizzazione per il modello Qwen3, con particolare attenzione all'esecuzione su Google Colab. Sono disponibili diversi approcci:
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
