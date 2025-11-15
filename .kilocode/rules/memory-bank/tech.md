# Tecnologie e Strumenti  
  
## Linguaggi e Framework  
- **Python 3.x**: Linguaggio principale per gli script di addestramento  
- **LLaMA Factory**: Framework per l'addestramento di modelli linguistici  
- **PyTorch**: Framework di deep learning sottostante  
- **Transformers**: Libreria Hugging Face per modelli transformer  
- **GGUF**: Formato di quantizzazione per ottimizzazione dei modelli  
  
## Strumenti e Setup  
- **Visual Studio Code**: Editor principale per lo sviluppo  
- **Git**: Controllo versione per il codice sorgente  
- **Conda/Virtualenv**: Gestione ambienti virtuali Python  
- **CUDA**: Accelerazione GPU per l'addestramento (se disponibile)  
- **TensorBoard**: Monitoraggio dell'addestramento
- **Google Colab**: Ambiente di sviluppo per l'addestramento con GPU
  
## Librerie principali  
- **torch**: Framework PyTorch per deep learning  
- **transformers**: Modelli pre-addestrati e tokenizers  
- **datasets**: Gestione dataset per l'addestramento  
- **accelerate**: Distribuzione dell'addestramento su multiple GPU  
- **peft**: Parameter-Efficient Fine-Tuning  
- **bitsandbytes**: Quantizzazione dei modelli  
  
## Configurazione  
- Ambiente virtuale: .venv/ nella directory del progetto  
- Dipendenze: requirements.txt con tutte le librerie necessarie  
- Repository LLaMA Factory: Integrato in llama_factory_data/  
- Dataset: Organizzati nella cartella data/  
- Configurazioni: Parametri di addestramento nei file di setup  
  
## Setup di sviluppo  
1. Creare ambiente virtuale: python -m venv .venv  
2. Attivare ambiente: .venv\Scripts\activate (Windows)  
3. Installare dipendenze: pip install -r requirements.txt  
4. Verificare installazione: python -c \"import torch; print^('PyTorch version:', torch.__version__^)\"  
  
## Note di sviluppo  
- Utilizzo di GPU consigliato per l'addestramento efficiente  
- Monitorare l'utilizzo di memoria durante l'addestramento  
- Backup regolari dei checkpoint del modello  
- Documentazione tecnica in technical_architecture_qwen3_amd.md  
- Aggiornamenti del Memory Bank ad ogni sessione significativa 
