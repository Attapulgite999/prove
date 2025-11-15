# Architettura

## Struttura del progetto
Il progetto implementa un sistema completo per l'addestramento e l'ottimizzazione del modello Qwen3 con la seguente struttura:

```
prove/
 .kilocode/rules/memory-bank/     # Documentazione del contesto
 .venv/                          # Ambiente virtuale Python
 data/                           # Dataset di addestramento
 llama_factory_data/             # Repository LLaMA Factory
    LLaMA-Factory/              # Codice sorgente del framework
 setup_and_train.py              # Script principale di addestramento
 setup_and_train_gguf.py         # Script per ottimizzazione GGUF
 setup_and_train_final.py        # Implementazione finale
 setup_and_train_simple.py       # Versione semplificata
 setup_and_train_colab.py        # Versione ottimizzata per Google Colab con GPU
 technical_architecture_qwen3_amd.md  # Documentazione tecnica
 requirements.txt                # Dipendenze Python
 README files                    # Documentazione
```

## Descrizione dei componenti
- **Script di setup e addestramento**: Diversi script Python per configurare e avviare l'addestramento
- **LLaMA Factory**: Framework principale per l'addestramento dei modelli linguistici
- **Data directory**: Contiene i dataset utilizzati per l'addestramento
- **Memory Bank**: Sistema di documentazione per mantenere il contesto
- **Technical documentation**: File di documentazione tecnica dettagliata

## Pattern di design
- **Modularità**: Script separati per diverse funzionalità (setup, training, optimization)
- **Configurazione esterna**: Parametri di addestramento definiti in file di configurazione
- **Logging strutturato**: Monitoraggio dettagliato del processo di addestramento
- **Versionamento**: Diverse versioni degli script per approcci differenti

## Percorsi critici
- setup_and_train.py: Punto di ingresso principale per l'addestramento
- llama_factory_data/LLaMA-Factory/src/: Codice sorgente del framework
- data/: Dataset utilizzati per l'addestramento
- technical_architecture_qwen3_amd.md: Documentazione tecnica di riferimento
- .kilocode/rules/memory-bank/: Documentazione del contesto del progetto

## Decisioni architetturali
- Utilizzo di LLaMA Factory come framework principale per la sua flessibilità
- Supporto multi-formato (standard e GGUF) per diverse esigenze di deployment
- Struttura modulare per facilitare manutenzione e estensioni
- Documentazione integrata nel progetto per garantire continuità
