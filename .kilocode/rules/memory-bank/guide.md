# Guida all'uso del Memory Bank Template

## Panoramica
Questo template fornisce una struttura predefinita per implementare il Memory Bank di Kilo Code in qualsiasi progetto software. I file forniti sono progettati per essere adattabili a diversi linguaggi di programmazione, framework e architetture.

## Istruzioni per l'uso

### 1. Copia dei file
Copiare l'intera cartella `.kilocode/` nella radice del progetto target. I file sono progettati per essere universali e funzionare con qualsiasi tipo di progetto software.

### 2. Personalizzazione
Dopo aver copiato i file, personalizzare i contenuti in base al progetto specifico:

- **brief.md**: Descrivere brevemente il progetto e i suoi obiettivi
- **product.md**: Spiegare la motivazione del progetto, i problemi che risolve e l'esperienza utente desiderata
- **context.md**: Aggiornare con lo stato attuale del lavoro e i prossimi passi
- **architecture.md**: Documentare l'architettura specifica del progetto, i pattern di design utilizzati e i percorsi critici
- **tech.md**: Elencare le tecnologie, linguaggi e strumenti specifici del progetto

### 3. Integrazione con MCP servers
I file del Memory Bank sono compatibili con i Model Context Protocol (MCP) servers. Per integrare con MCP servers:

- I file Markdown possono essere accessibili come risorse tramite MCP
- La struttura gerarchica permette una facile navigazione tramite strumenti MCP
- I contenuti strutturati sono ideali per l'elaborazione automatizzata

### 4. Gestione del contesto
- Aggiornare `context.md` all'inizio di ogni nuova sessione di lavoro
- Mantenere `architecture.md` aggiornato quando si introducono cambiamenti significativi
- Utilizzare `tech.md` per registrare aggiornamenti tecnologici e dipendenze

## Adattabilità ai linguaggi e framework

I template sono progettati per essere:
- Linguaggio-agnostici: funzionano con qualsiasi linguaggio di programmazione
- Framework-agnostici: adatti a qualsiasi framework o architettura
- Estensibili: possibile aggiungere altri file secondo necessità

## Esempi di personalizzazione

Per un progetto web in JavaScript:
```
tech.md potrebbe contenere:
- JavaScript, Node.js, Express
- React 18, Redux
- MongoDB, Mongoose
- Webpack, Babel
```

Per un progetto Python:
```
tech.md potrebbe contenere:
- Python 3.x
- Django o Flask
- PostgreSQL
- Docker, Celery
```

## Best practices
- Mantenere i contenuti concisi ma informativi
- Aggiornare regolarmente i file durante lo sviluppo
- Utilizzare un linguaggio chiaro e tecnico
- Includere percorsi di file e tecnologie specifiche quando rilevante