# Utilizzo del Memory Bank

## Attivazione del Memory Bank

Quando i file del Memory Bank sono presenti nella directory `.kilocode/rules/memory-bank/`, Kilo Code li leggerà automaticamente all'inizio di ogni sessione. Non è necessario alcun comando speciale per attivare questa funzionalità.

All'inizio di ogni risposta, vedrai uno dei seguenti indicatori:

- `[Memory Bank: Active]` - Indica che i file del Memory Bank sono stati letti e vengono utilizzati
- `[Memory Bank: Missing]` - Indica che i file del Memory Bank non sono stati trovati o sono vuoti

## Comandi disponibili

### Inizializzazione
Per inizializzare il Memory Bank per un nuovo progetto, usa il comando:
```
initialize memory bank
```

### Aggiornamento
Per aggiornare il Memory Bank con le ultime modifiche al progetto, usa il comando:
```
update memory bank
```

Puoi anche specificare un contesto particolare:
```
update memory bank using information from @/Makefile
```

### Aggiunta di task ripetitivi
Per memorizzare un task ripetitivo per uso futuro, usa:
```
add task
```
oppure
```
store this as a task
```

## Comportamento durante l'esecuzione dei task

All'inizio di ogni task, Kilo Code legge automaticamente tutti i file del Memory Bank. Alla fine di un task significativo, potrebbe chiederti:
```
Would you like me to update the memory bank to reflect these changes?
```

## Best practices

- Lascia che Kilo Code legga automaticamente i file del Memory Bank all'inizio di ogni sessione
- Usa il comando `update memory bank` dopo aver completato cambiamenti significativi
- Aggiorna manualmente il file `context.md` quando inizi a lavorare su una nuova funzionalità
- Usa i comandi sopra menzionati quando hai bisogno di una scansione approfondita del progetto