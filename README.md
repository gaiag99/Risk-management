# Risk-management
Il problema analizzato riguarda la gestione del rischio finanziario per AIFS, un’azienda che organizza
viaggi di studio all’estero. AIFS si trova esposta a due principali fonti di incertezza:
1. Tasso di cambio tra USD e EUR: essendo l’azienda americana, ma con costi in euro, è
vulnerabile a fluttuazioni del cambio.
2. Volume di studenti: il numero di iscrizioni effettive è incerto.

Per mitigare tali rischi, l’azienda utilizza un portafoglio di strumenti finanziari composto da:
- Contratti forward: contratti lineari e simmetrici che permettono di acquistare valuta in
un tempo futuro T ad un tasso fissato in t0
- Opzioni call: strumenti che danno il diritto, ma non l’obbligo, di acquistare valuta a un
tasso prefissato. Le call offrono flessibilità, ma hanno un costo iniziale (premio).

Le decisioni vanno prese al tempo t0 = 0 e si suppongono noti il prezzo forward, il prezzo strike e
il costo per la call e si assume, inoltre, che la maturità dei contratti sia T. L’obiettivo del modello
è determinare il mix ottimale di forward e opzioni per ridurre al minimo il rischio associato alle
variazioni del volume studenti e del tasso di cambio.
