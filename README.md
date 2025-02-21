# Risk-management
Il problema analizzato riguarda la gestione del rischio finanziario per AIFS, un’azienda che organizza
viaggi di studio all’estero. AIFS si trova esposta a due principali fonti di incertezza:
• Tasso di cambio tra USD e EUR: essendo l’azienda americana, ma con costi in euro, `e
vulnerabile a fluttuazioni del cambio.
• Volume di studenti: il numero di iscrizioni effettive `e incerto.
Per mitigare tali rischi, l’azienda utilizza un portafoglio di strumenti finanziari composto da:
• Contratti forward: contratti lineari e simmetrici che permettono di acquistare valuta in
un tempo futuro T ad un tasso fissato in t0
• Opzioni call: strumenti che danno il diritto, ma non l’obbligo, di acquistare valuta a un
tasso prefissato. Le call offrono flessibilit`a, ma hanno un costo iniziale (premio).
Le decisioni vanno prese al tempo t0 = 0 e si suppongono noti il prezzo forward, il prezzo strike e
il costo per la call e si assume, inoltre, che la maturit`a dei contratti sia T. L’obiettivo del modello
`e determinare il mix ottimale di forward e opzioni per ridurre al minimo il rischio associato alle
variazioni del volume studenti e del tasso di cambio.
