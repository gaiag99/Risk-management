import numpy as np
import pandas as pd
from scipy.stats import beta, norm
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

def solve_cvar_model(
    num_scenari=500,
    alpha_1=0.8,
    beta_2=0.25,
    min_volume=10000,
    max_volume=30000,
    media_tasso=1.22,
    std_tasso=0.08,
    cvar_level=0.95,
    seed=42
):
    """
    Risolve il modello di minimizzazione CVaR, restituendo:
    - Fwd_opt (Forward ottimo)
    - Call_opt (Call ottimo)
    - Theta_opt (variabile di VaR approssimata)
    - cvar_obj (valore della funzione obiettivo, ovvero la CVaR al livello specificato)
    - scenario_cost_values (lista dei costi scenario per scenario)
    """

    # Fissiamo il seed per la riproducibilità
    np.random.seed(seed)

    # 1. GENERAZIONE DEGLI SCENARI
    # Campionamento dei volumi con distribuzione Beta
    sampled_volumes = beta.rvs(alpha_1, beta_2, size=num_scenari)
    sampled_volumes = min_volume + (max_volume - min_volume) * sampled_volumes

    # Campionamento dei tassi con distribuzione Normale
    sampled_exrates = np.random.normal(media_tasso, std_tasso, num_scenari)
    sampled_exrates = np.maximum(sampled_exrates, 0)  # Evita valori negativi

    # Creazione DataFrame scenari
    df_scenari = pd.DataFrame({
        'ID_Scenario': range(1, num_scenari + 1),
        'Volumes': sampled_volumes,
        'ExchangeRate': sampled_exrates
    })
    df_scenari['Probability'] = 1.0 / num_scenari

    # 2. IMPOSTAZIONE MODELLO CVaR
    # Livello di confidenza e termine CVaR
    term_cvar = 1.0 / ((1.0 - cvar_level) * num_scenari)

    # Creazione modello Gurobi
    m = Model("Model_CVaR")
    m.Params.OutputFlag = 0  

    # Variabili di decisionali prima di conoscere gli esiti aleatori
    FwdVolume = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="FwdVolume")
    CallVolume = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="CallVolume")

    # Variabile ausiliaria per la CVaR
    Theta = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Theta")

    # Variabili decisionali dipendenti dallo scenario s
    Zvars = {}
    ExercisedCalls = {}

    for idx in df_scenari.index:
        Zvars[idx] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Z_{idx}")
        ExercisedCalls[idx] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"ExCall_{idx}")
        

    m.update()

    # Parametri economici
    strike_forward = 1.22
    premium_call = 0.05
    scaling_factor = 1000.0

    scenarioCost = {}

    # 3. VINCOLI E COSTO SCENARIO
    for idx in df_scenari.index:
        vol_i = df_scenari.loc[idx, 'Volumes']
        rate_i = df_scenari.loc[idx, 'ExchangeRate'] 

        cost_i = (
            (FwdVolume + ExercisedCalls[idx]) * strike_forward * scaling_factor
            + premium_call * strike_forward * scaling_factor * CallVolume
            + rate_i * (vol_i - FwdVolume - ExercisedCalls[idx]) * scaling_factor
        )
        scenarioCost[idx] = cost_i

        # Vincolo: non si può coprire più del volume totale
        m.addConstr(FwdVolume + ExercisedCalls[idx] <= vol_i, name=f"coverage_{idx}")
        # Vincolo: non si possono esercitare più call di quante ne siano state acquistate
        m.addConstr(ExercisedCalls[idx] <= CallVolume, name=f"calllimit_{idx}")
        # Vincolo: definizione di z
        m.addConstr(Zvars[idx] >= cost_i - Theta, name=f"z_def_{idx}")
        

    # 4. FUNZIONE OBIETTIVO: CVaR
    m.setObjective(Theta + term_cvar * quicksum(Zvars[idx] for idx in df_scenari.index), GRB.MINIMIZE)

    # 5. RISOLUZIONE
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        Fwd_opt = FwdVolume.X
        Call_opt = CallVolume.X
        Theta_opt = Theta.X
        cvar_obj = m.ObjVal

        
        scenario_cost_values = [scenarioCost[idx].getValue() for idx in df_scenari.index]
        return Fwd_opt, Call_opt, Theta_opt, cvar_obj, scenario_cost_values
    else:
        return None, None, None, None, None




# 1) Impatto del numero di scenari


scenario_numbers = [500,800,1000,1300,1500,2000]
fwd_solutions = []
call_solutions = []
theta_values = []
cvar_values = []

print("=== Impatto del numero di scenari ===")
for ns in scenario_numbers:
    fwd, call, theta, cvar_obj, costs = solve_cvar_model(num_scenari=ns, seed=42)
    fwd_solutions.append(fwd)
    call_solutions.append(call)
    theta_values.append(theta)
    cvar_values.append(cvar_obj)

    print(f"NumScenari={ns:4d} | Fwd={fwd:.2f}, Call={call:.2f}, Theta={theta:.2f}, Obj(CVaR)={cvar_obj:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(scenario_numbers, fwd_solutions, 'o-', label='FwdVolume')
plt.plot(scenario_numbers, call_solutions, 's-', label='CallVolume')
plt.xlabel('Numero di scenari')
plt.ylabel('Quantità ottima di copertura')
plt.title('Impatto del numero di scenari sulla stabilità della soluzione')
plt.legend()
plt.grid(True)
plt.show()



# 2) Impatto del grado di incertezza (std_tasso)

std_values = [0.02, 0.03,0.04, 0.05, 0.08, 0.10, 0.15, 0.20, 0.50]
fwd_uncertainty = []
call_uncertainty = []
theta_uncertainty = []
cvar_uncertainty = []

print("\n=== Impatto della deviazione standard del tasso (std_tasso) ===")
for std_ in std_values:
    fwd, call, theta, cvar_obj, costs = solve_cvar_model(std_tasso=std_, seed=42)
    fwd_uncertainty.append(fwd)
    call_uncertainty.append(call)
    theta_uncertainty.append(theta)
    cvar_uncertainty.append(cvar_obj)

    print(f"std_tasso={std_:.2f} | Fwd={fwd:.2f}, Call={call:.2f}, Theta={theta:.2f}, Obj(CVaR)={cvar_obj:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(std_values, fwd_uncertainty, 'o-', label='FwdVolume')
plt.plot(std_values, call_uncertainty, 's-', label='CallVolume')
plt.xlabel('Deviazione standard (std_tasso)')
plt.ylabel('Quantità ottima di copertura')
plt.title('Impatto della deviazione standard del tasso di cambio')
plt.legend()
plt.grid(True)
plt.show()



# 3) Impatto del grado di avversione al rischio (cvar_level)

cvar_levels = [0.99, 0.95, 0.90, 0.85, 0.80]
fwd_risk = []
call_risk = []
theta_risk = []
cvar_risk = []

print("\n=== Impatto del livello di confidenza (cvar_level) ===")
for cl in cvar_levels:
    fwd, call, theta, cvar_obj, costs = solve_cvar_model(cvar_level=cl, seed=42)
    fwd_risk.append(fwd)
    call_risk.append(call)
    theta_risk.append(theta)
    cvar_risk.append(cvar_obj)

    print(f"cvar_level={cl:.2f} | Fwd={fwd:.2f}, Call={call:.2f}, Theta={theta:.2f}, Obj(CVaR)={cvar_obj:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(cvar_levels, fwd_risk, 'o-', label='FwdVolume')
plt.plot(cvar_levels, call_risk, 's-', label='CallVolume')
plt.xlabel('Livello di confidenza (cvar_level)')
plt.ylabel('Quantità ottima di copertura')
plt.title('Impatto del livello di confidenza CVaR (avversione al rischio)')
plt.legend()
plt.grid(True)
plt.show()



# 4) Impatto del grado di incertezza per il rischio volume
#    Variazione di uno dei parametri della Beta, es: beta_2

beta2_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
fwd_volrisk = []
call_volrisk = []
theta_volrisk = []
cvar_volrisk = []

print("\n=== Impatto della variazione di beta_2 (distribuzione Beta per i volumi) ===")
for b2 in beta2_values:
    fwd, call, theta, cvar_obj, costs = solve_cvar_model(beta_2=b2, seed=42)
    fwd_volrisk.append(fwd)
    call_volrisk.append(call)
    theta_volrisk.append(theta)
    cvar_volrisk.append(cvar_obj)

    print(f"beta_2={b2:.2f} | Fwd={fwd:.2f}, Call={call:.2f}, Theta={theta:.2f}, Obj(CVaR)={cvar_obj:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(beta2_values, fwd_volrisk, 'o-', label='FwdVolume')
plt.plot(beta2_values, call_volrisk, 's-', label='CallVolume')
plt.xlabel('beta_2')
plt.ylabel('Quantità ottima di copertura')
plt.title('Impatto di beta_2 (distribuzione Beta) sul rischio volume')
plt.legend()
plt.grid(True)
plt.show()

