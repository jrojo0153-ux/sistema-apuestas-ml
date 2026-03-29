from engine.ev_kelly import calcular_ev, calcular_kelly

def evaluar_apuesta(prob, cuota):
    ev = calcular_ev(prob, cuota)
    kelly = calcular_kelly(prob, cuota)

    return ev, kelly
