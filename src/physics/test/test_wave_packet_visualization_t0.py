# ---------------------------------------------------------
# Idée générale du script
# ---------------------------------------------------------
#
# Le but ici est de représenter un électron non pas comme un
# petit point classique, mais comme un paquet d'onde.
#
# Une seule onde plane s'étend partout dans l'espace, donc elle
# ne permet pas vraiment de dire où se trouve la particule.
#
# Pour obtenir quelque chose de localisé, on superpose beaucoup
# d'ondes très proches les unes des autres. Leur addition crée
# une sorte de "groupe d'ondes" : le paquet.
#
# Visuellement, on obtient :
#
# - des oscillations rapides à l'intérieur ;
# - une enveloppe plus large autour de ces oscillations.
#
# Les oscillations sont liées à la longueur d'onde de de Broglie
# de l'électron.
#
# L'enveloppe indique la zone de l'espace dans laquelle la
# particule est principalement localisée.
#
# La fonction d'onde psi n'est pas directement une probabilité :
# c'est |psi|² qui donne la densité de probabilité de présence.
#
# Dans ce premier test, on construit le paquet à partir d'un
# électron libre de 100 eV et on l'observe à l'instant t = 0.
#
# La suite sera de faire évoluer ce paquet avec le temps pour
# voir son déplacement et son étalement, puis de pouvoir modifier
# ses paramètres dans une petite interface interactive.
# ---------------------------------------------------------
# ---------------------------------------------------------
# Explication simple
# ---------------------------------------------------------
#
# Imagine qu'on essaie de représenter un électron.
#
# En mécanique classique, on pourrait vouloir dire :
# "il est exactement ici et il va exactement à cette vitesse".
#
# En mécanique quantique, ce n'est pas vraiment comme ça.
# On décrit plutôt l'électron avec une fonction d'onde.
#
# Cette fonction d'onde ne donne pas directement la position
# exacte de l'électron. Elle indique plutôt comment sa présence
# est répartie dans l'espace.
#
# Pour construire quelque chose de localisé, on additionne
# plusieurs ondes très proches les unes des autres.
#
# Individuellement, ces ondes s'étendent partout.
# Mais quand on les additionne, elles se renforcent dans certaines
# zones et se compensent dans d'autres.
#
# Le résultat est un paquet d'onde :
# une zone limitée dans laquelle l'électron a davantage de chances
# d'être détecté.
#
# Les petites oscillations visibles à l'intérieur du paquet
# représentent le caractère ondulatoire de la particule.
#
# La grande forme extérieure représente l'enveloppe du paquet.
#
# Enfin, ce qu'on interprète comme "probabilité de trouver
# l'électron ici" est obtenu avec |psi|².
#
# Donc, en résumé :
#
# plusieurs ondes simples
#        ↓
# superposition
#        ↓
# paquet d'onde localisé
#        ↓
# |psi|² indique où l'électron a le plus de chances d'être trouvé
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from constants import m_e
from wave_packet import (
    energie_ev_vers_joule,
    impulsion,
    nombre_onde,
    psi,
)

Ec_eV = 100

Ec = energie_ev_vers_joule(Ec_eV)
p0 = impulsion(Ec, m_e)
k0 = nombre_onde(p0)

delta_k = 0.005 * k0

x = np.linspace(-80e-9, 80e-9, 10000)

t = 0.0

psi_0 = psi(
    x=x,
    t=t,
    k0=k0,
    delta_k=delta_k,
    m=m_e
)

densite = np.abs(psi_0) ** 2

x_nm = x * 1e9

psi_reelle = np.real(psi_0)
enveloppe = np.abs(psi_0)

plt.plot(
    x_nm,
    psi_reelle,
    color="red",
    linewidth=0.7,
    label="Re(ψ)"
)

plt.plot(
    x_nm,
    enveloppe,
    color="blue",
    linewidth=2,
    label="|ψ|"
)

plt.plot(
    x_nm,
    -enveloppe,
    color="blue",
    linewidth=2
)

plt.xlabel("Position x (nm)")
plt.ylabel("Amplitude")
plt.title("Paquet d'onde à t = 0")

plt.grid()
plt.legend()

plt.show()