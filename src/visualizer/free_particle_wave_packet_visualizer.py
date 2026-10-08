# ---------------------------------------------------------
# Wave Packet Visualizer
# ---------------------------------------------------------
#
# Ce programme est un visualiseur interactif de paquet d'onde
# pour une particule libre en mécanique quantique.
#
# Le paquet d'onde est calculé à partir du moteur physique défini
# dans physics/wave_packet.py.
#
# L'interface permet de :
#
# - faire évoluer le paquet dans le temps ;
# - mettre la simulation en pause ;
# - avancer ou reculer dans le temps ;
# - modifier indépendamment le nombre d'onde central k0 ;
# - modifier indépendamment la largeur spectrale delta_k ;
# - observer la partie réelle de la fonction d'onde Re(psi) ;
# - observer son enveloppe |psi|.
#
# Le nombre d'onde central k0 détermine notamment la longueur
# d'onde de de Broglie et l'impulsion moyenne de la particule.
#
# delta_k représente l'étendue des nombres d'onde présents dans
# la superposition. Il contrôle donc la largeur spectrale du paquet
# et influence sa localisation dans l'espace.
#
# Le but de ce visualiseur est d'expérimenter directement avec
# les paramètres physiques et d'observer comment une superposition
# d'ondes planes se comporte au cours du temps.
#
# Ce fichier contient uniquement l'interface et la visualisation.
# Les équations et calculs physiques restent séparés dans le module
# physics/.
# ---------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, TextBox


# =========================================================
# IMPORT DU MOTEUR PHYSIQUE
# =========================================================

# app.py se trouve dans :
# src/visualizer/
#
# On ajoute src/ aux chemins Python afin de pouvoir importer :
# src/physics/constants.py
# src/physics/wave_packet.py

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from physics.constants import m_e, hbar, eV

from physics.wave_packet import (
    energie_ev_vers_joule,
    impulsion,
    nombre_onde,
    psi,
)


# =========================================================
# PARAMÈTRES INITIAUX
# =========================================================

# On part initialement d'un électron de 100 eV
# uniquement pour obtenir une valeur de départ cohérente de k0.

energie_initiale_eV = 100.0

energie_initiale_J = energie_ev_vers_joule(
    energie_initiale_eV
)

p0_initial = impulsion(
    energie_initiale_J,
    m_e
)

k0_initial = nombre_onde(
    p0_initial
)


# Largeur initiale du spectre en k
#
# Ici :
# Δk = 1 % du k0 initial
#
# Après le lancement, k0 et Δk sont totalement indépendants.

delta_k_initial = 0.01 * k0_initial


# =========================================================
# ESPACE DE SIMULATION
# =========================================================

# Domaine spatial :
# -100 nm -> +100 nm

x = np.linspace(
    -100e-9,
    100e-9,
    3000
)

# Conversion en nm uniquement pour l'affichage

x_nm = x * 1e9


# =========================================================
# TEMPS
# =========================================================

FS = 1e-15

t_min_fs = 0.0
t_max_fs = 15.0

# Pas physique lorsque le player avance

dt_fs = 0.1


# =========================================================
# PRÉCISION DE L'INTÉGRALE SUR k
# =========================================================

# Nombre de valeurs de k utilisées pour approximer :
#
# psi(x,t) = ∫ A(k) exp(i(kx - omega*t)) dk
#
# Plus cette valeur est grande, plus c'est précis,
# mais plus l'animation demande de calcul.

N_K = 300


# =========================================================
# FONCTIONS DE L'INTERFACE
# =========================================================

def lire_k0():
    """
    Lit la valeur de k0 saisie dans l'interface.
    """

    try:
        valeur = float(input_k0.text)

        if valeur <= 0:
            return k0_initial

        return valeur

    except ValueError:
        return k0_initial


def lire_delta_k():
    """
    Lit la largeur Δk saisie dans l'interface.
    """

    try:
        valeur = float(input_delta_k.text)

        if valeur <= 0:
            return delta_k_initial

        return valeur

    except ValueError:
        return delta_k_initial


def calculer_energie_depuis_k(k):
    """
    Énergie cinétique correspondant à un nombre d'onde k.

    E = hbar² k² / (2m)
    """

    E_joule = (
        hbar**2
        * k**2
        / (2 * m_e)
    )

    return E_joule / eV


def calculer_longueur_onde(k):
    """
    lambda = 2*pi / k
    """

    return (
        2 * np.pi / k
    )


# =========================================================
# ÉTAT INITIAL
# =========================================================

psi_initial = psi(
    x=x,
    t=0.0,
    k0=k0_initial,
    delta_k=delta_k_initial,
    m=m_e,
    nombre_points=N_K
)


# Pour l'interface, on utilise une amplitude relative.
#
# Cela permet d'observer facilement la forme du paquet,
# même lorsque k0 ou Δk changent fortement.

amplitude_initiale = np.max(
    np.abs(psi_initial)
)


# =========================================================
# CRÉATION DU GRAPHIQUE
# =========================================================

fig, ax = plt.subplots(
    figsize=(13, 8)
)

# Place réservée en bas pour les contrôles

plt.subplots_adjust(
    bottom=0.34
)


# ---------------------------------------------------------
# Partie réelle de psi
# ---------------------------------------------------------

ligne_reelle, = ax.plot(
    x_nm,
    np.real(psi_initial) / amplitude_initiale,
    color="red",
    linewidth=0.7,
    label="Re(ψ)"
)


# ---------------------------------------------------------
# Enveloppe +|psi|
# ---------------------------------------------------------

ligne_enveloppe_plus, = ax.plot(
    x_nm,
    np.abs(psi_initial) / amplitude_initiale,
    color="blue",
    linewidth=2,
    label="|ψ|"
)


# ---------------------------------------------------------
# Enveloppe -|psi|
# ---------------------------------------------------------

ligne_enveloppe_moins, = ax.plot(
    x_nm,
    -np.abs(psi_initial) / amplitude_initiale,
    color="blue",
    linewidth=2
)


ax.set_xlim(
    x_nm[0],
    x_nm[-1]
)

ax.set_ylim(
    -1.1,
    1.1
)

ax.set_xlabel(
    "Position x (nm)"
)

ax.set_ylabel(
    "Amplitude relative"
)

ax.grid()

ax.legend()

titre = ax.set_title(
    "Paquet d'onde — t = 0.00 fs"
)


# =========================================================
# SLIDER TEMPOREL
# =========================================================

axe_slider_temps = plt.axes(
    [0.18, 0.22, 0.65, 0.035]
)

slider_temps = Slider(
    ax=axe_slider_temps,
    label="Temps (fs)",
    valmin=t_min_fs,
    valmax=t_max_fs,
    valinit=0.0,
    valstep=dt_fs
)


# =========================================================
# INPUT k0
# =========================================================

axe_input_k0 = plt.axes(
    [0.17, 0.145, 0.20, 0.045]
)

input_k0 = TextBox(
    axe_input_k0,
    "k0 (m⁻¹)",
    initial=f"{k0_initial:.4e}"
)


axe_k0_moins = plt.axes(
    [0.385, 0.145, 0.05, 0.045]
)

axe_k0_plus = plt.axes(
    [0.44, 0.145, 0.05, 0.045]
)

bouton_k0_moins = Button(
    axe_k0_moins,
    "-"
)

bouton_k0_plus = Button(
    axe_k0_plus,
    "+"
)


# =========================================================
# INPUT DELTA k
# =========================================================

axe_input_delta_k = plt.axes(
    [0.58, 0.145, 0.20, 0.045]
)

input_delta_k = TextBox(
    axe_input_delta_k,
    "Δk (m⁻¹)",
    initial=f"{delta_k_initial:.4e}"
)


axe_delta_moins = plt.axes(
    [0.795, 0.145, 0.05, 0.045]
)

axe_delta_plus = plt.axes(
    [0.85, 0.145, 0.05, 0.045]
)

bouton_delta_moins = Button(
    axe_delta_moins,
    "-"
)

bouton_delta_plus = Button(
    axe_delta_plus,
    "+"
)


# =========================================================
# BOUTONS DU PLAYER
# =========================================================

axe_retour = plt.axes(
    [0.27, 0.055, 0.10, 0.055]
)

axe_play = plt.axes(
    [0.41, 0.055, 0.15, 0.055]
)

axe_avance = plt.axes(
    [0.60, 0.055, 0.10, 0.055]
)

axe_reset = plt.axes(
    [0.74, 0.055, 0.10, 0.055]
)


bouton_retour = Button(
    axe_retour,
    "←"
)

bouton_play = Button(
    axe_play,
    "Play"
)

bouton_avance = Button(
    axe_avance,
    "→"
)

bouton_reset = Button(
    axe_reset,
    "Reset"
)


# =========================================================
# MISE À JOUR DE LA SIMULATION
# =========================================================

def mettre_a_jour(_=None):

    # Temps actuel sélectionné

    t_fs = slider_temps.val

    t = (
        t_fs * FS
    )


    # Valeurs choisies manuellement

    k0 = lire_k0()

    delta_k = lire_delta_k()


    # -----------------------------------------------------
    # Calcul avec TA fonction psi()
    # définie dans physics/wave_packet.py
    # -----------------------------------------------------

    psi_t = psi(
        x=x,
        t=t,
        k0=k0,
        delta_k=delta_k,
        m=m_e,
        nombre_points=N_K
    )


    # -----------------------------------------------------
    # Échelle relative pour l'affichage
    # -----------------------------------------------------

    amplitude = np.max(
        np.abs(psi_t)
    )

    if amplitude == 0:
        return


    partie_reelle = (
        np.real(psi_t)
        / amplitude
    )

    enveloppe = (
        np.abs(psi_t)
        / amplitude
    )


    # -----------------------------------------------------
    # Actualisation des courbes
    # -----------------------------------------------------

    ligne_reelle.set_ydata(
        partie_reelle
    )

    ligne_enveloppe_plus.set_ydata(
        enveloppe
    )

    ligne_enveloppe_moins.set_ydata(
        -enveloppe
    )


    # -----------------------------------------------------
    # Informations physiques dérivées de k0
    # -----------------------------------------------------

    energie_eV = calculer_energie_depuis_k(
        k0
    )

    lambda_m = calculer_longueur_onde(
        k0
    )

    lambda_nm = (
        lambda_m * 1e9
    )


    titre.set_text(
        f"k0 = {k0:.3e} m⁻¹"
        f"   |   Δk = {delta_k:.3e} m⁻¹"
        f"   |   λ₀ = {lambda_nm:.4f} nm"
        f"   |   E ≈ {energie_eV:.2f} eV"
        f"   |   t = {t_fs:.2f} fs"
    )


    fig.canvas.draw_idle()


# Le déplacement du temps recalcule psi

slider_temps.on_changed(
    mettre_a_jour
)


# Entrée dans les champs k0 / Δk

input_k0.on_submit(
    mettre_a_jour
)

input_delta_k.on_submit(
    mettre_a_jour
)


# =========================================================
# MODIFICATION DE k0
# =========================================================

def k0_plus(event):

    valeur = lire_k0()

    # +10 %
    valeur *= 1.10

    input_k0.set_val(
        f"{valeur:.4e}"
    )


def k0_moins(event):

    valeur = lire_k0()

    # - environ 10 %
    valeur /= 1.10

    input_k0.set_val(
        f"{valeur:.4e}"
    )


bouton_k0_plus.on_clicked(
    k0_plus
)

bouton_k0_moins.on_clicked(
    k0_moins
)


# =========================================================
# MODIFICATION DE DELTA k
# =========================================================

def delta_plus(event):

    valeur = lire_delta_k()

    # Δk augmente de 20 %
    valeur *= 1.20

    input_delta_k.set_val(
        f"{valeur:.4e}"
    )


def delta_moins(event):

    valeur = lire_delta_k()

    # Δk diminue de 20 %
    valeur /= 1.20

    input_delta_k.set_val(
        f"{valeur:.4e}"
    )


bouton_delta_plus.on_clicked(
    delta_plus
)

bouton_delta_moins.on_clicked(
    delta_moins
)


# =========================================================
# PLAYER
# =========================================================

lecture = False


def play_pause(event):

    global lecture

    lecture = not lecture

    if lecture:

        bouton_play.label.set_text(
            "Pause"
        )

    else:

        bouton_play.label.set_text(
            "Play"
        )


def avancer(event):

    nouveau_t = min(
        slider_temps.val + dt_fs,
        t_max_fs
    )

    slider_temps.set_val(
        nouveau_t
    )


def reculer(event):

    nouveau_t = max(
        slider_temps.val - dt_fs,
        t_min_fs
    )

    slider_temps.set_val(
        nouveau_t
    )


def reset(event):

    global lecture

    lecture = False

    bouton_play.label.set_text(
        "Play"
    )


    # Temps remis à zéro

    slider_temps.set_val(
        0.0
    )


    # Paramètres physiques initiaux

    input_k0.set_val(
        f"{k0_initial:.4e}"
    )

    input_delta_k.set_val(
        f"{delta_k_initial:.4e}"
    )


bouton_play.on_clicked(
    play_pause
)

bouton_avance.on_clicked(
    avancer
)

bouton_retour.on_clicked(
    reculer
)

bouton_reset.on_clicked(
    reset
)


# =========================================================
# TIMER DE L'ANIMATION
# =========================================================

# 80 ms = vitesse de rafraîchissement réelle
# de l'interface.
#
# dt_fs correspond au temps PHYSIQUE simulé.

timer = fig.canvas.new_timer(
    interval=80
)


def animation():

    global lecture

    if not lecture:
        return


    nouveau_t = (
        slider_temps.val
        + dt_fs
    )


    # Arrêt lorsque la fin est atteinte

    if nouveau_t > t_max_fs:

        lecture = False

        bouton_play.label.set_text(
            "Play"
        )

        return


    slider_temps.set_val(
        nouveau_t
    )


timer.add_callback(
    animation
)

timer.start()


# =========================================================
# LANCEMENT DU VISUALISEUR
# =========================================================

plt.show()