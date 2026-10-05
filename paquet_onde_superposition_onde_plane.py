import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Constantes physiques
# -----------------------------

m_e = 9.1093837e-31
h = 6.62607015e-34
hbar = h / (2 * np.pi)
eV = 1.602176634e-19


# -----------------------------
# Fonctions physiques
# -----------------------------

# ---------------------------------------------------------
# Conversion énergie : électron-volt -> joule
#
# Relation :
# 1 eV = 1.602176634 × 10^-19 J
#
# E_eV : énergie exprimée en électron-volts
# retourne : énergie en joules
# ---------------------------------------------------------

def energie_ev_vers_joule(E_eV):
    return E_eV * eV


# ---------------------------------------------------------
# Impulsion d'une particule non relativiste
#
# Énergie cinétique :
# Ec = p² / (2m)
#
# Donc :
# p = sqrt(2mEc)
#
# E : énergie cinétique en joules
# m : masse de la particule en kg
# retourne : impulsion p en kg.m/s
# ---------------------------------------------------------

def impulsion(E, m):
    return np.sqrt(2 * m * E)


# ---------------------------------------------------------
# Longueur d'onde de de Broglie
#
# Relation de de Broglie :
# lambda = h / p
#
# p : impulsion de la particule
# h : constante de Planck
#
# retourne : longueur d'onde en mètres
# ---------------------------------------------------------

def longueur_onde_de_broglie(p):
    return h / p


# ---------------------------------------------------------
# Nombre d'onde k
#
# Définition :
# k = 2π / lambda
#
# Avec de Broglie :
# lambda = h / p
#
# et :
# hbar = h / (2π)
#
# Donc :
# k = p / hbar
#
# p : impulsion
# retourne : nombre d'onde en m^-1
# ---------------------------------------------------------

def nombre_onde(p):
    return p / hbar


# ---------------------------------------------------------
# Vitesse d'une particule non relativiste
#
# Relation classique :
# p = m v
#
# Donc :
# v = p / m
#
# p : impulsion
# m : masse
#
# retourne : vitesse en m/s
# ---------------------------------------------------------

def vitesse(p, m):
    return p / m


# ---------------------------------------------------------
# Relation de dispersion d'une particule libre
#
# Pour une particule libre non relativiste :
#
# E = p² / (2m)
#
# avec :
# p = hbar * k
#
# donc :
# E = hbar² k² / (2m)
#
# Et comme :
# E = hbar * omega
#
# alors :
# omega(k) = hbar * k² / (2m)
#
# k : nombre d'onde
# m : masse de la particule
#
# retourne : pulsation omega en rad/s
# ---------------------------------------------------------

def omega_libre(k, m):
    return hbar * k**2 / (2 * m)


# ---------------------------------------------------------
# Distribution des amplitudes dans l'espace des k
#
# Un paquet d'onde est une superposition d'ondes planes :
#
# psi(x) = ∫ A(k) exp(i k x) dk
#
# A(k) représente le poids de chaque composante k
# dans le paquet.
#
# Pour l'instant, on choisit une distribution uniforme :
#
# A(k) = 1 si k appartient à :
#
# [k0 - delta_k ; k0 + delta_k]
#
# A(k) = 0 ailleurs
#
# k0       : nombre d'onde central
# delta_k  : largeur de la distribution en k
# ---------------------------------------------------------

def amplitude_k(k, k0, delta_k):
    if k0 - delta_k/2 <= k <= k0 + delta_k/2:
        return 1.0
    return 0.0


# ---------------------------------------------------------
# Onde plane associée à un nombre d'onde k
#
# À t = 0 :
#
# phi_k(x) = exp(i k x)
#
# Plus généralement :
#
# phi_k(x,t) = exp(i(kx - omega t))
#
# Ici on ne considère pour l'instant que t = 0.
#
# k : nombre d'onde
# x : position
#
# retourne : amplitude complexe de l'onde plane
# ---------------------------------------------------------

def onde_plane_k(k, x, t):
    return np.exp(1j * (k * x - omega_libre(k, m_e) * t))


# -----------------------------
# Données du problème
# -----------------------------

Ec_eV = 100
Ec = energie_ev_vers_joule(Ec_eV)

p0 = impulsion(Ec, m_e)
lambda0 = longueur_onde_de_broglie(p0)
k0 = nombre_onde(p0)
v0 = vitesse(p0, m_e)

delta_k = 0.01 * k0


# -----------------------------
# Fonction d'onde à t = 0
# -----------------------------

def psi_t0(x, k0, delta_k, nombre_points=1000):
    k_values = np.linspace(
        k0 - delta_k/2,
        k0 + delta_k/2,
        nombre_points
    )

    dk = k_values[1] - k_values[0]

    psi = 0j

    for k in k_values:
        psi += amplitude_k(k, k0, delta_k) * onde_plane_k(k, x, 0) * dk

    return psi

# ------------------------------
# Affichage du paquet d'onde t=0
# ------------------------------

# Positions où l'on observe la fonction d'onde
x = np.linspace(-20e-9, 20e-9, 5000)

# Calcul du paquet à t = 0
psi_0 = np.array([psi_t0(x_i, k0, delta_k) for x_i in x])

# Densité associée à la fonction d'onde
densite = np.abs(psi_0)**2

# Conversion de x en nanomètres pour rendre le graphique lisible
x_nm = x * 1e9

plt.plot(x_nm, densite)

plt.xlabel("Position x (nm)")
plt.ylabel("|ψ(x,0)|²")
plt.title("Paquet d'onde de l'électron à t = 0")
plt.grid()

plt.show()