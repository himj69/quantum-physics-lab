import numpy as np

from physics.constants import m_e, h, hbar, eV


# Conversion eV -> J

def energie_ev_vers_joule(E_eV):
    return E_eV * eV


# Impulsion d'une particule non relativiste
#
# Ec = p² / (2m)
# p = sqrt(2mEc)

def impulsion(E, m):
    return np.sqrt(2 * m * E)


# Longueur d'onde de de Broglie
#
# lambda = h / p

def longueur_onde_de_broglie(p):
    return h / p


# Nombre d'onde
#
# k = p / hbar

def nombre_onde(p):
    return p / hbar


# Vitesse non relativiste
#
# p = mv
# v = p / m

def vitesse(p, m):
    return p / m


# Relation de dispersion d'une particule libre
#
# omega(k) = hbar * k² / (2m)

def omega_libre(k, m):
    return hbar * k**2 / (2 * m)


# Distribution des amplitudes dans l'espace des k
#
# Pour l'instant :
# A(k) = 1 dans l'intervalle autour de k0
# A(k) = 0 ailleurs

def amplitude_constante_k(k, k0, delta_k, val_cste):
    if k0 - delta_k / 2 <= k <= k0 + delta_k / 2:
        return val_cste
    return 0.0


# Onde plane
#
# phi_k(x,t) = exp(i(kx - omega(k)t))

def onde_plane_k(k, x, t, m):
    return np.exp(
        1j * (
            k * x
            - omega_libre(k, m) * t
        )
    )


# Paquet d'onde
#
# psi(x,t) = ∫ A(k) exp(i(kx - omega(k)t)) dk

def psi(x, t, k0, delta_k, m, nombre_points=1000):

    k_values = np.linspace(
        k0 - delta_k / 2,
        k0 + delta_k / 2,
        nombre_points
    )

    dk = k_values[1] - k_values[0]

    resultat = np.zeros_like(x, dtype=complex)

    for k in k_values:
        resultat += (
            amplitude_constante_k(k, k0, delta_k, 1.0)
            * onde_plane_k(k, x, t, m)
            * dk
        )

    return resultat