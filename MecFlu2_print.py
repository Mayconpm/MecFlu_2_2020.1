import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    g = 9.81
    d2 = 51 * 10 ** (-3)
    d1 = 6.5
    massa_inicial = 795 * 10 ** 3

    z1 = 2.44
    z2 = 1.53

    delta_t = 0.1

    with pd.ExcelWriter("Dados.xlsx") as writer:
        dados = gerar_dados(d1, d2, g, massa_inicial, z1, z2, delta_t, writer)

    label = f"Delta = {delta_t}"
    base_dados = dados

    fig, ax = plt.subplots()
    ax.plot(base_dados[["Tempo(s)"]], base_dados[["Massa(kg)"]], label=label)
    ax.set_xlabel("Tempo(s)")
    ax.set_ylabel("Massa(kg)")
    ax.grid(True)
    ax.legend(loc="best")
    plt.ticklabel_format(style="sci", axis="y", useMathText=True)
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("Massa.jpg", dpi=150)

    fig, ax = plt.subplots()
    ax.plot(base_dados[["Tempo(s)"]], base_dados[["zp(m)"]], label=label)
    ax.set_xlabel("Tempo(s)")
    ax.set_ylabel("$z_p$(kg)")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("zp.jpg", dpi=150)

    fig, ax = plt.subplots()
    ax.plot(base_dados[["Tempo(s)"]], base_dados[["xp(m)"]], label=label)
    ax.set_xlabel("Tempo(s)")
    ax.set_ylabel("$x_p$(m)")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("xp.jpg", dpi=150)
    input("")


def gerar_dados(d1, d2, g, massa_inicial, z1, z2, delta_t, writer):
    t, zp = altura_reservatorio(d1, d2, g, z1, z2, delta_t)
    massa = calculo_massa(d1, z1, zp, massa_inicial)
    xp = calculo_distancia(g, z2, zp, d1, d2)
    dados = salvar_dados(t, massa, zp, xp, delta_t, writer)
    return dados


def altura_reservatorio(d1, d2, g, z1, z2, delta_t):
    zp = np.array([z1])
    t = np.array([0])
    k = -(np.sqrt(2 * g / (1 - (d2 / d1) ** 4))) * ((d2 / d1) ** 2)
    f = lambda t, zp: k * np.sqrt(zp - z2)
    t, zp = runge_kutta(f, t, zp, z2, delta_t)
    return t, zp


def runge_kutta(f, t, zp, z2, delta_t):
    i = 0
    while True:
        k1 = f(t[i], zp[i])
        k2 = f(t[i] + delta_t / 2, zp[i] + delta_t / 2 * k1)
        k3 = f(t[i] + delta_t / 2, zp[i] + delta_t / 2 * k2)
        k4 = f(t[i] + delta_t, zp[i] + delta_t * k3)
        zp = np.append(zp, zp[i] + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        i = i + 1
        t = np.append(t, t[i - 1] + delta_t)
        if np.round(zp[i], decimals=2) <= z2:
            break
    return t, zp


def calculo_massa(d1, z1, zp, massa_inicial):
    area = np.pi * d1 ** 2 / 4
    densidade_agua = 998
    delta_z = z1 - zp
    delta_massa = delta_z * area * densidade_agua
    massa = massa_inicial - delta_massa
    return massa


def calculo_distancia(g, z2, zp, d1, d2):
    velocidade_2 = np.sqrt((zp - z2)) * (np.sqrt(2 * g / (1 - (d2 / d1) ** 4)))
    tempo_queda_livre = np.sqrt(2 * z2 / g)
    distancia = velocidade_2 * tempo_queda_livre
    return distancia


def salvar_dados(t, massa, zp, xp, delta_t, writer):
    dados = pd.DataFrame({"Tempo(s)": t, "Massa(kg)": massa, "zp(m)": zp, "xp(m)": xp})
    dados_resumidos = dados[dados["Tempo(s)"].round(1) % 200 == 0]
    dados.to_excel(writer, index=False, sheet_name=f"delta_{delta_t}")
    dados_resumidos.to_excel(
        writer, index=False, sheet_name=f"resumidos_delta_{delta_t}"
    )
    return dados


main()
