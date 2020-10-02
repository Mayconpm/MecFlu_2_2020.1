import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Constantes
    altura_1 = 2.44
    altura_2 = 1.53
    gravidade = 9.81
    diametro_1 = 6.5
    diametro_2 = 51 * 10 ** (-3)
    massa_inicial = 795 * 10 ** 3

    delta_t = 1

    arquivo = "Dados"

    tempo, zp = altura_reservatorio(
        diametro_1, diametro_2, gravidade, altura_1, altura_2, delta_t,
    )
    massa = calculo_massa(diametro_1, altura_1, zp, massa_inicial)
    xp = calculo_distancia(gravidade, altura_2, zp, diametro_1, diametro_2)
    dados = salvar_dados(tempo, massa, zp, xp, delta_t, arquivo)

    label = f"Delta = {delta_t}"
    base_dados = dados

    fig1, eixo1 = plt.subplots()
    eixo1.plot(base_dados[["Tempo(s)"]], base_dados[["Massa(kg)"]], label=label)
    eixo1.set_xlabel("Tempo(s)")
    eixo1.set_ylabel("Massa(kg)")
    eixo1.grid(True)
    eixo1.legend(loc="best")
    plt.ticklabel_format(style="sci", axis="y", scilimits=(3, 3), useMathText=True)
    plt.tight_layout()
    plt.savefig("Massa.jpg", dpi=150)

    fig2, eixo2 = plt.subplots()
    eixo2.plot(base_dados[["Tempo(s)"]], base_dados[["zp(m)"]], label=label)
    eixo2.set_xlabel("Tempo(s)")
    eixo2.set_ylabel("$z_p$(kg)")
    eixo2.grid(True)
    eixo2.legend(loc="best")
    plt.tight_layout()
    plt.savefig("zp.jpg", dpi=150)

    fig3, eixo3 = plt.subplots()
    eixo3.plot(base_dados[["Tempo(s)"]], base_dados[["xp(m)"]], label=label)
    eixo3.set_xlabel("Tempo(s)")
    eixo3.set_ylabel("$x_p$(m)")
    eixo3.grid(True)
    eixo3.legend(loc="best")
    plt.tight_layout()
    plt.savefig("xp.jpg", dpi=150)


def altura_reservatorio(
    diametro_1, diametro_2, gravidade, altura_1, altura_2, delta_t,
):
    zp = np.array([altura_1])
    tempo = np.array([0])
    k = -(np.sqrt(2 * gravidade / (1 - (diametro_2 / diametro_1) ** 4))) * (
        (diametro_2 / diametro_1) ** 2
    )
    f = lambda tempo, zp: k * np.sqrt(zp - altura_2)
    tempo, zp = runge_kutta(f, tempo, zp, altura_2, delta_t)
    return tempo, zp


def runge_kutta(f, tempo, zp, altura_2, delta_t):
    i = 0
    while True:
        k1 = f(tempo[i], zp[i])
        k2 = f(tempo[i] + delta_t / 2, zp[i] + delta_t / 2 * k1)
        k3 = f(tempo[i] + delta_t / 2, zp[i] + delta_t / 2 * k2)
        k4 = f(tempo[i] + delta_t, zp[i] + delta_t * k3)
        zp = np.append(zp, zp[i] + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        i = i + 1
        tempo = np.append(tempo, tempo[i - 1] + delta_t)
        if np.round(zp[i], decimals=2) <= altura_2:
            break
    return tempo, zp


def calculo_massa(diametro_1, altura_1, zp, massa_inicial):
    area = np.pi * diametro_1 ** 2 / 4
    densidade_agua = 998
    delta_z = altura_1 - zp
    delta_massa = delta_z * area * densidade_agua
    massa = massa_inicial - delta_massa
    return massa


def calculo_distancia(gravidade, altura_2, zp, diametro_1, diametro_2):
    velocidade_2 = np.sqrt((zp - altura_2)) * (
        np.sqrt(2 * gravidade / (1 - (diametro_2 / diametro_1) ** 4))
    )
    tempo_queda_livre = np.sqrt(2 * altura_2 / gravidade)
    distancia = velocidade_2 * tempo_queda_livre
    return distancia


def salvar_dados(tempo, massa, zp, xp, delta_t, arquivo):
    dados = pd.DataFrame(
        {"Tempo(s)": tempo, "Massa(kg)": massa, "zp(m)": zp, "xp(m)": xp}
    )
    dados_resumidos = dados[dados["Tempo(s)"].round(1) % 200 == 0]
    with pd.ExcelWriter(f"{arquivo}.xlsx") as writer:
        dados.to_excel(writer, index=False, sheet_name=f"delta_{delta_t}")
        dados_resumidos.to_excel(
            writer, index=False, sheet_name=f"resumidos_delta_{delta_t}"
        )
    return dados


main()
