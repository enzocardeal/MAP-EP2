# Exercicio 1 - Teste
# MAP3122 - Métodos Numéricos e Aplicações
# Resolvido por:
# Enzo Cardeal Neves NUSP - 11257522
# Guilherme Mariano S.F NUSP - 11257539

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# Valores globais para Runge-Kutta 4
A = np.array([[-2,-1,-1,-2],[1,-2,2,-1],[-1,-2,-2,-1],[2,-1,1,-2]])

# Valores globais para euler
t = np.linspace(1.1, 3.0, 5000)
x = t**2 + 1/(1-t)

# Funcoes Runge Kutta
def funcao_kutta4(t,x):
    funcao = np.dot(A,x)
    return funcao

def funcao_exata_kutta4(t):
    funcao = []
    funcao_1 =  np.e**(-t)*np.sin(t) + np.e**(-3*t)*np.cos(3*t)
    funcao_2 =  np.e**(-t)*np.cos(t) + np.e**(-3*t)*np.sin(3*t)
    funcao_3 = -np.e**(-t)*np.sin(t) + np.e**(-3*t)*np.cos(3*t)
    funcao_4 = -np.e**(-t)*np.cos(t) + np.e**(-3*t)*np.sin(3*t)

    funcao.extend([funcao_1,funcao_2,funcao_3,funcao_4])
    return(funcao)

# Funcoes Euler
def funcao_euler(t,x):
    return 2*t+(x-t**2)**2

def funcao_exata_euler(t):
    return t**2 + 1/(1-t)

# Metodo de Runge Kutta 4

# retorna somatoria CpKp
def funcao_phi(t,x,h):
    # coeficientes
    k1 = h*funcao_kutta4(t,x)
    k2 = h*funcao_kutta4(t+h/2,x+k1/2)
    k3 = h*funcao_kutta4(t+h/2,x+k2/2)
    k4 = h*funcao_kutta4(t+h,x+k3)

    k_total = (k1)/6+(k2)/3+(k3)/3+(k4)/6
    return(k_total)

# Realiza o algoritmo de Runge Kutta 4
def runge_kutta4(t_inicial, t_final, x_inicial, n):
    solucao = []
    solucao = np.insert(solucao, len(solucao), t_inicial)
    solucao = np.insert(solucao, len(solucao), x_inicial)

    h = (t_final - t_inicial)/n

    tn = t_inicial
    xn = x_inicial

    while tn < t_final:
        v_aux = []
        
        xn1 = xn + funcao_phi(tn,xn,h)
        xn = xn1

        tn += h
        
        v_aux.append(tn)
        v_aux.extend(xn)
        solucao = np.vstack((solucao, v_aux))
        
        
    return(solucao)

# Funcao que devolve E1,n(t) := max 1<i<4 |xi*(t) - xi(t)|
def erro_kutta4(solucao):
    erro = []
    for i in range (len(solucao)):
        dif_abs =-1
        for j in range(1,5):   
            t = solucao[i,0]
            dif_abs = max(dif_abs, abs(solucao[i,j] - funcao_exata_kutta4(t)[j-1]))
        if(i == 0):
            erro.extend([t, dif_abs])
        else:
            erro = np.vstack((erro, [t,dif_abs]))
    return(erro)
    
# Devolve a lista de erros com n igual a 20,40,80,160,320 e 640
def lista_erro_rk4():
    n = [20,40,80,160,320,640]
    
    erro_1 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], n[0]))
    erro_2 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], n[1]))
    erro_3 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], n[2]))
    erro_4 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], n[3]))
    erro_5 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], n[4]))
    erro_6 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], n[5]))

    erro = [erro_1, erro_2, erro_3, erro_4, erro_5, erro_6]

    return(erro)

# Funcao que plota o erro de Runge Kutta e tambem salva
def plota_erro():
    erro = lista_erro_rk4()
    n = [20,40,80,160,320,640]
    #tamanho = len(erro)

    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=.1, hspace=.25)
    fig = plt.figure(figsize=(25, 10))
    fig.suptitle('Gráficos de $E_{1, n}(t)$', size='xx-large')

    ax1 = plt.subplot(gs[0, 0])
    ax1 = ax1.set_title('Erro para $n=20$')
    plt.plot(erro[0][:, 0], erro[0][:, 1], c='blue')
    plt.xlabel('Tempo t')
    plt.ylabel('$E_{1, n}(t)$')

    ax2 = plt.subplot(gs[0, 1])
    ax2 = ax2.set_title('Erro para $n=40$')
    plt.plot(erro[1][:, 0], erro[1][:, 1], c='blue')
    plt.xlabel('Tempo t')

    ax3 = plt.subplot(gs[0, 2])
    ax3 = ax3.set_title('Erro para $n=80$')
    plt.plot(erro[2][:, 0], erro[2][:, 1], c='blue')
    plt.xlabel('Tempo')

    ax4 = plt.subplot(gs[1, 0])
    ax4 = ax4.set_title('Erro para $n=160$')
    plt.plot(erro[3][:, 0], erro[3][:, 1], c='blue')
    plt.xlabel('Tempo t')
    plt.ylabel('$E_{1, n}(t)$')

    ax5 = plt.subplot(gs[1, 1])
    ax5 = ax5.set_title('Erro para $n=320$')
    plt.plot(erro[4][:, 0], erro[4][:, 1], c='blue')
    plt.xlabel('Tempo t')

    ax6 = plt.subplot(gs[1, 2])
    ax6 = ax6.set_title('Erro para $n=640$')
    plt.plot(erro[5][:, 0], erro[5][:, 1], c='blue')
    plt.xlabel('Tempo t')

    plt.savefig("ex_1-erro-runge-kuttan_4", dpi=300)
    print("Gráficos dos Erros de RK4 gerados. Imagem Salva!")

# Funcao que calcula e plota o R
def calcula_R_rk4():
    erro = lista_erro_rk4()
    R_i = []
    delta_n = [20, 40, 80, 160, 320]
    for i in range(5):
        div_maxima = (np.amax(erro[i], axis=0)[1])/(np.amax(erro[i+1], axis=0)[1])
        print(div_maxima)
        R_i.append(div_maxima)

    fig, ax = plt.subplots()
    ax.plot(delta_n, R_i, color='green')
    ax.set_title("$R_{i}$ em funcao de $n$")
    ax.set_xlabel('$n$')
    ax.set_ylabel('$R_{i}$')
    plt.savefig("ex_1_R_i_" +"rungekutta4_", dpi=300)
    print("Gráfico de R_i em função de n gerado. Imagem Salva!")

    

# Euler
def inv_jacob_euler(t,x, h):
    g_linha = 1-h*(2*(x-t**2))
    if g_linha != 0:
        inv_jacobiano = 1/g_linha
        return(inv_jacobiano)
    return("erro: infinito")

def g_euler(tk, x_k1, x_k, h):
    g_k1 = x_k1 - h*funcao_euler(tk, x_k1) - x_k

    return(g_k1)

def metodo_de_newton(h, tk, xk_inic):

    xk1 = xk_inic

    for l in range(7):
        Jacob_inv = inv_jacob_euler(tk, xk1, h)
        G = g_euler(tk, xk1, xk_inic, h)

        aux = xk1 - Jacob_inv*G
        xk1 = aux

    return xk1

# Metodo de Euler Implicito
def euler_implicito(t_inicial, t_final, x_inicial, n):
    h = (t_final - t_inicial)/n

    # faz a matriz solucao
    solucao = []
    solucao = np.insert(solucao, len(solucao), t_inicial)
    solucao = np.insert(solucao, len(solucao), x_inicial)

    xk = x_inicial
    tk = t_inicial + h
    
    # iteracao metodo de euler
    while tk < t_final:
        xk1 =  metodo_de_newton(h, tk,xk)

        solucao = np.vstack([solucao,[tk, xk1]])
        tk += h
        xk = xk1
        
    return(solucao)

# Funcao que calcula o erro E2(t) := |x*-x|
def erro_euler(solucao):
    erro = []
    
    for i in range (len(solucao)):
        t = solucao[i,0]
        dif_abs = abs(solucao[i,1] - funcao_exata_euler(t))

        if(i == 0):
            erro.extend([t, dif_abs])
        else:
            erro = np.vstack((erro, [t,dif_abs]))
            
    return(erro)

# Plota o grafico da solucao esperada, resolvida e do erro
def plota_grafico(solucao):
    # Calcula erro
    erro = erro_euler(solucao)

    # Cria 2x2 sub plotagem
    gs = gridspec.GridSpec(2,2)

    fig = plt.figure()
    fig.suptitle('Comparação da solução explícita e da obtida\na partir do método de Euler implícito')

    ax = plt.subplot(gs[0,0]) # linha 0, coluna 0
    ax.set_title("Esperado")
    plt.plot(t,x, c='blue')
    plt.ylabel("$x^{*}(t)$")
    plt.xlabel("Tempo t")

    ax = plt.subplot(gs[0,1]) # linha 0, coluna 1
    ax.set_title("Resolvido por Euler")
    plt.plot(solucao[:, 0], solucao[:, 1], c='orange')
    plt.ylabel("$x(t)$")
    plt.xlabel("Tempo t")

    ax = plt.subplot(gs[1, :])  # linha 1, toda a coluna
    ax.set_title("Erro")
    plt.plot(erro[:,0], erro[:,1], c='green')
    plt.ylabel("$E_{2}(t)$")
    plt.xlabel("Tempo t")

    fig.tight_layout()
    plt.savefig("ex_1_solucao_" +"euler_implicito", dpi=300)
    print("Gráficos comparando solução exata e solução calculada gerados. Imagem Salva!")

#Cabeçalho
print("                Escola Politécnica da Universidade de São Paulo")
print("               MAP3122 - Métodos numéricos para resolução de EDOs")
print("                                     Exercício 1")
print("---------------------------------------------------------------------------------------------------\n\n")

# Runge Kutta 4
print('Gerando gráfico de E1,n.')
plota_erro()
print('Gráficos gerados e salvados na pasta raíz do programa.\n')
print('Gerando R_i em função do n.')
calcula_R_rk4()
print('Gráfico gerado e salvado na pasta raíz do programa.\n')
# euler
print('Gerando gráfico solução a partir do método de Euler implícito')
solucao_euler = euler_implicito(1.1, 3.0, -8.79, 5000)
print('Plotando gráficos comparativos da solução exata com a solução a partir de Euler implícito')
plota_grafico(solucao_euler)
print("Gráficos gerados e salvados na pasta raíz do programa.\n\n")

print("Finalizado. Conferir pasta raiz para arquivos dos gráficos gerados.")
