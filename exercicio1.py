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
    tamanho = len(erro)
    
    for i in range(tamanho):
        funcao = erro[i]
        fig, ax = plt.subplots()
        ax.plot(funcao[:,0], funcao[:,1], 'tab:purple')
        ax.set_title("Erro com n igual a " + str(n[i]))

        plt.savefig("solucao_" +"rungekutta4_" + str(i)
                    + ".jpg",bbox_inches='tight')
        print("Imagem Salva!")

        plt.show()

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
    ax.set_title("$R_{i}$ em funcao de $\Delta n$")
    ax.set_xlabel('$\Delta n$')
    ax.set_ylabel('$R_{i}$')
    plt.savefig("R_i_" +"rungekutta4_"+ ".jpg",bbox_inches='tight')
    print("Imagem Salva!")

    plt.show()

    

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
    
    # interacao metodo de euler
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
    fig.suptitle('Comparacao do Esperado e do resolvido por Euler')

    ax = plt.subplot(gs[0,0]) # linha 0, coluna 0
    ax.set_title("Esperado")
    plt.plot(t,x)

    ax = plt.subplot(gs[0,1]) # linha 0, coluna 1
    ax.set_title("Resolvido por Euler")
    plt.plot(solucao[:, 0], solucao[:, 1], 'tab:orange')

    ax = plt.subplot(gs[1, :])  # linha 1, toda a coluna
    ax.set_title("Erro")
    plt.plot(erro[:,0], erro[:,1], 'tab:green')

    fig.tight_layout()
    plt.savefig("solucao_" +"euler_implicito" + ".jpg",bbox_inches='tight')
    print("Imagem Salva!")
    plt.show()

# Runge Kutta 4
plota_erro()
calcula_R_rk4()
# euler
solucao_euler = euler_implicito(1.1, 3.0, -8.79, 5000)
plota_grafico(solucao_euler)


