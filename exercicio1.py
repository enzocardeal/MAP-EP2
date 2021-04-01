# Exercicio 1 - Teste
# MAP3122 - Métodos Numéricos e Aplicações
# Resolvido por:
# Enzo Cardeal Neves NUSP - 11257522
# Guilherme Mariano S.F NUSP - 11257539

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Valores globais para Runge-Kutta 4
A = np.array([[-2,-1,-1,-2],[1,-2,2,-1],[-1,-2,-2,-1],[2,-1,1,-2]])

# Funcoes Runge Kutta
def funcao_kutta4(t,x):
    return np.dot(A,x)

def funcao_exata_kutta4(t):
    funcao = []
    funcao_1 =  np.e**(-t)*np.sin(t) + np.e**(-3*t)*np.cos(3*t)
    funcao_2 =  np.e**(-t)*np.cos(t) + np.e**(-3*t)*np.sin(3*t)
    funcao_3 = -np.e**(-t)*np.sin(t) + np.e**(-3*t)*np.cos(3*t)
    funcao_4 = -np.e**(-t)*np.cos(t) + np.e**(-3*t)*np.sin(3*t)

    funcao.extend([funcao_1,funcao_2,funcao_3,funcao_4])
    return(funcao)

# Valores globais para euler
t = np.linspace(1.1, 3.0, 5000)
x = t**2 + 1/(1-t)

# Funcoes Euler
def funcao_euler(t,x):
    return 2*t+(x-t**2)**2

def funcao_exata_euler(t):
    return t**2 + 1/(1-t)

# Metodo de Runge Kutta 4

# retorna somatoria CpKp
def funcao_phi(t,x,h):
    # coeficientes
    k1 = funcao_kutta4(t,x)
    k2 = funcao_kutta4(t+h/2,x+h/2*k1)
    k3 = funcao_kutta4(t+h/2,x+h/2*k2)
    k4 = funcao_kutta4(t+h,x+h*k3)

    k_total = (k1)/6+(k2)/3+(k3)/3+(k4)/6

    return(k_total)

# Realiza o algoritmo de Runge Kutta 4
def runge_kutta4(t_inicial, t_final, x_inicial, h):
    solucao = []
    solucao = np.insert(solucao, len(solucao), t_inicial)
    solucao = np.insert(solucao, len(solucao), x_inicial)

    tn = t_inicial
    xn = x_inicial

    while tn < t_final:
        v_aux = []
        
        xn1 = xn + h*funcao_phi(tn,xn,h)
        xn = xn1

        v_aux.append(tn)
        v_aux.extend(xn)
        solucao = np.vstack((solucao, v_aux))
        
        tn += h
        
    return(solucao)

# Funcao que devolve E1,n(t) := max 1<i<4 |xi*(t) - xi(t)|
def erro_kutta4(solucao):
    erro = []
    guarda_valor = 0
    guarda_j = 0

    # Procura o maior erro entre as funcoes
    for j in range (1,5):
        for i in range(len(solucao)):
            t = solucao[i,0]
            dif_abs = abs(solucao[i,j] - funcao_exata_kutta4(t)[j-1])

        if(dif_abs > guarda_valor):
            guarda_valor = dif_abs
            guarda_j = j

    j = guarda_j
    
    # Pega a funcao com maior erro
    for i in range (len(solucao)):
        t = solucao[i,0]
        dif_abs = abs(solucao[i,j] - funcao_exata_kutta4(t)[j])

        if(i == 0):
            erro.extend([t, dif_abs])
        else:
            erro = np.vstack((erro, [t,dif_abs]))
    return(erro)
    
# Devolve a lista de erros com n igual a 20,40,80,160,320 e 640
def lista_erro_rk4():
    n = [20,40,80,160,320,640]
    h = [2/20,2/40,2/80,2/160,2/320, 2/640]
    
    erro_1 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], h[0]))
    erro_2 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], h[1]))
    erro_3 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], h[2]))
    erro_4 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], h[3]))
    erro_5 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], h[4]))
    erro_6 = erro_kutta4(runge_kutta4(0, 2, [1,1,1,-1], h[5]))

    erro = [erro_1, erro_2, erro_3, erro_4, erro_5, erro_6]

    return(erro)

# Funcao que plota o erro de Runge Kutta e tambem salva
def plota_erro():
    erro = lista_erro_rk4()
    n = [20,40,80,160,320,640]
    tamanho = len(erro)-1
    
    for i in range(tamanho):
        funcao = erro[i]
        fig, ax = plt.subplots()
        ax.plot(funcao[:,0], funcao[:,1], 'tab:purple')
        ax.set_title("Erro com n igual a " + str(n[i]))

        plt.savefig("solucao_" +"rungekutta4_" + str(i)
                    + ".jpg",bbox_inches='tight')
        print("Imagem Salva!")

        plt.show()

# Funcao que calcula o R 
def calcula_R_rk4():
    erro = lista_erro_rk4()
    
    for i in range(5):
        R_i = (np.amax(erro[i], axis=1)[1])/(np.amax(erro[i+1], axis=1)[1])
        print(R_i)

# Metodo de Euler Implicito
def euler_implicito(t_inicial, t_final, x_inicial, h):
    # faz a matriz solucao
    solucao = []
    solucao = np.insert(solucao, len(solucao), t_inicial)
    solucao = np.insert(solucao, len(solucao), x_inicial)

    # valores iniciais
    tn = t_inicial
    xn = x_inicial
    
    # interacao metodo de euler
    while tn < t_final:
        tn += h
        xn1 = xn + h*funcao_euler(tn, xn+h)
        xn = xn1

        solucao = np.vstack([solucao,[tn, xn]])
        
    return(solucao)

# Funcao que calcula o erro E2(t) := |x*-x|*100
# Obs: o *100 é utilizado para ver alguma diferença,
# já que o metodo é bastante eficiente
def erro_euler(solucao):
    erro = []
    
    for i in range (len(solucao)):
        t = solucao[i,0]
        dif_abs = abs(solucao[i,1] - funcao_exata_euler(t))*100

        if(i == 0):
            erro.extend([t, dif_abs])
        else:
            erro = np.vstack((erro, [t,dif_abs]))
            
    return(erro)

# Plota o grafico da solucao esperada, resolvida e do erro
def plota_grafico(solucao):
    fig, (ax1,ax2, ax3) = plt.subplots(1,3,sharex='col', sharey='row')
    fig.suptitle('Comparacao do Esperado e do resolvido por Euler')
    ax1.set_title("Esperado")
    ax1.plot(t,x)

    ax2.plot(solucao[:, 0], solucao[:, 1], 'tab:orange')
    ax2.set_title("Resolvido por Euler")

    erro = erro_euler(solucao)
    
    ax3.plot(erro[:,0], erro[:,1], 'tab:green')
    ax3.set_title("Erro")

    plt.savefig("solucao_" +"euler_implicito" + ".jpg",bbox_inches='tight')
    print("Imagem Salva!")
    plt.show()

# Runge Kutta 4
plota_erro()
calcula_R_rk4()

# euler
h = (3.0-1.1)/5000
solucao_euler = euler_implicito(1.1, 3.0, -8.79, h)
plota_grafico(solucao_euler)