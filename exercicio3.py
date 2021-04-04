# Exercicio 3 - Modelo duas presas-um predador
# MAP3122 - Métodos Numéricos e Aplicações
# Resolvido por:
# Enzo Cardeal Neves NUSP - 11257522
# Guilherme Mariano S.F NUSP - 11257539

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# Matriz A
def matriz_A(alpha):
    A = np.array([[0.001, 0.001, 0.015],[0.0015, 0.001, 0.001],
                  [-alpha, -0.0005, 0]])
    return(A)

# matriz B
def matriz_B():
    B =[1,1,-1]
    return B

# Funcoes
def x_linha(x,y,z, A, B):
    x_linha = x*(B[0] - A[0][0]*x - A[0][1]*y- A[0][2]*z)
    return x_linha

def y_linha(x,y,z, A, B):
    y_linha = y*(B[1] - A[1][0]*x - A[1][1]*y - A[1][2]*z)
    return y_linha

def z_linha(x, y, z, A, B):
    z_linha = z*(B[2] - A[2][0]*x - A[2][1]*y)
    return z_linha
    

# Metodo de Runge Kutta 4
def funcao_kutta4(ui, alpha):
    A = matriz_A(alpha)
    B = matriz_B()
    x = x_linha(ui[0],ui[1], ui[2], A,B)
    y = y_linha(ui[0],ui[1], ui[2], A, B)
    z = z_linha(ui[0],ui[1], ui[2], A, B)
    funcao = np.array([x, y, z])

    return funcao
    

# retorna somatoria CpKp
def funcao_phi(u,h, alpha):
    # coeficientes
    k1 = funcao_kutta4(u, alpha)
    k2 = funcao_kutta4(u+h/2*k1, alpha)
    k3 = funcao_kutta4(u+h/2*k2, alpha)
    k4 = funcao_kutta4(u+h*k3, alpha)

    k_total = (k1)/6+(k2)/3+(k3)/3+(k4)/6

    return(k_total)

# Realiza o algoritmo de Runge Kutta 4
def runge_kutta4(t_inicial, t_final, condicoes_iniciais, alpha):
    h = (t_final - t_inicial)/5000
    solucao = []
    solucao = np.insert(solucao, len(solucao), t_inicial)
    solucao = np.insert(solucao, len(solucao), condicoes_iniciais[0])
    solucao = np.insert(solucao, len(solucao), condicoes_iniciais[1])
    solucao = np.insert(solucao, len(solucao), condicoes_iniciais[2])

    tn = t_inicial
    u = condicoes_iniciais

    while tn < t_final:
        v_aux = []
        
        # Calcula a interacao atual
        soma = h*funcao_phi(u, h, alpha)
        u1 = u + soma
        # Atualiza as variaveis
        u = u1
        tn += h
        
        # Coloca na solucao
        v_aux.append(tn)
        v_aux.extend(u1)
        solucao = np.vstack((solucao, v_aux))
                
    return(solucao)

# Metodo de Euler Implicito
def euler_explicito(t_inicial, t_final, condicoes_iniciais, alpha):
    h=(t_final - t_inicial)/5000
    
    # faz a matriz solucao
    solucao = []
    solucao = np.insert(solucao, len(solucao), t_inicial)
    solucao = np.insert(solucao, len(solucao), condicoes_iniciais[0])
    solucao = np.insert(solucao, len(solucao), condicoes_iniciais[1])
    solucao = np.insert(solucao, len(solucao), condicoes_iniciais[2])

    # valores iniciais
    tn = t_inicial
    xn = condicoes_iniciais[0]
    yn = condicoes_iniciais[1]
    zn = condicoes_iniciais[2]

    # Matrizes A e B
    A = matriz_A(alpha)
    B = matriz_B()
    
    # interacao metodo de euler
    while tn < t_final:
        v_aux = []
        tn += h
        # Coelhos
        xn1 = xn + h*x_linha(xn+h, yn+h, zn+h, A, B)
        xn = xn1
        
        # Lebres
        yn1 = yn + h*y_linha(xn+h, yn+h, zn+h, A, B)
        yn = yn1
        
        # Raposas
        zn1 = zn + h*z_linha(xn+h, yn+h, zn+h, A, B)
        zn = zn1

        # Coloca na Solucao
        v_aux = [tn, xn, yn, zn]
        solucao = np.vstack([solucao,v_aux])
        
    return(solucao)

# Plota o grafico da solucao esperada, resolvida e do erro
def plota_grafico3D(solucao, alpha, algoritmo):
    # Cria as coordenadas x,y, e z
    x = solucao[:,1]
    y = solucao[:,2]
    z = solucao[:,3]

    # Faz a plotagem
    fig = plt.figure()
    ax =plt.axes(projection="3d")

    fig.suptitle('Retrato de fase 3D com alpha=' + str(alpha))

    ax.plot(x,y,z, label='retrato de fase')
    ax.legend()
    
    ax.set_xlabel('Coelhos')
    ax.set_ylabel('Lebres')
    ax.set_zlabel('Raposas')
    
    plt.savefig("retrato3d_" +algoritmo+ "_" + str(alpha) + ".jpg",bbox_inches='tight')
    print("Imagem Salva!")
    plt.show()
    
def plota_grafico2D(solucao, alpha, algoritmo):
    # Cria as coordenadas x,y, e z
    x = solucao[:,1]
    y = solucao[:,2]
    z = solucao[:,3]
    
    # Cria plotagem 2x2
    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure()
    fig.suptitle('Retrato de fase 2D com alpha=' + str(alpha))
    
    ax = plt.subplot(gs[0, 0]) # linha 0, coluna 0
    ax.set_title('Coelhos X Lebres')
    plt.plot(x,y)

    ax = plt.subplot(gs[0, 1]) # linha 0, coluna 1
    ax.set_title('Coelhos X Raposas')
    plt.plot(x,z)

    ax = plt.subplot(gs[1, :]) # linha 1, toda a coluna
    ax.set_title('Lebres X Raposas')
    plt.plot(y,z)

    fig.tight_layout()
    plt.savefig("retrato2d_" +algoritmo+ "_" + str(alpha) + ".jpg",bbox_inches='tight')
    print("Imagem Salva!")
    plt.show()

def plota_tamanho(solucao, alpha, algoritmo):
    # Cria as coordenadas
    t = solucao[:,0]
    x = solucao[:,1]
    y = solucao[:,2]
    z = solucao[:,3]

    # Cria plotagem 2x2
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    fig.suptitle('Tamanho da população com alpha=' + str(alpha))
    
    ax = plt.subplot(gs[0, 0]) 
    ax.set_title('Coelhos X tempo')
    plt.plot(t, x)

    ax = plt.subplot(gs[0, 1])
    ax.set_title('Lebres X tempo')
    plt.plot(t, y, color="black")

    ax = plt.subplot(gs[1, :])
    ax.set_title('Raposas X tempo')
    plt.plot(t,z, 'tab:red')
    
    fig.tight_layout()
    plt.savefig("populacao_" +algoritmo+ "_" + str(alpha) + ".jpg",bbox_inches='tight')
    print("Imagem Salva!")
    plt.show()

# Euler
alpha = [0.001, 0.002, 0.0033, 0.0036, 0.005, 0.0055]
to = 0
tf = [100, 500, 2000]
condicoes_iniciais = [500, 500, 10]
for i in range(len(alpha)):
    solucao = euler_explicito(to, tf[int(i/2)], condicoes_iniciais, alpha[i])
    plota_grafico3D(solucao, alpha[i], "Euler")
    plota_grafico2D(solucao, alpha[i], "Euler")
    plota_tamanho(solucao, alpha[i], "Euler")

# Teste de sensibilidade
alpha =  0.005
condicoes_iniciais = [[35,75, 137],[37,74,137]]
tf = 400
for i in range(len(condicoes_iniciais)):
    solucao = euler_explicito(to, tf, condicoes_iniciais[i], alpha)
    plota_grafico3D(solucao, alpha, "Euler_sensib" + str(i))
    plota_grafico2D(solucao, alpha, "Euler_sensib" + str(i))
    plota_tamanho(solucao, alpha, "Euler_sensib" + str(i))
    
# Runge Kutta 4
alpha = [0.001, 0.002, 0.0033, 0.0036, 0.005, 0.0055]
to = 0
tf = [100, 500, 2000]
condicoes_iniciais = [500, 500, 10]
for i in range(len(alpha)):
    solucao = runge_kutta4(to, tf[int(i/2)], condicoes_iniciais, alpha[i])
    plota_grafico3D(solucao, alpha[i], "RK4")
    plota_grafico2D(solucao, alpha[i], "RK4")
    plota_tamanho(solucao, alpha[i], "RK4")

# Teste de sensibilidade
alpha =  0.005
condicoes_iniciais = [[35,75, 137],[37,74,137]]
tf = 400
for i in range(len(condicoes_iniciais)):
    solucao = runge_kutta4(to, tf, condicoes_iniciais[i], alpha)
    plota_grafico3D(solucao, alpha, "RK4_sensib" + str(i))
    plota_grafico2D(solucao, alpha, "RK4_sensib" + str(i))
    plota_tamanho(solucao, alpha, "RK4_sensib" + str(i))
