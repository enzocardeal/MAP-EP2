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
        # Coelhos
        xn1 = xn + h*x_linha(xn, yn, zn, A, B)
        xn = xn1
        
        # Lebres
        yn1 = yn + h*y_linha(xn, yn, zn, A, B)       
        yn = yn1
        
        # Raposas
        zn1 = zn + h*z_linha(xn, yn, zn, A, B)
        zn = zn1
        
        # Atualiza variavel
        tn += h

        # Coloca na Solucao
        v_aux = [tn, xn, yn, zn]
        solucao = np.vstack([solucao,v_aux])
        
    return(solucao)

def plota_grafico(solucao, alpha, algoritmo, i=''):
    
    print("Gerando imagem com os gráficos correspondentes de cada método.")
    # Coordenadas
    t = solucao[:,0]
    x = solucao[:,1]
    y = solucao[:,2]
    z = solucao[:,3]

    # plotagem
    gs = gridspec.GridSpec(3,4)
    gs.update(wspace=.6, hspace=.60)
    fig = plt.figure(figsize=(15,10))
    fig.suptitle('Resultado com ' + r'$\alpha$' + ' igual a ' + str(alpha)
                  + ' usando: ' + algoritmo,
                  size='xx-large')

    # Grafico 3D
    ax1 = plt.subplot(gs[0, 0:2], projection="3d")
    ax1.set_title('Retrato de fase 3D - n=5000')
    ax1.plot(x,y,z, label='retrato de fase')
    ax1.set_xlabel('Coelhos')
    ax1.set_ylabel('Lebres')
    ax1.set_zlabel('Raposas')


    # Retrato de fase 2D
    ax1 = plt.subplot(gs[0,2:])
    ax1.set_title('Retrato de fase 2D - Coelhos x Lebres n=5000')
    plt.xlabel('Coelhos')
    plt.ylabel('Lebres')
    plt.plot(x,y)

    ax = plt.subplot(gs[1, 0:2])
    ax.set_title('Retrato de fase 2D - Coelhos X Raposas n=5000')
    plt.xlabel('Coelhos')
    plt.ylabel('Raposas')
    plt.plot(x,z)

    ax = plt.subplot(gs[1, 2:])
    ax.set_title('Retrato de fase 2D - Lebres X Raposas n=5000')
    plt.xlabel('Lebres')
    plt.ylabel('Raposas')
    plt.plot(y,z)

    # Tamanho da Populacao
    ax = plt.subplot(gs[2, 1:3])
    ax.set_title('População ao longo do tempo n=5000')
    plt.plot(t, x, color='blue', label='Coelhos')
    plt.plot(t, y, color='black', label='Lebres')
    plt.plot(t, z, color='red', label='Raposas')
    plt.xlabel('Tempo')
    plt.ylabel('População')
    plt.legend()

    # Salva Figura
    plt.savefig('resultado_' + algoritmo + '_' + str(alpha) + i + '.jpg', bbox_inches='tight', dpi=300)
    print("Imagem gerada. Confira a pasta raiz do programa.")
    print("---------------------------------------------------------------------------------------------------\n\n")
    plt.show()

# ---------------------------------------------------------------
#Cabeçalho
print("                Escola Politécnica da Universidade de São Paulo")
print("               MAP3122 - Métodos num´ericos para resolução de EDOs")
print("                                     Exercício 3")
print("---------------------------------------------------------------------------------------------------\n\n")

print("Parâmetros inicializados de acordo com o enunciado\n")
alpha = [0.001, 0.002, 0.0033, 0.0036, 0.005, 0.0055]
to = 0
tf = [100, 500, 2000]
condicoes_iniciais = [500, 500, 10]
# ---------------- Resolvendo a questão 3.2 ---------------------
for i in range(len(alpha)):
    # Euler
    solucao = euler_explicito(to, tf[int(i/2)], condicoes_iniciais, alpha[i])
    plota_grafico(solucao, alpha[i], "Euler")
    
    # Runge Kutta 4
    solucao = runge_kutta4(to, tf[int(i/2)], condicoes_iniciais, alpha[i])
    plota_grafico(solucao, alpha[i], "RK4")
    
# ---------------- Resolvendo a questão 3.3 ---------------------
# Teste de sensibilidade
print("Teste de Sensibilidade")
alpha =  0.005
condicoes_iniciais = [[37,75, 137],[37,74,137]]
tf = 400
for i in range(len(condicoes_iniciais)):
    # Euler
    solucao = euler_explicito(to, tf, condicoes_iniciais[i], alpha) 
    plota_grafico(solucao, alpha, "Euler", str(i))

    # Imprime o instante final
    print("Tamanho da População para o teste de sensibilidade " + str(i+1) +" :" + " Euler")
    print(solucao[len(solucao)-1])
    print()

    # Runge Kutta 4
    solucao = runge_kutta4(to, tf, condicoes_iniciais[i], alpha)
    plota_grafico(solucao, alpha, "RK4", str(i))

    # Imprime no instante final
    print("Tamanho da População para o teste de sensibilidade " +  str(i+1) + " :"  + " RK4")
    print(solucao[len(solucao)-1])
    print()
