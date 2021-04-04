# importando dependencias necessárias
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# ------------------ Definindo funções --------------------------
# calcula a inversa do Jacobiano da F do exercício 2
def ex2_jacobiana_inv (h, u, lamb, alpha, beta, gamma):
    # Jacabiano calculado de acordo com o enunciado do exercício
    J = np.array([[1-h*(lamb - alpha*u[1]), h*alpha*u[0]],
                   [-h*beta*u[1], 1-h*(beta*u[0]-gamma)]], dtype='f')
    
    J_inv = np.linalg.inv(J)
    
    return J_inv

# calcula a F de acordo com os parâmetros do exercício
def ex2_f_func (u, lamb, alpha, beta, gamma):
    F = np.array([lamb*u[0]-alpha*u[0]*u[1], beta*u[0]*u[1]-gamma*u[1]], dtype='f')
    
    return F

# a partir da F, calcula o G de acordo com os parâmetros do exercício
def ex2_g_func (h, uk, uk1, lamb, alpha, beta, gamma):
    F = np.array([lamb*uk1[0]-alpha*uk1[0]*uk1[1], beta*uk1[0]*uk1[1]-gamma*uk1[1]], dtype='f')
    
    Gk1 = uk1-h*F-uk
    
    return Gk1 

# algorítmo do método de Newton
def ex2_metodo_newton (n, h, uk1_l0, lamb, alpha, beta, gamma):
    
    uk1_l = uk1_l0
    
    for l in range(n):
        J_inv = ex2_jacobiana_inv(h, uk1_l, lamb, alpha, beta, gamma)
        G = ex2_g_func(h, uk1_l0, uk1_l, lamb, alpha, beta, gamma) # se considerarmos uk1_l0 = uk
        
        # calculando uk1 seguinte
        uk1_l1 = uk1_l - np.dot(J_inv, G)
        uk1_l = uk1_l1
        
    return uk1_l1

# algorítmo do método de Euler explícito de acordo com os slides da disciplina
def ex2_euler_exp (T0, Tf, n, u0, lamb, alpha, beta, gamma, linspc):
    arr_euler_exp = np.array([u0], dtype='f')
    
    h = (Tf-T0)/n
    
    uk=u0
    
    # linspc é um parâmetro utilizado para determinar o número de pontos a serem plotados
    divisor = int(n/linspc)
    
    if(divisor < 1): #linspc deve ser maior do que o número de passos tomados no método de newton
        print('A quantidade de pontos a serem plotados deve ser menor ou igual ao número de passos')
        exit()
    
    for k in range(n):
        uk1 = uk + h*ex2_f_func(uk, lamb, alpha, beta, gamma)
        uk = uk1
        
        if(k%divisor == 0): # condicional que seleciona os pontos a serem plotados
            arr_euler_exp = np.vstack([arr_euler_exp, uk])
            
    return arr_euler_exp

# algorítmo do método de Euler implícito
def ex2_euler_imp (T0, Tf, n, n_l, u0, lamb, alpha, beta, gamma, linspc):
    arr_euler_imp = np.array([u0], dtype='f')
    
    h = (Tf-T0)/n
        
    uk = u0
        
    divisor = int(n/linspc)
        
    if(divisor < 1):
        print('A quantidade de pontos a serem plotados deve ser menor ou igual ao número de passos')
        exit()        
        
    for k in range(n):
        # utilização do método de newton de acordo com o enunciado do exercício
        uk1 = ex2_metodo_newton(n_l, h, uk, lamb, alpha, beta, gamma)
        uk = uk1
        
        if(k%divisor == 0):
            arr_euler_imp = np.vstack([arr_euler_imp, uk])
        
    return arr_euler_imp

# calculando Ex e Ey de acordo com o enunciado do exercício
def ex2_erro_euler (T0, Tf, n, n_l, u0, lamb, alpha, beta, gamma):
    
    # chama Euler explícito para n passos
    arr_euler_exp = ex2_euler_exp(T0, Tf, n, u0, lamb, alpha, beta, gamma, n)
    # chama Euler implícito para n passos
    arr_euler_imp = ex2_euler_imp(T0, Tf, n, n_l, u0, lamb, alpha, beta, gamma, n)
    
    #calcula a diferença entre os resultados do Euler implícito e explícito
    arr_erro = arr_euler_imp - arr_euler_exp
    
    return arr_erro

# algorítimo do método de Runge Kuttan 4 de acordo com o material da disciplina
def ex2_runge_kuttan4 (T0, Tf, n, u0, lamb, alpha, beta, gamma, linspc):
    
    # definindo os Ck' s
    C1=1/6
    C2=1/3
    C3=1/3
    C4=1/6
    
    arr_rk4 = np.array([u0], dtype='f')
    
    h = (Tf-T0)/n
    passo = np.array([h, h], dtype='f')
    
    ui=u0
    
    divisor = int(n/linspc)
    
    if(divisor < 1):
        print('A quantidade de pontos a serem plotados deve ser menor ou igual ao número de passos')
        exit()     

    # calculando os K' s
    for i in range(n):
        K1=ex2_f_func(ui, lamb, alpha, beta, gamma)
        K2=ex2_f_func(ui+passo*K1/2, lamb, alpha, beta, gamma)
        K3=ex2_f_func(ui+passo*K2/2, lamb, alpha, beta, gamma)
        K4=ex2_f_func(ui+passo*K3, lamb, alpha, beta, gamma)
        
        soma = C1*K1 + C2*K2 + C3*K3 + C4*K4
        
        # calculando o u seguinte
        ui1=ui + passo*soma
        ui=ui1
        
        if(i%divisor == 0):
            arr_rk4 = np.vstack([arr_rk4, ui])
        
    return arr_rk4   
# ---------------------------------------------------------------

#Cabeçalho
print("                Escola Politécnica da Universidade de São Paulo")
print("               MAP3122 - Métodos num´ericos para resolução de EDOs")
print("                                     Exercício 2")
print("---------------------------------------------------------------------------------------------------\n\n")

# -----------Resolvendo as questões 2.1, 2.2 e 2.4---------------
print("Parâmetros inicializados de acordo com o enunciado\n")
print("Calculando a variação das populações do modelo presa-predador com método de Euler explícito e n=10000")
u0 = np.array([1.5, 1.5], dtype='f')
R_exp = ex2_euler_exp(0, 10, 10000, u0, 2/3, 4/3, 1, 1, 1000)
print("Processado. Valores de u_exp(t) salvados.\n")

print("Calculando a variação das populações do modelo presa-predador com método de Euler implícito, n=5000 e método de Newton com n_l=7")
R_imp = ex2_euler_imp(0, 10, 5000, 7, u0, 2/3, 4/3, 1, 1, 1000)
print("Processado. Valores de u_imp(t) salvados.\n")

print("Calculando a variação das populações do modelo presa-predador com método de Runge Kuttan 4 e n=1000")
R_rk4 = ex2_runge_kuttan4(0, 10, 1000, u0, 2/3, 4/3, 1, 1, 1000)
print("Processado. Valores de u_rk(t) salvados.\n\n")

print("Gerando imagem com os gráficos correspondentes de cada método.")
gs = gridspec.GridSpec(2,3)
gs.update(wspace=.1, hspace=.25)
fig = plt.figure(figsize=(25, 10))
fig.suptitle('Comparando resultados de cada método', size='xx-large')
t = np.arange(0, 10.0000000001, 10/1000)

ax1 = plt.subplot(gs[0, 0])
ax1.set_title('Raposa x Coelho - Euler explícito n=10000')
plt.plot(R_exp[:,0], R_exp[:,1], c='blue')
plt.xlabel('População de coelhos')
plt.ylabel('População de raposas')

ax2 = plt.subplot(gs[0, 1])
ax2.set_title('Pop. raposas x Pop. coelhos - Euler implícito n=5000')
plt.plot(R_imp[:,0], R_imp[:,1], color='blue')
plt.xlabel('População de coelhos')

ax3 = plt.subplot(gs[0, 2])
ax3.set_title('Pop. raposas x Pop. coelhos - Runge Kuttan 4 n=1000')
plt.plot(R_rk4[:,0], R_rk4[:,1], color='blue')
plt.xlabel('População de coelhos')

ax4 = plt.subplot(gs[1, 0])
ax4.set_title('População x Tempo - Euler explícito n=10000')
plt.plot(t, R_exp[:,0], color='blue', label='Coelhos')
plt.plot(t, R_exp[:,1], color='red', label='Raposas')
plt.xlabel('Tempo')
plt.ylabel('População')
plt.legend()

ax5 = plt.subplot(gs[1, 1])
ax5.set_title('População x Tempo - Euler impícito n=5000')
plt.plot(t, R_imp[:,0], color='blue', label='Coelhos')
plt.plot(t, R_imp[:,1], color='red', label='Raposas')
plt.xlabel('Tempo')
plt.legend()

ax6 = plt.subplot(gs[1, 2])
ax6.set_title('População x Tempo - Runge Kuttan n=1000')
plt.plot(t, R_rk4[:,0], color='blue', label='Coelhos')
plt.plot(t, R_rk4[:,1], color='red', label='Raposas')
plt.xlabel('Tempo')
plt.legend()

plt.savefig('ex_2-compara-metodos-numericos', dpi=300)
print("Imagem gerada. Confira a pasta raiz do programa.")
print("---------------------------------------------------------------------------------------------------\n\n")
# ---------------------------------------------------------------

# ---------------- Resolvendo a questão 2.3 ---------------------
print("Calculando Ex e Ey para os diferentes valores de n pedidos.")
print("Para n=250.")
erro250 = ex2_erro_euler(0, 10, 250, 7, u0, 2/3, 4/3, 1, 1)
print("erro250 salvado.\n")

print("Para n=500.")
erro500 = ex2_erro_euler(0, 10, 500, 7, u0, 2/3, 4/3, 1, 1)
print("erro500 salvado.\n")

print("Para n=1000.")
erro1000 = ex2_erro_euler(0, 10, 1000, 7, u0, 2/3, 4/3, 1, 1)
print("erro1000 salvado.\n")

print("Para n=2000.")
erro2000 = ex2_erro_euler(0, 10, 2000, 7, u0, 2/3, 4/3, 1, 1)
print("erro2000 salvado.\n")

print("Para n=4000.")
erro4000 = ex2_erro_euler(0, 10, 4000, 7, u0, 2/3, 4/3, 1, 1)
print("erro4000 salvado.\n\n")

print("Gerando imagem com os gráficos correspondentes de cada n.")
gs = gridspec.GridSpec(3, 4)
gs.update(wspace=.2, hspace=.25)
fig = plt.figure(figsize=(18, 20))
fig.suptitle('Comparando Erros', size='xx-large')

t = np.arange(0, 10.0000000001, 10/250)
ax1 = plt.subplot(gs[0, 0:2])
ax1.set_title('n=250')
plt.plot(t, erro250[:,0], color='blue', label='$E_{x}$')
plt.plot(t, erro250[:,1], color='red', label='$E_{y}$')
plt.xlabel('Tempo')
plt.ylabel('Erro')
plt.legend()

t = np.arange(0, 10.0000000001, 10/500)
ax2 = plt.subplot(gs[0, 2:])
ax2.set_title('n=500')
plt.plot(t, erro500[:,0], color='blue', label='$E_{x}$')
plt.plot(t, erro500[:,1], color='red', label='$E_{y}$')
plt.xlabel('Tempo')
plt.legend()

t = np.arange(0, 10.0000000001, 10/1000)
ax3 = plt.subplot(gs[1, 0:2])
ax3.set_title('n=1000')
plt.plot(t, erro1000[:,0], color='blue', label='$E_{x}$')
plt.plot(t, erro1000[:,1], color='red', label='$E_{y}$')
plt.xlabel('Tempo')
plt.ylabel('Erro')
plt.legend()

t = np.arange(0, 10.0000000001, 10/2000)
ax4 = plt.subplot(gs[1, 2:])
ax4.set_title('n=2000')
plt.plot(t, erro2000[:,0], color='blue', label='$E_{x}$')
plt.plot(t, erro2000[:,1], color='red', label='$E_{y}$')
plt.xlabel('Tempo')
plt.legend()

t = np.arange(0, 10.0000000001, 10/4000)
ax5 = plt.subplot(gs[2, 1:3])
ax5.set_title('n=4000')
plt.plot(t, erro4000[:,0], color='blue', label='$E_{x}$')
plt.plot(t, erro4000[:,1], color='red', label='$E_{y}$')
plt.xlabel('Tempo')
plt.ylabel('Erro')
plt.legend()

plt.savefig('ex_2-compara-erros', dpi=300)
print("Imagem gerada. Confira a pasta raiz do programa.\n")
# ---------------------------------------------------------------
print("Fim da execução.")