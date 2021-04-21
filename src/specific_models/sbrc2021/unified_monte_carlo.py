### K: Tempos em s
import random
from random import randint
from random import seed
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

INFINITO = 10**10
TAMANHO_MODELO = 4.3 * 8 * 10**6 # <b>

ROUNDS = 250
CLIENTES = 5
NUMERO_MINIMO_CLIENTES = 5


### K: Clientes demoram de 1 ate 5 segundos para executar nas simulacoes
L_ps = [1, 2, 3, 4, 5] # <s>

### K: Velocidade de transmissao obtidos com o iperf3
L_ts = [53.2*(10**6), 51.8*(10**6), 94.0*(10**6)]  # Mb * [<b/s>]
L_ts = [TAMANHO_MODELO / x for x in L_ts] # [<b/s>] / b = <s>

print ('maxlts:', max(L_ts))
print ('maxlps:', max(L_ps))
T = 2*(2 * max (L_ts) + max (L_ps))
print ('t:', T)
#exit ()

x = []
y = []
for probabilidade_de_falha in range (0, 101):
  latencia_total = 0

  print ('P_f:', probabilidade_de_falha)
  for _ in range (ROUNDS):
    latencia_maxima_rodada = 0
    clientes_com_falha = 0
    for client in range (CLIENTES):
      L_p = random.choice (L_ps)
      L_t = random.choice (L_ts)
      L_c = L_p + L_t # processamento + transmissao

      #falha = randint (1, 10)
      #if (falha <= probabilidade_de_falha/10):
      #  clientes_com_falha += 1
      #if ( (CLIENTES - clientes_com_falha) < NUMERO_MINIMO_CLIENTES):
      #  L_c = INFINITO


      #if (client >= NUMERO_MINIMO_CLIENTES):
      #  break
      falha = random.uniform (0, 100)
      #print (falha)
      if (falha <= probabilidade_de_falha):
        L_c = INFINITO
        #print ('falhou')
      #else:
        #print ('nao falho')

      #print ('min:', min (L_c, T))
      latencia_maxima_rodada = max (latencia_maxima_rodada, min (L_c, T))

    latencia_total += latencia_maxima_rodada
    #latencia_total += latencia_maxima_rodada

  x.append (probabilidade_de_falha)
  y.append (latencia_total)
  #plt.plot(probabilidade_de_falha, latencia_total, 'm.')

  print (latencia_total)


#fig = plt.gcf()
#width = 13
#height = 8
#fig.set_size_inches(width, height, forward=True)
#plt.legend()#loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend ()
#plt.figure(figsize=(10,5))
plt.plot(x, y, 'r-', markersize=6, label='5 clientes')

x = []
y = []
CLIENTES = 10
for probabilidade_de_falha in range (0, 101):
  latencia_total = 0

  print ('P_f:', probabilidade_de_falha)
  for _ in range (ROUNDS):
    latencia_maxima_rodada = 0
    clientes_com_falha = 0
    for client in range (CLIENTES):
      L_p = random.choice (L_ps)
      L_t = random.choice (L_ts)
      L_c = L_p + L_t # processamento + transmissao

      falha = random.uniform (0, 100)
      #print (falha)
      if (falha <= probabilidade_de_falha):
        clientes_com_falha += 1
      if ( (CLIENTES - clientes_com_falha) < NUMERO_MINIMO_CLIENTES):
        L_c = INFINITO

      latencia_maxima_rodada = max (latencia_maxima_rodada, min (L_c, T))

      if (client >= NUMERO_MINIMO_CLIENTES):
        break



    latencia_total += latencia_maxima_rodada
  x.append (probabilidade_de_falha)
  y.append (latencia_total)
  #plt.plot(probabilidade_de_falha, latencia_total, 'm.')


fig = plt.gcf()
width = 10
height = 7
fig.set_size_inches(width, height, forward=True)
#plt.figure(figsize=(10,5))
plt.plot(x, y, 'b.', markersize=6, label='10 clientes, encerramento antecipado em 5 respostas')










plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
#plt.legend(loc='upper center', bbox_to_anchor=(1, 1))
plt.xlabel ('Probabilidade de falha percentual ($P_f$)')
plt.ylabel ('Latência total em segundos (250 rodadas)')
plt.xlim (0, 102)#len (accuracies) + 1)
#plt.ylim (0, 100)
plt.tight_layout()
plt.savefig ('unified_monte_carlo.pdf')
print ('saved')
