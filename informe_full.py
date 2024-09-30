#Librerias para el análisis de los archivos
import datetime
import matplotlib.pyplot as plt
import obspy.signal
import numpy as np
import pandas as pd
import scipy.fftpack 
from numpy.core.fromnumeric import size
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy import signal
from matplotlib.dates import DateFormatter
from obspy import read
from obspy import Trace, Stream
from obspy.core import UTCDateTime
from obspy.signal.detrend import polynomial
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import StrMethodFormatter
from mpl_point_clicker import clicker
import tkinter as tk
from tkinter import filedialog
#Librerias para el reporte
from reportlab.platypus import (SimpleDocTemplate, Spacer)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table
from reportlab.platypus import Image
from reportlab.platypus import TableStyle
from reportlab.lib import colors
###################################################################

############################################################################################
#Análisis
#Variables
# enterInt= True         #activar o desactivar ingreso de tiempos
sensibilidad= 512000     #sensibilidad del sensor (cuentas/g)
gtom2= 9800              #9800 for mm/s2 or 9.8 for m/s2
scaleGlobal = True       #para escoger la escala en que se grafican los 3 canales
                         #True para que fije la escala al canal con los min y max - False para que se auto ajuste la escala en cada canal
scale = 'local'
unidadesA = 'mm/s2' if gtom2 == 9800 else 'm/s2'
unidadesV = 'mm/s' if gtom2 == 9800 else 'm/s'
fCorte= 0.5              #frecuencia de corte para el filtro
ordenFiltro= 4           #orden del filtro
tipoFiltro= 'highpass'   #tipo de filtro
gravedades= []           #lista de gravedades
ac = []                  #lista de aceleraciones
velocidad= []            #lista para almacenar los valores calculados de velociad
despla= []               #lista para almacenar los valores calculados de desplazamiento
color=['r','g','b','c','m','y','violet','k','brown']
minG, maxG, minAcc, maxAcc, minVel, maxVel = [], [], [], [], [], []

#Carga de archivos miniseed
#Carga de archivos
root = tk.Tk()
root.withdraw()
fname = filedialog.askopenfilenames(filetypes=[("Cargar Archivos","*.*")])

#Lectura de archivos
var_prom1=0
for file in fname:
    try:
        if var_prom1 == 0:
            st = read(file)
        else:
            st += read(file)
        var_prom1 +=1
        print(file)
    except:
        print("Please enter a valid file name")
# #Selecciono la ventana del evento  
# st.trim(UTCDateTime("2021-10-05T14:38:00"),UTCDateTime("2021-10-05T14:39:00"))

st.merge(fill_value='interpolate')  #Para unir los espacios de las trazas del stream 

lfinal=[]
for i in range(len(st)):        #calculo el canal con el menor número de muestras para que todos queden del mismo tamaño
    lfinal.append(len(st[i]))
    # print(lfinal)
    if i != 0:
        if lfinal[i]<lfinal[i-1]:
            final= lfinal[i]
        else:
            final= len(st[i])
    else:
        final= len(st[i])
# print(final)
# print(st[0].times("utcdatetime"))

# klicker= []
# plotting the points
fig1= plt.figure("Forma de Onda - cuentas")  #creo una figura con su título
fig1.subplots_adjust(hspace=5, wspace=0.5)       #configuro algunos parámetros
for i in range(len(st)):
    ax = fig1.add_subplot(len(st), 1, (i+1))             #agrego una gráfica, (número de fila, columna y componente)
    ax.plot(st[i].times()[0:final], st[i][0:final])
    klicker=(clicker(ax, ["event"], markers=["x"])) #guardo las dos marcas
    fig1.tight_layout()     # Ajustar el espaciado entre subplots para que no se superpongan etiquetas
    # fig1.autofmt_xdate()
    if i == (len(st)-1):
        ax.set_xlabel('Muestra')
    ax.set_ylabel(st[i].stats.channel, fontsize='small')
    if i==0:
        ax.set_title('Gráfico de datos [cuentas/g]')
    # ax2 = fig1.add_subplot(len(st)+1, 1, (i+2)) 
    # ax2.plot(st[i].times("matplotlib")[0:final], st[i][0:final], 'r-')
    # ax2.xaxis_date()
    # ax2.set_xlabel('Hora')
plt.draw()
plt.show()

######################################################
#Busco los tiempos marcados en las gráficas en sus respectivas trazas para hallar las cuentas y posteriormente calcular el promedio
pos= []
x= klicker.get_positions()
indexitos= []
for i in range(len(x["event"])):
    indexitos.append(np.where(st[2].times() == round(x["event"][i][0],0)))
    if i == 1:
        pos.append(int(indexitos[i-1][0][0]))
        pos.append(int(indexitos[i][0][0]))
dti= st[2].times("utcdatetime")[pos[0]]
dtf= st[2].times("utcdatetime")[pos[1]]
# print(dti, dtf)
st.trim(dti, dtf)
if len(st)<3:
    print("el intervalo de tiempo [",dti, dtf,"] no es el mismo en todos los canales")

st.detrend("linear")    # Se elimina la línea base de las señales

#Calculo la frecuencia de muestreo de los datos
ts= st[0].stats.delta
fs= st[0].stats.sampling_rate

#Cálculo de las gravedades
gravedades=st.copy()
for i in gravedades:
    i.normalize(512000)
    minG.append(min(i))
    maxG.append(max(i))
####Figura de gravedades
fig= plt.figure("Forma de Onda - gravedades")  #creo una figura con su título
# fig1.subplots_adjust(hspace=0.5, wspace=0.5)       #configuro algunos parámetros
fig.set_size_inches(13, 8)
for i in range(len(gravedades)):
    ax = fig.add_subplot(len(gravedades), 1, (i+1))             #agrego una gráfica, (número de fila, columna y componente)
    ax.plot(gravedades[i].times("matplotlib"), gravedades[i].data, color=color[i])
    ax.xaxis_date()
    if scaleGlobal:
        ax.set_ylim(min(minG), max(maxG))
        scale= 'global'
#     fig.autofmt_xdate()
    if i == (len(gravedades)-1):
        ax.set_xlabel('Hora')
    ax.set_ylabel(gravedades[i].stats.channel, fontsize='small')
    if i==0:
        ax.set_title('Gráfico de gravedades [g] (' + str(gravedades[i].times("utcdatetime")[0]) + ') - escala: ' + scale)
plt.savefig('./images/g' + '-escala_' + scale + '.png')

# #Cálculo de las aceleraciones
ac = gravedades.copy()
for i in ac:
    i.normalize(1/gtom2)
    minAcc.append(min(i))
    maxAcc.append(max(i))
#######Figura de aceleraciones
fig1= plt.figure("Forma de Onda - Aceleraciones")  #creo una figura con su título
# fig1.subplots_adjust(hspace=0.5, wspace=0.5)       #configuro algunos parámetros
fig1.set_size_inches(13, 8)
for i in range(len(ac)):
    ax = fig1.add_subplot(len(ac), 1, (i+1))             #agrego una gráfica, (número de fila, columna y componente)
    ax.plot(ac[i].times("matplotlib"), ac[i].data, color=color[i])
    ax.xaxis_date()
    if scaleGlobal:
        ax.set_ylim(min(minAcc), max(maxAcc))
        scale= 'global'
#     fig1.autofmt_xdate()
    if i == (len(st)-1):
        ax.set_xlabel('Hora')
    ax.set_ylabel(ac[i].stats.channel, fontsize='small')
    if i==0:
        ax.set_title('Gráfico de Aceleraciones [' + unidadesA + '] (' + str(ac[i].times("utcdatetime")[0]) + ') - escala: ' + scale)
plt.savefig('./images/accel' + '-escala_' + scale + '.png')

#Cálculo de las velocidades
#Copio los datos para luego integrarlos, ya que al integrar modifico permanentemente las trazas - stream
#Integro la copia de los datos 
velocidad= ac.copy()
velocidad.integrate(method='cumtrapz')
velocidad.filter(tipoFiltro, freq= fCorte, corners= ordenFiltro)
for i in velocidad:
        minVel.append(min(i))
        maxVel.append(max(i))
#####Figura de velocidades
fig2= plt.figure("Forma de Onda - Velocidades")  #creo una figura con su título
# fig1.subplots_adjust(hspace=0.5, wspace=0.5)       #configuro algunos parámetros
fig2.set_size_inches(13, 8)
for i in range(len(velocidad)):
    ax = fig2.add_subplot(len(st), 1, (i+1))             #agrego una gráfica, (número de fila, columna y componente)
    ax.plot(velocidad[i].times("matplotlib"), velocidad[i].data, color=color[i])
    ax.xaxis_date()
    if scaleGlobal:
            ax.set_ylim(min(minVel), max(maxVel))
            scale= 'global'
#     fig2.autofmt_xdate()
    if i == (len(velocidad)-1):
        ax.set_xlabel('Hora')
    ax.set_ylabel(velocidad[i].stats.channel, fontsize='small')
    if i==0:
        ax.set_title('Gráfico de Velocidades [' + unidadesV + '] (' + str(st[i].times("utcdatetime")[0]) + ')- escala: ' + scale)
plt.savefig('./images/vel' + '-escala_' + scale + '.png')
plt.close('all')

n= len(velocidad[0].times())          #tamaño de lista de datos, n muestras

#####Cálculo del valor RMS de Velocidad
w = int(fs); #width of the window for computing RMS
steps = int(n/w); #Number of steps for RMS
t_RMS= []
x_RMS= []
xf= []
yf= []
# f= np.zeros((3,1))
# t2= np.zeros((3,1))
# Sxx= np.zeros((3,1))
for i in range(len(velocidad)):
    t_RMS.append(np.zeros((steps,1))); #Create array for RMS time values
    x_RMS.append(np.zeros((steps,1))); #Create array for RMS values
    for j in range (0, steps):
        t_RMS[i][j] = np.mean(velocidad[i].times()[(j*w):((j+1)*w)])
        x_RMS[i][j] = np.sqrt(np.mean(velocidad[i][(j*w):((j+1)*w)]**2))
# #Cálculo de la FFT
    xf.append(np.linspace(0.0, 1.0/(2.0*ts), n//2))
    yf.append(fft(velocidad[i]))
####Figuras de RMS y FFT de la velocidad
fig4= plt.figure("Análisis - Velocidades (valores RMS)")  #creo una figura con su título
# fig1.subplots_adjust(hspace=0.5, wspace=0.5)       #configuro algunos parámetros
fig4.set_size_inches(13, 8)
for i in range(len(velocidad)):
    ax = fig4.add_subplot(len(st), 1, (i+1))             #agrego una gráfica, (número de fila, columna y componente)
    ax.plot(t_RMS[i], x_RMS[i], color=color[i])
#     ax.xaxis_date()
#     fig1.autofmt_xdate()
    if i == (len(velocidad)-1):
        ax.set_xlabel('Tiempo (muestreo)')
    ax.set_ylabel(velocidad[i].stats.channel, fontsize='small')
    if i==0:
        ax.set_title('Gráfico del RMS de Velocidades')
plt.savefig('./images/rms_velocidad.png')
        
fig5= plt.figure("Análisis - Velocidades (FFT)")  #creo una figura con su título
# fig1.subplots_adjust(hspace=0.5, wspace=0.5)       #configuro algunos parámetros
fig5.set_size_inches(13, 8)
for i in range(len(velocidad)):
    ax = fig5.add_subplot(len(st), 1, (i+1))             #agrego una gráfica, (número de fila, columna y componente)
    yfPlot= 2.0/n * np.abs(yf[i][0:n//2])
    sizeyf= len(yfPlot)//8
    ax.plot(xf[i][0:sizeyf], yfPlot[0:sizeyf], color=color[i])
#     ax.xaxis_date()
#     fig1.autofmt_xdate()
    if i == (len(velocidad)-1):
        ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel(velocidad[i].stats.channel, fontsize='small')
    if i==0:
        ax.set_title('Gráfico de la FFT de Velocidades')
plt.savefig('./images/FFT_velocidad.png')

#########Calculo de desplazamiento
#Copio nuevamente los datos, para integrarlos y hallar el desplazamiento
despla= velocidad.copy()
#Integro para calcular el desplazamiento
despla.integrate(method='cumtrapz')
# despla.detrend("linear")
despla.filter(tipoFiltro, freq= fCorte, corners= ordenFiltro)

#Cálculo de picos
picosVel=[]
tPicosVel=[]
fPicosVel=[]
sizeWin= int(fs*2)         #tamaño de la ventana para recorrer la lista de datos
paso= 0.5                           #para definir el paso del reocrrido de la ventana por la señal    
window= np.hanning(sizeWin)    #se define la ventana de Hann
for vel in velocidad:
    i= 0                    #variable para iterar la lista de datos de ventana en ventana
    velPico= []             #lista para almacenar los picos máximos de velocidad en cada ventana
#     jaux= 0                 #var aux para iterar las listas de Picos y ayudar a calcular los tiempos de cada Pico
#     indexPicoVel= []        #
    tVelPico= []
    amplPicoFFTVel= []
    fVelPico= []
    while(i+sizeWin<len(vel)):
#         jaux2= 0    #var aux que indica la posión del pico de velocidad
        velPico.append(abs(max(vel[i:i+sizeWin], key=abs)))   #cálculo del máximo valor en la ventana
        timeAux= vel.times("utcdatetime")[i:i+sizeWin]  #lista para guardar los instantes de tiempo de cada ventana
        indexPicost= np.where(vel[i:i+sizeWin] == max(vel[i:i+sizeWin], key=abs))
#         print('index picos 1:',indexPicost[0][0])
        tVelPico.append(timeAux[indexPicost[0][0]])
#         for j in vel[i:i+sizeWin]:  #se recorre la ventana de datos de las Velocidades
#             if abs(j) == velPico[jaux]:   #se busca donde coincidan con el pico de Velocidades
# #                 indexPicoVel.append(jaux2)             #guardo las posiciones que coincidan con el pico de Velocidades
# #                 print(jaux2)
#                 tPicoVel.append(timeAux[jaux2])   #guardo el instante de tiempo del pico
#             jaux2 += 1
#         jaux += 1
        velmod= vel[i:i+sizeWin]*window
        fttVelmod= fft(velmod)
        fttVelmodAmpl= abs(fttVelmod[:len(fttVelmod)//2])  #Calculo la mitad de las amplitudes de toda la lista 
        amplPicoFFTVel.append(max(fttVelmodAmpl))          
        fVelmo= scipy.fftpack.fftfreq(len(velmod),d=ts)    #Se calculan las frecuencias
        fVelmod= fVelmo[:len(fVelmo)//2]                   #tomo solo las frecuencias negativas
        indexPicosf= np.where(fttVelmodAmpl == max(fttVelmodAmpl))
#         print('index picos 2:',indexPicosf[0][0])
        fVelPico.append(fVelmod[indexPicosf[0][0]])
#         vel[i:i+sizeWin].taper(0,type='hann')      # Solo funciona con la traza o stream completos

#         print(velPico, fVelPico)
        i= i+int(sizeWin*paso)     #se incrementa la variale de iteración
        
#     print((fVelmod[:len(fVelmod)//2]))
#     print((fttVelmod))
#     print(len(fVelmod), len(fttVelmod))
    picosVel.append(velPico)
    tPicosVel.append(tVelPico)
    fPicosVel.append(fVelPico)

print(len(picosVel))
print(picosVel)
print(max(picosVel[0]),max(picosVel[1]),max(picosVel[2]))
idx= np.where(picosVel[0] == max(picosVel[0], key=abs))
idx1= np.where(picosVel[1] == max(picosVel[1], key=abs))
idx2= np.where(picosVel[2] == max(picosVel[2], key=abs))
# print(tPicosVel[0][idx[0][0]],tPicosVel[1][idx1[0][0]],tPicosVel[2][idx2[0][0]])   
print(fPicosVel[0][idx[0][0]],fPicosVel[1][idx1[0][0]],fPicosVel[2][idx2[0][0]])

#Gráfica de Norma DIN
fig, axs = plt.subplots(1)
fig.suptitle("Análisis DIN 4150", fontsize='large', fontweight='bold')
# fig.set_size_inches(15, 8)
colors=["k*","ks","ko"]
textos=["Vertical","Transversal","Radial"]
for i in range(len(picosVel)):
    axs.plot(fPicosVel[i],picosVel[i],colors[i],label=velocidad[i].stats.channel)

axs.plot([1,10,50,100],[20,20,40,50],"g",linestyle='-',label="Tolerancia 3")
axs.plot([1,10,50,100],[5,5,15,20],"b",linestyle='--',label="Tolerancia 2")
axs.plot([1,10,50,100],[3,3,8,10],"r",linestyle=':',label="Tolerancia 1")
#axs[1].plot(ev[1].times("matplotlib"),ev[1].data,"g-")
#axs[2].plot(ev[2].times("matplotlib"),ev[2].data,"b-")
#ax.xaxis_date()
#fig.autofmt_xdate()
#for ax in fig.get_axes():
#    ax.xaxis_date()
#    ax.label_outer()
axs.set_xscale('log', base=2)
axs.set_yscale('log', base=10)

axs.legend(loc='center left',bbox_to_anchor=(0.75, 0.75))
plt.xlim([1,100])
plt.xlabel("Frecuencia (Hz)", fontsize='xx-large', fontweight='bold')
plt.ylabel("Velocidad (mm / s)", fontsize='xx-large', fontweight='bold')

plt.ylim([0.01,200])
for axis in [axs.xaxis, axs.yaxis]:
    #axis.set_major_formatter(ScalarFormatter())
    axis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
# plt.rcParams["font.size"] = "20"
plt.savefig('./images/normaDIN.png')

# print(abs(max(despla[0], key=abs)),abs(max(despla[1], key=abs)),abs(max(despla[2], key=abs)))

def guardar_tabla_en_excel(datos, nombre_archivo):
    df = pd.DataFrame(datos[1:], columns=datos[0])
    df.to_excel(nombre_archivo, index=False)
datos = [
        ["-", str(velocidad[0].stats.channel), str(velocidad[1].stats.channel), str(velocidad[2].stats.channel)],
        ["Velocidades pico[mm/s]:", str(max(picosVel[0])), str(max(picosVel[1])), str(max(picosVel[2]))],
        ["f pico: ", str(fPicosVel[0][idx[0][0]]), str(fPicosVel[1][idx1[0][0]]), str(fPicosVel[2][idx2[0][0]])],
        ["Gravedades pico[g]:", str(abs(max(gravedades[0], key=abs))), str(abs(max(gravedades[1], key=abs))), str(abs(max(gravedades[2], key=abs)))],
        ["Aceleraciones pico[mm/s2]:", str(abs(max(ac[0], key=abs))), str(abs(max(ac[1], key=abs))), str(abs(max(ac[2], key=abs)))],
        ["Desplazamientos Pico[mm]:", str(abs(max(despla[0], key=abs))), str(abs(max(despla[1], key=abs))), str(abs(max(despla[2], key=abs)))]
    ]
nombre_archivo_excel = "./images/resumenDatos.xlsx"
guardar_tabla_en_excel(datos, nombre_archivo_excel)
print(f"La tabla se ha guardado en el archivo '{nombre_archivo_excel}'")

###################################################################################################################################
#Informe
w, h = letter  #(612.0, 792.0)

fileName = 'Reporte.pdf'
pdf = SimpleDocTemplate(fileName,pagesize=letter)

def genPinTable():

    pinElemTable = None
    pinElemWidth = 600

    # Variables para editar la  tabla de la info del proyecto
    varNameArch= ''
    varDate= ''
    varNameProject= ''
    varClient= ''
    varUbicacion= ''
    varNotas= ''

    # Variables para editar la tabla resumen
    varCH= [velocidad[0].stats.channel, velocidad[1].stats.channel, velocidad[2].stats.channel]
    varPPV= [format(max(picosVel[0]), '.3E'),format(max(picosVel[1]), '.3E'),format(max(picosVel[2]), '.3E')]
    varFreq= [format(fPicosVel[0][idx[0][0]], '.1E'), format(fPicosVel[1][idx1[0][0]], '.1E'), format(fPicosVel[2][idx2[0][0]], '.1E')]
    # varTime=['','',''] # [format(no1,'.1E')(tPicosVel[0][idx[0][0]], 2),  format(no1,'.1E')(tPicosVel[1][idx1[0][0]], 2), format(no1,'.1E')(tPicosVel[2][idx2[0][0]], 2)]
    varPPA= [format(abs(max(gravedades[0], key=abs)), '.3E'), format(abs(max(gravedades[1], key=abs)), '.3E'), format(abs(max(gravedades[2], key=abs)), '.3E')]
    varPPD= [format(abs(max(despla[0], key=abs)), '.3E'), format(abs(max(despla[1], key=abs)), '.3E'), format(abs(max(despla[2], key=abs)), '.3E')]

    # Variable para almacenar el gárfico de la norma
    graficaNorma= Image('./images/normaDIN.png')
    graficaNorma.drawWidth = 230
    graficaNorma.drawHeight = 190

    # Variables para tabla info de SSI
    logoSSI= Image('./images/LogoSSI.png')
    logoSSI.drawWidth = 0.4
    logoSSI.drawWidth = 0.4
    varRefSensor= 'Waleker SMA'
    varSerialSensor= 'ME-0'+velocidad[0].stats.station
    varCalibrate= ''
    varOperator= ''

    # Variable para tabla de forma de ondas de Velocidades
    graficaOndas= Image('./images/vel' + '-escala_' + scale + '.png')
    graficaOndas.drawWidth = 520
    graficaOndas.drawHeight = 210

    # Variable para tabla de gráficas RMS
    graficaRMS= Image('./images/rms_velocidad.png')
    graficaRMS.drawWidth = 260
    graficaRMS.drawHeight = 220
    
    # Variable para tabla de gráficas FFT
    graficaFFT= Image('./images/FFT_velocidad.png')
    graficaFFT.drawWidth = 260
    graficaFFT.drawHeight = 220

    # Variable para tabla de forma de ondas de Gravedades
    graficaG= Image('./images/g' + '-escala_' + scale + '.png')
    graficaG.drawWidth = 260
    graficaG.drawHeight = 220

    # Variable para tabla de forma de ondas de Aceleraciones 
    graficaAcc= Image('./images/accel' + '-escala_' + scale + '.png')
    graficaAcc.drawWidth = 260
    graficaAcc.drawHeight = 220


    # 1) Build Structure
#######TABLA TITULO#######################################################################################################
    # Tabla para el título del informe
    tableTile = Table([
        ['Análisis de Vibraciones']
    ], pinElemWidth)

#######TABLA1######################################################################################################
    # Tabla para la información del Proyecto
    tableNameArch= Table([
        ['Nombre del Archivo: ', varNameArch]
    ], [125, 125])

    tableDate= Table([
        ['Fecha y Hora: ', varDate]
    ], [125, 125])

    tableNameProject= Table([
        ['Nombre del Proyecto: ', varNameProject]
    ], [125, 125])

    tableClient= Table([
        ['Cliente: ', varClient]
    ], [125, 125])

    tableUbicacion= Table([
        ['Ubicación: ', varUbicacion]
    ], [125, 125])

    tableNotas= Table([
        ['Notas: ', varNotas]
    ], [125, 125])

    tableInfoProject= Table([
        [tableNameArch],
        [tableDate],
        [tableNameProject],
        [tableClient],
        [tableUbicacion],
        [tableNotas]
    ])

    # Tabla para el resumen del análisis
    tableCH= Table([
        ['Peak Measure', varCH[0], varCH[1], varCH[2]]
    ], [70, 80, 80, 80])

    tablePPV= Table([
        ['PPV (mm/s)', varPPV[0], varPPV[1], varPPV[2]]
    ], [70, 80, 80, 80])

    tableFreq= Table([
        ['f (Hz)', varFreq[0], varFreq[1], varFreq[2]]
    ], [70, 80, 80, 80])

    # tableTime= Table([
    #     ['Tiempo (ms)', varTime[0], varTime[1], varTime[2]]
    # ], [70, 80, 80, 80])

    tablePPA= Table([
        ['PPA (g)', varPPA[0], varPPA[1], varPPA[2]]
    ], [70, 80, 80, 80])

    tablePPD= Table([
        ['PPD (mm)', varPPD[0], varPPD[1], varPPD[2]]
    ], [70, 80, 80, 80])

    tableResumen= Table([
        [tableCH],
        [tablePPV],
        [tableFreq],
        # [tableTime],
        [tablePPA],
        [tablePPD],
    ])

    # Tabla 1
    table1= Table([
        [tableInfoProject, tableResumen]
    ], [250, 280])

#####TABLA2##############################################################################################################
    # Tabla para el gráfico de la norma
    tableGrafNorma= Table([
        [graficaNorma] #graficaNorma
    ], 300)

    # Tabla Logo SSI
    tableLogo= Table([
        [logoSSI] #
    ],0.9)

    # Tabla info sensor
    tableRefSensor= Table([
        ['Referencia del sensor: ', varRefSensor]
    ], [150, 150])

    tableSerialSensor= Table([
        ['Serial del sensor: ', varSerialSensor]
    ], [150, 150])

    tableCalibrate= Table([
        ['Operador(es): ', varOperator]
    ], [150, 150])

    tableOperator= Table([
        ['Fecha de Calibración: ', varCalibrate]
    ], [150, 150])

    # Tabla info SSI
    tableInfoSSI= Table([
        [tableLogo],
        [tableRefSensor],
        [tableSerialSensor],
        [tableOperator],
        [tableCalibrate],
    ])

    # Tabla 2
    table2= Table([
        [tableGrafNorma, tableInfoSSI]
    ],[300, 300])

####TABLA 3##############################################################################################################
    # Tabla ondas Velocidad
    tableOndas= Table([
        [graficaOndas] 
    ], 500)

    # Tabla 3
    table3= Table([
        [tableOndas]
    ],500)

####TABLA 4##############################################################################################################
    # Tabla ondas RMS
    tableRMS= Table([
        [graficaRMS] 
    ], 300)

    # Tabla ondas FFT
    tableFFT= Table([
        [graficaFFT] 
    ], 300)

    # Tabla 4
    table4= Table([
        [tableRMS, tableFFT]
    ],[300,300])

####TABLA 5##############################################################################################################
    # Tabla ondas Gravedad
    tableG= Table([
        [graficaG] 
    ], 300)

    # Tabla ondas Aceleración
    tableAcc= Table([
        [graficaAcc] 
    ], 300)

    # Tabla 5
    table5= Table([
        [tableG, tableAcc]
    ],[300,300])

#######################################################################################################################
    # # Creo la tabla que contendrá cada una de las tablas de la estructura del informe
    # pinElemTable= Table([
    #     [tableTile],
    #     [table1]
    # ], pinElemWidth)

    # 2) Add Style
    # Estilo de la tabla de título
    titleTableStyle = TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 18),
        ('FONTNAME', (0,0), (-1,-1), 
            'Helvetica-Bold'
            ), 
        ('TOPPADDING',(0,0),(-1,-1), 2),
        ('BOTTOMPADDING',(0,0),(-1,-1), 2),
        ('VALIGN', (0,0), (-1, -1), 'MIDDLE'), 
    ])
    tableTile.setStyle(titleTableStyle)

    infoProjectTableStyle = TableStyle([
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('TOPPADDING',(0,0),(-1,-1), 0),
        ('BOTTOMPADDING',(0,0),(-1,-1), 0),
        ('VALIGN', (0,0), (-1, -1), 'LEFT'),
    ]) 
    tableInfoProject.setStyle(infoProjectTableStyle)

    resumTableStyle = TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        # ('BOX',(0,0),(-1,-1),1,colors.black),
        # ('GRID',(0,0),(-1,-1),1,colors.black),
        ('TOPPADDING',(0,0),(-1,-1), 0),
        ('BOTTOMPADDING',(0,0),(-1,-1), 0),
        ('LEFTPADDING',(0,0),(-1,-1), 5),
        ('RIGHTPADDING',(0,0),(-1,-1), 5),
        ('VALIGN', (0,0), (-1, -1), 'LEFT'),
    ]) 
    tableResumen.setStyle(resumTableStyle)

    celdasResumTableStyle = TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        # ('BOX',(0,0),(-1,-1),1,colors.black),
        # ('GRID',(0,0),(-1,-1),1,colors.black),
    ]) 
    tableCH.setStyle(celdasResumTableStyle)
    tablePPV.setStyle(celdasResumTableStyle)
    tableFreq.setStyle(celdasResumTableStyle)
    # tableTime.setStyle(celdasResumTableStyle)
    tablePPA.setStyle(celdasResumTableStyle)
    tablePPD.setStyle(celdasResumTableStyle)

    table1TableStyle = TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('LEFTPADDING',(0,0),(-1,-1), 0),
        ('RIGHTPADDING',(0,0),(-1,-1), 5),
    ]) 
    table1.setStyle(table1TableStyle)

    # table2TableStyle = TableStyle([
    #     ('ALIGN',(0,0),(-1,-1),'CENTER'),
    #     ('LEFTPADDING',(0,0),(-1,-1), 0),
    #     ('RIGHTPADDING',(0,0),(-1,-1), 0),
    # ]) 
    # table2.setStyle(table2TableStyle)

    grafNormaTableStyle = TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('LEFTPADDING',(0,0),(-1,-1), 15),
        ('RIGHTPADDING',(0,0),(-1,-1), 0),
        ('VALIGN', (0,0), (-1, -1), 'CENTER'),
    ]) 
    tableGrafNorma.setStyle(grafNormaTableStyle)
    
    infoSSITableStyle = TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('LEFTPADDING',(0,0),(-1,-1), 0),
        ('RIGHTPADDING',(0,0),(-1,-1), 0),
        ('VALIGN', (0,0), (-1, -1), 'CENTER'),
    ]) 
    tableInfoSSI.setStyle(infoSSITableStyle)
    tableLogo.setStyle(infoSSITableStyle)

    ondasTableStyle = TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('LEFTPADDING',(0,0),(-1,-1), 0),
        ('RIGHTPADDING',(0,0),(-1,-1), 0),
        ('VALIGN', (0,0), (-1, -1), 'CENTER'),
    ]) 
    tableOndas.setStyle(ondasTableStyle)

    RMSFFTTableStyle = TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('LEFTPADDING',(0,0),(-1,-1), 0),
        ('RIGHTPADDING',(0,0),(-1,-1), 0),
        ('VALIGN', (0,0), (-1, -1), 'CENTER'),
    ]) 
    tableRMS.setStyle(RMSFFTTableStyle)
    tableFFT.setStyle(RMSFFTTableStyle)

    GAccTableStyle = TableStyle([
    ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ('LEFTPADDING',(0,0),(-1,-1), 0),
    ('RIGHTPADDING',(0,0),(-1,-1), 0),
    ('VALIGN', (0,0), (-1, -1), 'CENTER'),
    ]) 
    tableG.setStyle(GAccTableStyle)
    tableAcc.setStyle(GAccTableStyle)

    return tableTile, table1, table2, table3, table4, table5

tableInforme= genPinTable()

# mainTable = Table([
#     [tableInforme]
# ])

elems = []
elems.append(tableInforme[0])
elems.append(Spacer(0, 20))
elems.append(tableInforme[1])
elems.append(Spacer(0, 10))
elems.append(tableInforme[2])
elems.append(Spacer(0, 0.5))
elems.append(tableInforme[3])
elems.append(Spacer(0, 0.5))
elems.append(tableInforme[4])
elems.append(Spacer(0, 0.5))
elems.append(tableInforme[5])
pdf.build(elems)