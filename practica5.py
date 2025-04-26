import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs = 500_000  # Frecuencia de muestreo (Hz)
t = np.arange(0, 2e-3, 1/fs)  # Tiempo 0 a 2ms
audio_freq = 2000  # Frecuencia de la señal de audio (Hz)
tri_freq = 44000  # Frecuencia de la señal triangular (Hz)

# Señales
audio_signal = np.sin(2 * np.pi * audio_freq * t)
triangular_wave = 2 * (2*(t*tri_freq - np.floor(t*tri_freq + 0.5)))  # Generador triangular normalizado

# SPWM: comparador
spwm = (audio_signal > triangular_wave).astype(float)

# Integración usando método de Euler
def integrate_euler(signal, dt):
    integral = np.zeros_like(signal)
    for i in range(1, len(signal)):
        integral[i] = integral[i-1] + signal[i-1]*dt
    return integral

# Integración usando método Trapezoidal
def integrate_trapezoidal(signal, dt):
    integral = np.zeros_like(signal)
    for i in range(1, len(signal)):
        integral[i] = integral[i-1] + 0.5*(signal[i-1] + signal[i])*dt
    return integral

dt = 1/fs

# Aplicar integradores
integrated_euler = integrate_euler(spwm - 0.5, dt)  # Restamos 0.5 para centrar la señal
integrated_trap = integrate_trapezoidal(spwm - 0.5, dt)

# Normalizar para comparación
integrated_euler = integrated_euler / np.max(np.abs(integrated_euler))
integrated_trap = integrated_trap / np.max(np.abs(integrated_trap))

# Gráficas
plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(t*1000, audio_signal)
plt.title('Señal de Audio (2 kHz)')
plt.grid()

plt.subplot(4,1,2)
plt.plot(t*1000, triangular_wave)
plt.title('Señal Triangular (44 kHz)')
plt.grid()

plt.subplot(4,1,3)
plt.plot(t*1000, spwm)
plt.title('Señal SPWM (Salida del Comparador)')
plt.grid()

plt.subplot(4,1,4)
plt.plot(t*1000, integrated_euler, label="Euler")
plt.plot(t*1000, integrated_trap, label="Trapezoidal", linestyle="--")
plt.title('Señal Integrada (Filtro)')
plt.legend()
plt.grid()

plt.xlabel('Tiempo (ms)')
plt.tight_layout()
plt.show()
