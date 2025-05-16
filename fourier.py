import numpy as np
import matplotlib.pyplot as plt

# === 1. Cargar datos desde archivo ===
# El archivo tiene datos en columnas: t, ..., vspwm, ..., vout
data = np.loadtxt('ngspice_output.dat')

# === 2. Extraer columnas relevantes ===
# Ajusta índices según tu archivo; asumimos:
# t = columna 0
# vspwm = columna 7
# vout = columna 9

t = data[:, 0]
vspwm = data[:, 7]
vout = data[:, 9]

# === 3. Graficar señales en el tiempo ===
plt.figure(figsize=(10, 5))
plt.plot(t * 1e3, vspwm, 'r', label='Vspwm (PWM)')
plt.plot(t * 1e3, vout, 'b', label='Vout (filtrada)')
plt.title('Señales en dominio del tiempo')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === 4. Calcular y graficar FFT ===
def plot_fft(signal, label, color):
    N = len(signal)
    Fs = 1 / (t[1] - t[0])  # Frecuencia de muestreo
    fft_vals = np.fft.fft(signal)
    fft_mag = np.abs(fft_vals) / N
    fft_mag = fft_mag[:N//2] * 2  # Solo la mitad positiva
    freqs = np.fft.fftfreq(N, d=1/Fs)[:N//2]
    fft_db = 20 * np.log10(fft_mag + 1e-12)
    plt.plot(freqs / 1000, fft_db, label=label, color=color)

plt.figure(figsize=(10, 5))
plot_fft(vspwm, 'FFT Vspwm', 'red')
plot_fft(vout, 'FFT Vout', 'blue')
plt.title('Transformada de Fourier (FFT) en escala dB')
plt.xlabel('Frecuencia (kHz)')
plt.ylabel('Magnitud (dB)')
plt.xlim(0, 10)  # Zoom en bajas frecuencias
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
