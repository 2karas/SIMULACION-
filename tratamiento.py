import serial
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

class SensorData:
    def __init__(self, puerto='COM6', velocidad=9600, num_muestras=100):
        self.puerto = puerto
        self.velocidad = velocidad
        self.num_muestras = num_muestras
        self.datos = []

    def leer_datos(self):
        """Leer datos desde el puerto serial y almacenar en un array."""
        with serial.Serial(self.puerto, self.velocidad) as arduino:
            while len(self.datos) < self.num_muestras:
                if arduino.in_waiting:
                    try:
                        valor = int(arduino.readline().decode().strip())
                        self.datos.append(valor)
                    except ValueError:
                        continue
        return np.array(self.datos)

class SignalProcessor:
    @staticmethod
    def calcular_autocorrelacion(señal):
        """Calcular la autocorrelación de la señal."""
        señal_centrada = señal - np.mean(señal)
        autocorr = np.correlate(señal_centrada, señal_centrada, mode='full')
        return autocorr[autocorr.size // 2:]  # Solo parte positiva

    @staticmethod
    def detectar_periodo(autocorr):
        """Detectar el periodo usando los picos de la autocorrelación."""
        picos, _ = find_peaks(autocorr, distance=5)
        if len(picos) > 1:
            periodo_muestras = picos[1]  # Primer pico después del desplazamiento 0
            return periodo_muestras
        return None

    @staticmethod
    def calcular_frecuencia(periodo_muestras, tiempo_entre_muestras=0.005):
        """Calcular la frecuencia en Hz dado el periodo en muestras."""
        periodo_segundos = periodo_muestras * tiempo_entre_muestras
        return 1 / periodo_segundos if periodo_segundos > 0 else None

def graficar_datos(muestras, autocorr):
    """Graficar las señales de la muestra y su autocorrelación."""
    plt.figure(figsize=(10, 4))
    plt.plot(muestras, label="Señal original", color='blue')
    plt.title("Señal desde Arduino")
    plt.xlabel("Muestra")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(autocorr, label="Autocorrelación", color='orange')
    plt.title("Autocorrelación de la señal")
    plt.xlabel("Desplazamiento")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Configurar y leer datos
    sensor = SensorData(puerto='COM6', velocidad=9600, num_muestras=100)
    muestras = sensor.leer_datos()

    # Procesar la señal
    procesador = SignalProcessor()
    autocorr = procesador.calcular_autocorrelacion(muestras)

    # Graficar la señal y la autocorrelación
    graficar_datos(muestras, autocorr)

    # Detectar y mostrar el periodo y la frecuencia
    periodo_muestras = procesador.detectar_periodo(autocorr)
    if periodo_muestras:
        frecuencia_hz = procesador.calcular_frecuencia(periodo_muestras)
        print(f"Período estimado: {periodo_muestras} muestras")
        print(f"Frecuencia estimada: {frecuencia_hz:.2f} Hz")
    else:
        print("No se pudo detectar un período claro.")

if __name__ == "__main__":
    main()
