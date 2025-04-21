import numpy as np
import matplotlib.pyplot as plt

# Parámetros del circuito
R = 1e6  # 1MΩ
C = 0.1e-6  # 0.1µF
tau = R * C  # Constante de tiempo
I = 1.0  # 1A

# Configuración de la simulación
t_start = 0
t_end = 5 * tau  # Simulamos hasta 5 constantes de tiempo
dt = tau / 100  # Paso de tiempo
steps = int((t_end - t_start) / dt)


# Solución analítica exacta
def exact_solution(t):
    return I * R * (1 - np.exp(-t / tau))


# Método de Euler
def euler_method():
    V = np.zeros(steps)
    t = np.zeros(steps)
    V[0] = 0  # Condición inicial

    for i in range(1, steps):
        t[i] = t[i - 1] + dt
        dVdt = (I - V[i - 1] / R) / C
        V[i] = V[i - 1] + dVdt * dt

    return t, V


# Método Trapezoidal
def trapezoidal_method():
    V = np.zeros(steps)
    t = np.zeros(steps)
    V[0] = 0  # Condición inicial

    for i in range(1, steps):
        t[i] = t[i - 1] + dt
        # Predictor (Euler)
        dVdt_prev = (I - V[i - 1] / R) / C
        V_pred = V[i - 1] + dVdt_prev * dt
        # Corrector (Trapezoidal)
        dVdt_new = (I - V_pred / R) / C
        V[i] = V[i - 1] + (dVdt_prev + dVdt_new) * dt / 2

    return t, V


# Ejecutar simulaciones
t_euler, V_euler = euler_method()
t_trap, V_trap = trapezoidal_method()
t_exact = np.linspace(t_start, t_end, steps)
V_exact = exact_solution(t_exact)

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(t_exact, V_exact, 'k-', label='Solución Exacta')
plt.plot(t_euler, V_euler, 'r--', label='Método de Euler')
plt.plot(t_trap, V_trap, 'b-.', label='Método Trapezoidal')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.title('Respuesta del Circuito RC a Escalón de Corriente')
plt.legend()
plt.grid(True)
plt.show()

# Calcular errores
error_euler = np.abs(V_euler - V_exact)
error_trap = np.abs(V_trap - V_exact)

plt.figure(figsize=(10, 6))
plt.semilogy(t_euler, error_euler, 'r--', label='Error Euler')
plt.semilogy(t_trap, error_trap, 'b-.', label='Error Trapezoidal')
plt.xlabel('Tiempo (s)')
plt.ylabel('Error Absoluto')
plt.title('Error de los Métodos Numéricos')
plt.legend()
plt.grid(True)
plt.show()
