import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Defining Pauli matrices
sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
identity = np.eye(2, dtype=complex)

# Defining parameters
Theta_0 = 1.0
chi_param = 0.5
t_vals = np.linspace(0, 10, 100)  # Time range
chi_vals = np.linspace(-np.pi, np.pi, 100) # Chi range

# Function to compute c(t)
def c_t_complex(t):
    d = np.sqrt(chi_param**2 - 2 * Theta_0 * chi_param + 0j)
    return np.exp(-chi_param * t / 2) * (np.cosh(d * t / 2) + (chi_param / (d + 1e-10)) * np.sinh(d * t / 2))

# Computing time derivative of c(t)
def c_dot_t_complex(t):
    d = np.sqrt(chi_param**2 - 2 * Theta_0 * chi_param + 0j)
    return - (chi_param / 2) * c_t_complex(t) + (d / 2) * np.exp(-chi_param * t / 2) * (np.sinh(d * t / 2) + (chi_param / (d + 1e-10)) * np.cosh(d * t / 2))

# Computing Gamma(t) and S(t)
gamma_t_complex = -2 * np.real(c_dot_t_complex(t_vals) / (c_t_complex(t_vals) + 1e-10))
S_t_complex = -2 * np.imag(c_dot_t_complex(t_vals) / (c_t_complex(t_vals) + 1e-10))

# Defining the tilted Liouvillian including particle counting
def tilted_Liouvillian_complex(chi, t_idx):
    commutator = -1j * S_t_complex[t_idx] * (np.kron(identity, sigma_plus @ sigma_minus) - np.kron((sigma_plus @ sigma_minus).T, identity))

    dissipation = gamma_t_complex[t_idx] * (np.exp(1j * chi) * np.kron(sigma_minus, sigma_plus.T) -  
                                            0.5 * np.kron(identity, (sigma_plus @ sigma_minus).T) -  
                                            0.5 * np.kron((sigma_plus @ sigma_minus), identity))

    return commutator + dissipation

# Computing CGF at different times
cgf_time_series_complex = []
for t_idx, t in enumerate(t_vals):
    cgf_vals = []
    for chi in chi_vals:
        K_chi = tilted_Liouvillian_complex(chi, t_idx)
        eigvals = la.eigvals(K_chi)
        cgf_vals.append(eigvals[np.argmax(np.real(eigvals))])  # CGF from dominant eigenvalue
    cgf_time_series_complex.append(cgf_vals)

cgf_time_series_complex = np.array(cgf_time_series_complex)



def numerical_derivative(cgf_vals, chi_vals, order):
    derivatives = cgf_vals
    for _ in range(order):
        derivatives = np.gradient(derivatives, chi_vals, edge_order=2)  # Differentiate
    idx_0 = np.argmin(np.abs(chi_vals))  # Find index closest to χ = 0
    return (1j)**order * derivatives[idx_0]  # Evaluate at χ = 0


mean_current = [numerical_derivative(cgf_time_series_complex[i], chi_vals, 1) for i in range(len(t_vals))]
second_moment = [numerical_derivative(cgf_time_series_complex[i], chi_vals, 2) for i in range(len(t_vals))]
third_moment = [numerical_derivative(cgf_time_series_complex[i], chi_vals, 3) for i in range(len(t_vals))]


# Function to check if imaginary part is negligible
def check_imaginary(name, values, threshold=1e-10):
    max_imag = np.max(np.abs(np.imag(values)))
    if max_imag > threshold:
        print(f"Warning: {name} has non-negligible imaginary parts (max imaginary value: {max_imag})")
    return np.real(values)  # Return only the real part

# Converting to real values only if imaginary part is negligible
mean_current = check_imaginary("Mean Current", mean_current)
second_moment = check_imaginary("Second Moment", second_moment)
third_moment = check_imaginary("Third Moment", third_moment)

# Plotting results
plt.figure(figsize=(6, 4))
plt.plot(t_vals, mean_current, label="Mean Current ⟨n⟩", color="b")
plt.plot(t_vals, second_moment, label="Second moment ⟨n²⟩", color="r")
plt.plot(t_vals, third_moment, label="Third moment ⟨n³⟩", color="g")
plt.xlabel("Time t")
plt.ylabel("Cumulants")
plt.title("Cumulants using FCS")
plt.legend()
plt.grid()
plt.show()