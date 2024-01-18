import numpy as np

def compute_trace(E, t, Δt):
    """
    Computes the SHG-FROG trace of a pulse E(t).
    Given by:
        T(ω, τ) =  | ∫ E(t)E(t - τ) exp(-i ω t) dt |²

    Where ω is the frequency and τ is the time delay.

    Args:
        E (np.array(complex)): Electric field of the pulse
        t (np.array): time array of the pulse

    Returns:
        T (np.array): SHG-FROG trace. NxN matrix of real numbers
    """
    N = E.size

    ω = -1 / (2 * Δt) + np.arange(N) /(N * Δt) 
    ω *= 2 * np.pi
    Δω = 2 * np.pi / Δt # Reciprocity relation

    # Time delays
    τ = np.array([-np.floor(0.5 * N) * Δt + i * Δt for i in range(N)])

    # Introduce the delay in the frequency domain, as a phase factor exp(i ω τ)
    delays = np.zeros((N, N), dtype=np.complex128)
    for i, tau in enumerate(τ):
        delays[i][:] = np.exp(1j * ω * tau)

    spectrum = DFT(E, Δt)

    # Compute the trace
    T = np.zeros((N,N))
    for i in range(N):
        T[i][:] = np.abs(DFT(E * IDFT(spectrum * delays[i], Δt), Δt))**2

    return T

def DFT(E, Δt):
    """
    Computes the Discrete Fourier Transform of a given pulse E(t), using
    the Fast Fourier Transform algorithm.

    Args:
        E (np.array(complex)): Electric field of the pulse in the time domain
        Δt (float): time step

    Returns:
        Ẽ (np.array(complex)): Electric field of the pulse in the frequency domain
    """
    N = E.size
    
    ω = -1 / (2 * Δt) + np.arange(N) /(N * Δt)
    ω *= 2 * np.pi
    Δω = 2 * np.pi / Δt 

    # Time array centered at zero with N points and Δt step
    t = np.array([-np.floor(0.5 * N) * Δt + i * Δt for i in range(N)])

    # Phase factors
    r_n = np.exp(-1j * np.arange(np.size(ω)) * t[0] * Δω)
    s_j = np.exp(-1j * ω[0] * t)

    return Δt / (2 * np.pi) * r_n * np.fft.ifft(E * s_j)

def IDFT(Ẽ, Δt):
    """
    Computes the Inverse Discrete Fourier Transform of a given pulse Ẽ(ω), using
    the Fast Fourier Transform algorithm.

    Args:
        Ẽ (np.array(complex)): Electric field of the pulse in the frequency domain
        Δt (float): time step

    Returns:
        E (np.array(complex)): Electric field of the pulse in the time domain
    """
    N = Ẽ.size

    ω = -1 / (2 * Δt) + np.arange(N) /(N * Δt)
    ω *= 2 * np.pi
    Δω = 2 * np.pi / Δt 

    # Time array centered at zero with N points and Δt step
    t = np.array([-np.floor(0.5 * N) * Δt + i * Δt for i in range(N)])

    # Phase factors
    r_n_conj = np.exp(1j * np.arange(np.size(ω)) * t[0] * Δω)
    s_j_conj = np.exp(1j * ω[0] * t)

    return s_j_conj * Δω * np.fft.fft(Ẽ * r_n_conj)