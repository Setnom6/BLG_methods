import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt

# === Filtering and signal shaping helpers === 

def applySlewRateLimit(signal, tList, maxSlopePerNs):
    """
    Limit the maximum slope (slew rate) of the signal.
    Parameters
    ----------
    signal : np.ndarray
        Input signal array.
    tList : np.ndarray
        Time list in ns.
    maxSlopePerNs : float
        Maximum allowed change per ns in same units as signal.
    Returns
    -------
    np.ndarray
        Slew-rate limited signal.
    """
    out = np.empty_like(signal)
    out[0] = signal[0]
    dt = float(np.mean(np.diff(tList)))
    maxDelta = maxSlopePerNs * dt
    for i in range(1, len(signal)):
        delta = np.clip(signal[i] - out[i-1], -maxDelta, maxDelta)
        out[i] = out[i-1] + delta
    return out

def applyMultiStageLowPass(signal, tList, fcListMHz=(8.0, 25.0), perStageOrder=1):
    """
    Apply a cascade of low-pass Butterworth filters to emulate multiple RC stages.
    Parameters
    ----------
    signal : np.ndarray
        Input signal array.
    tList : np.ndarray
        Time list in ns.
    fcListMHz : tuple of floats
        Cutoff frequencies (MHz) for each stage, representing e.g. 300K and 4K filtering.
    perStageOrder : int
        Order of each Butterworth stage.
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    y = np.copy(signal)
    dtNs = float(np.mean(np.diff(tList)))
    fsMHz = 1.0 / dtNs
    nyq = fsMHz / 2.0
    for fc in fcListMHz:
        wn = min(max(fc / nyq, 1e-6), 0.999999)
        sos = butter(perStageOrder, wn, btype='low', output='sos')
        y = sosfiltfilt(sos, y)
    return y

def applyRinging(signal, tList, f0MHz=70.0, zeta=0.3):
    """
    Apply a lightly underdamped second-order system to emulate ringing due to impedance mismatch.
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    tList : np.ndarray
        Time list in ns.
    f0MHz : float
        Resonance frequency in MHz.
    zeta : float
        Damping factor (<1 underdamped).
    Returns
    -------
    np.ndarray
        Signal with ringing effect.
    """
    from scipy.signal import cont2discrete
    dt_s = float(np.mean(np.diff(tList))) * 1e-9
    w0 = 2 * np.pi * f0MHz * 1e6
    num = [w0**2]
    den = [1.0, 2*zeta*w0, w0**2]
    bz, az, _ = cont2discrete((num, den), dt_s, method='bilinear')
    return filtfilt(np.ravel(bz), np.ravel(az), signal)

def applyDelay(signal, samples):
    """
    Apply a pure delay by shifting the signal.
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    samples : int
        Delay in number of samples.
    Returns
    -------
    np.ndarray
        Delayed signal.
    """
    if samples <= 0:
        return signal
    return np.concatenate([np.full(samples, signal[0]), signal[:-samples]])

def applyQuantization(signal, nBits=14, vMin=None, vMax=None):
    """
    Quantize the signal to simulate finite DAC resolution.
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    nBits : int
        Number of DAC bits.
    vMin, vMax : float or None
        Full-scale range limits. Defaults to min and max of signal.
    Returns
    -------
    np.ndarray
        Quantized signal.
    """
    xMin = np.min(signal) if vMin is None else vMin
    xMax = np.max(signal) if vMax is None else vMax
    levels = 2**nBits
    q = np.round((signal - xMin) / (xMax - xMin) * (levels - 1))
    return xMin + q * (xMax - xMin) / (levels - 1)

def addGaussianNoise(signal, sigma):
    """
    Add white Gaussian noise to the signal.
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sigma : float
        Noise standard deviation in same units as signal.
    Returns
    -------
    np.ndarray
        Noisy signal.
    """
    return signal + np.random.normal(0.0, sigma, size=signal.shape)


