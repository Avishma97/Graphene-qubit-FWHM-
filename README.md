# Graphene-qubit-FWHM-
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy import signal

# Load data
DataPath = r"D:\Others\Curve fit\COOL DOWN-13\Graphene transmon\YY011_YY012_YY013\FluxTuning\YY012DAC_-5.0to5.0mA_SP_20MHz"
f_data = np.load(DataPath + '/VNA_Freq.npy') / 1e9  # GHz
Y = np.load(DataPath + '/Flux_mA.npy')              # mA
Z = np.load(DataPath + '/S21_LOG.npy')              # dB

# Plotting parameters
axis_props = {
    'ticks': 'outside', 'nticks': 11, 'ticklen': 10,
    'showspikes': True, 'tickfont_size': 20, 'title_font_size': 25,
    'exponentformat': 'SI',
}
layout_props = {'width': 1000, 'height': 800, 'title_font_size': 30}

# --- 1. 2D Colormap of Flux Tuning ---
fig = go.Figure(data=go.Heatmap(
    z=Z, x=f_data, y=Y,
    colorscale='Viridis',
    colorbar=dict(title='S21 (dB)')
))
fig.update_layout(
    title="Flux Tuning (S21 vs Freq vs Flux Current)",
    xaxis_title="Frequency (GHz)",
    yaxis_title="Flux Current (mA)",
    **layout_props
)
fig.update_xaxes(**axis_props)
fig.update_yaxes(**axis_props)
fig.show()

# --- 2. Lorentzian Fitting ---
# Define Lorentzian
def lorentzian(f, A, f0, gamma, B):
    return A / (1 + ((f - f0) / (gamma / 2))**2) + B

# Select currents (mA) for linecuts and fitting
desired_current = np.array([-0.64, -0.49, -0.34, -0.19, -0.04])
desired_flux = desired_current * 0.09011081568 * 8.364908922846956
indices = [np.abs(Y - cur).argmin() for cur in desired_flux]

# Optional smoothing
b, a = signal.butter(1, 0.05)

linewidths = []
res_freqs = []

for idx in indices:
    Z_linear = 10**(Z[idx] / 20)
    # Z_linear = signal.filtfilt(b, a, Z_linear)  # Optional

    # Fit Lorentzian
    A_guess = np.max(Z_linear) - np.min(Z_linear)
    f0_guess = f_data[np.argmax(Z_linear)]
    gamma_guess = 0.01
    B_guess = np.min(Z_linear)

    try:
        popt, _ = curve_fit(lorentzian, f_data, Z_linear, p0=[A_guess, f0_guess, gamma_guess, B_guess])
        A_fit, f0_fit, gamma_fit, B_fit = popt
        fit_curve = lorentzian(f_data, *popt)

        linewidths.append(gamma_fit * 1000)  # MHz
        res_freqs.append(f0_fit)

        # Plot fit vs data
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=f_data, y=Z_linear, name="Data"))
        fig_fit.add_trace(go.Scatter(x=f_data, y=fit_curve, name="Lorentzian Fit", line=dict(dash='dash')))
        fig_fit.update_layout(title=f"Flux: {Y[idx]:.3f} mA | Center Freq: {f0_fit:.4f} GHz | Q: {f0_fit/gamma_fit:.0f}",
                              xaxis_title="Frequency (GHz)", yaxis_title="S21 (Linear Mag)",
                              **layout_props)
        fig_fit.update_xaxes(**axis_props)
        fig_fit.update_yaxes(**axis_props)
        fig_fit.show()

    except RuntimeError:
        print(f"Fit failed at Flux Current: {Y[idx]} mA")

print(f"Linewidths (MHz): {linewidths}")
print(f"Resonance Frequencies (GHz): {res_freqs}")
