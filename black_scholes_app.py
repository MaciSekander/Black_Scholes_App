# importing packages
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def black_scholes_calls(S, K, T, r, sigma):
    # calculating the black scholes call option price
    d1 = (np.log(S/K) + (r + 0.5 * sigma **2 )*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

def black_scholes_puts(S, K, r, T, sigma):
    # calculating the black scholes put option price
    d1 = (np.log(S/K) + (r + 0.5 * sigma **2 )*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Calculating the greeks
def calc_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5 * sigma **2 )*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    if option_type == 'call':
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}

# Streamlit App
st.set_page_config(page_title='Black-Scholes Options Pricer', layout='wide')

st.title('Black-Scholes Options Pricing Model')
st.markdown('Interactive tool for pricing European call and put options using the Black-Scholes formula. Adjust parameters to see how option prices and Greeks change.')

# Sidebar for user inputs
st.sidebar.header('Option Parameters')

S = st.sidebar.number_input('Current Stock Price (S)', min_value=0.01, value=100.0, step=0.01, format="%.2f")
K = st.sidebar.number_input('Strike Price (K)', min_value=0.01, value=100.0, step=0.01, format="%.2f")
T = st.sidebar.number_input('Time to Expiration (T in years)', min_value=0.01, value=1.0, step=0.01, format="%.2f")
r = st.sidebar.number_input('Risk-Free Interest Rate (r)', min_value=0.0, value=0.05, step=0.001, format="%.3f")
sigma = st.sidebar.slider('Volatility (σ)', min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f")
option_type = st.sidebar.selectbox('Option Type', options=['call', 'put'])

# Calculate option price and greeks
if option_type == 'call':
    option_price = black_scholes_calls(S, K, T, r, sigma)
else:
    option_price = black_scholes_puts(S, K, r, T, sigma)

greeks = calc_greeks(S, K, T, r, sigma, option_type.lower())

# Display results
st.header('Option Price and Greeks')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Option Price', f"${option_price:.2f}")
with col2:
    intrinsic_value = max(0, S - K) if option_type == 'call' else max(K - S, 0)
    st.metric('Intrinsic Value', f"${intrinsic_value:.2f}")
with col3:
    time_value = option_price - intrinsic_value
    st.metric('Time Value', f"${time_value:.2f}")

# Display Greeks
st.header('Greeks')
greek_cols = st.columns(5)
greek_names = list(greeks.keys())
for i, (greek_name, greek_value) in enumerate(greeks.items()):
    with greek_cols[i]:
        st.metric(greek_name, f"{greek_value:.4f}")

# Visualizations
st.header('Sensitivity Analysis')

tab1, tab2, tab3 = st.tabs(['Price vs Stock Price', 'Price vs Volatility', 'Price vs Time to Expiration'])

with tab1:
    stock_range = np.linspace(S * 0.5, S * 1.5, 100)
    call_prices = [black_scholes_calls(s, K, T, r, sigma) for s in stock_range]
    put_prices = [black_scholes_puts(s, K, T, r, sigma) for s in stock_range]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=stock_range, y=call_prices, name='Call', line=dict(color='green', width=2)))
    fig1.add_trace(go.Scatter(x=stock_range, y=put_prices, name='Put', line=dict(color='red', width=2)))
    fig1.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text="Current Price")
    fig1.add_vline(x=K, line_dash="dash", line_color="blue", annotation_text="Strike Price")
    fig1.update_layout(title="Option Price vs Stock Price", xaxis_title="Stock Price", yaxis_title="Option Price", height=400)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    vol_range = np.linspace(0.01, 1.0, 100)
    call_prices_vol = [black_scholes_calls(S, K, T, r, v) for v in vol_range]
    put_prices_vol = [black_scholes_puts(S, K, T, r, v) for v in vol_range]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=vol_range, y=call_prices_vol, name='Call', line=dict(color='green', width=2)))
    fig2.add_trace(go.Scatter(x=vol_range, y=put_prices_vol, name='Put', line=dict(color='red', width=2)))
    fig2.add_vline(x=sigma, line_dash="dash", line_color="gray", annotation_text="Current Volatility")
    fig2.update_layout(title="Option Price vs Volatility", xaxis_title="Volatility (σ)", yaxis_title="Option Price", height=400)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    time_range = np.linspace(0.01, T * 2, 100)
    call_prices_time = [black_scholes_calls(S, K, t, r, sigma) for t in time_range]
    put_prices_time = [black_scholes_puts(S, K, t, r, sigma) for t in time_range]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=time_range, y=call_prices_time, name='Call', line=dict(color='green', width=2)))
    fig3.add_trace(go.Scatter(x=time_range, y=put_prices_time, name='Put', line=dict(color='red', width=2)))
    fig3.add_vline(x=T, line_dash="dash", line_color="gray", annotation_text="Current Time")
    fig3.update_layout(title="Option Price vs Time to Expiry", xaxis_title="Time to Expiry (Years)", yaxis_title="Option Price", height=400)
    st.plotly_chart(fig3, use_container_width=True)

# Information section
with st.expander("ℹ️ About the Black-Scholes Model"):
    st.markdown("""
    The **Black-Scholes model** is a mathematical model for pricing European-style options. It assumes:
    - The stock follows a geometric Brownian motion with constant volatility
    - No dividends are paid
    - Markets are efficient (no arbitrage opportunities)
    - Risk-free interest rate is constant
    - Options can only be exercised at expiration (European style)

    **Greeks** measure the sensitivity of the option price to various factors:
    - **Delta (Δ)**: Rate of change of option price with respect to stock price
    - **Gamma (Γ)**: Rate of change of delta with respect to stock price
    - **Theta (Θ)**: Rate of change of option price with respect to time (time decay)
    - **Vega (ν)**: Rate of change of option price with respect to volatility
    - **Rho (ρ)**: Rate of change of option price with respect to interest rate
    """)
