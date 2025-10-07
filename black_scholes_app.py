import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


def _calculate_d1_d2(S, K, T, r, sigma):
    """Calculate d1 and d2 for Black-Scholes formula"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)

    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    theta_base = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        theta = (theta_base - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (theta_base + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}

def monte_carlo_simulation(S, K, T, r, sigma, num_simulations, num_steps, option_type='call'):
    """Run Monte Carlo simulation for option pricing"""
    dt = T / num_steps
    np.random.seed(42)

    paths = np.zeros((num_simulations, num_steps + 1))
    paths[:, 0] = S

    for t in range(1, num_steps + 1):
        z = np.random.standard_normal(num_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    final_prices = paths[:, -1]
    payoffs = np.maximum(final_prices - K, 0) if option_type == 'call' else np.maximum(K - final_prices, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)

    return paths, option_price, payoffs

def main():
    st.set_page_config(page_title="Black-Scholes Options Pricer", layout="wide")

    st.title("📈 Black-Scholes Options Pricing Model")
    st.markdown("Interactive tool for pricing European options and analyzing Greeks")

    st.sidebar.header("Option Parameters")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        S = st.number_input("Stock Price (S)", min_value=1.0, value=100.0, step=1.0)
        K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
    with col2:
        T = st.number_input("Time to Expiry (Years)", min_value=0.01, value=1.0, step=0.1)
        r = st.number_input("Risk-free Rate (%)", min_value=0.0, value=5.0, step=0.5) / 100

    sigma = st.sidebar.slider("Volatility (σ) %", min_value=1, max_value=100, value=20, step=1) / 100
    option_type = st.sidebar.radio("Option Type", ["Call", "Put"])

    option_price = black_scholes_call(S, K, T, r, sigma) if option_type == "Call" else black_scholes_put(S, K, T, r, sigma)
    greeks = calculate_greeks(S, K, T, r, sigma, option_type.lower())

    st.header("Option Price")
    col1, col2, col3 = st.columns(3)
    intrinsic = max(S - K, 0) if option_type == "Call" else max(K - S, 0)
    time_value = option_price - intrinsic

    with col1:
        st.metric("Option Price", f"${option_price:.2f}")
    with col2:
        st.metric("Intrinsic Value", f"${intrinsic:.2f}")
    with col3:
        st.metric("Time Value", f"${time_value:.2f}")

    st.header("Greeks")
    greek_cols = st.columns(5)
    for i, (greek_name, greek_value) in enumerate(greeks.items()):
        with greek_cols[i]:
            st.metric(greek_name, f"{greek_value:.4f}")

    st.header("Sensitivity Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Price vs Stock Price", "Price vs Volatility", "Price vs Time", "🎲 Monte Carlo Simulation"])

    with tab1:
        stock_range = np.linspace(S * 0.5, S * 1.5, 100)
        call_prices = [black_scholes_call(s, K, T, r, sigma) for s in stock_range]
        put_prices = [black_scholes_put(s, K, T, r, sigma) for s in stock_range]

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=stock_range, y=call_prices, name='Call', line=dict(color='green', width=2)))
        fig1.add_trace(go.Scatter(x=stock_range, y=put_prices, name='Put', line=dict(color='red', width=2)))
        fig1.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text="Current Price")
        fig1.add_vline(x=K, line_dash="dash", line_color="blue", annotation_text="Strike Price")
        fig1.update_layout(title="Option Price vs Stock Price", xaxis_title="Stock Price", yaxis_title="Option Price", height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        vol_range = np.linspace(0.01, 1.0, 100)
        call_prices_vol = [black_scholes_call(S, K, T, r, v) for v in vol_range]
        put_prices_vol = [black_scholes_put(S, K, T, r, v) for v in vol_range]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=vol_range * 100, y=call_prices_vol, name='Call', line=dict(color='green', width=2)))
        fig2.add_trace(go.Scatter(x=vol_range * 100, y=put_prices_vol, name='Put', line=dict(color='red', width=2)))
        fig2.add_vline(x=sigma * 100, line_dash="dash", line_color="gray", annotation_text="Current Vol")
        fig2.update_layout(title="Option Price vs Volatility", xaxis_title="Volatility (%)", yaxis_title="Option Price", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        time_range = np.linspace(0.01, T * 2, 100)
        call_prices_time = [black_scholes_call(S, K, t, r, sigma) for t in time_range]
        put_prices_time = [black_scholes_put(S, K, t, r, sigma) for t in time_range]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=time_range, y=call_prices_time, name='Call', line=dict(color='green', width=2)))
        fig3.add_trace(go.Scatter(x=time_range, y=put_prices_time, name='Put', line=dict(color='red', width=2)))
        fig3.add_vline(x=T, line_dash="dash", line_color="gray", annotation_text="Current Time")
        fig3.update_layout(title="Option Price vs Time to Expiry", xaxis_title="Time to Expiry (Years)", yaxis_title="Option Price", height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("### 🎲 Monte Carlo Simulation")
        st.markdown("Simulate thousands of possible stock price paths to estimate option value through probability")

        col1, col2 = st.columns(2)
        with col1:
            num_simulations = st.slider("Number of Simulations", 100, 10000, 1000, step=100)
        with col2:
            num_steps = st.slider("Steps per Path", 50, 500, 100, step=50)

        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Running Monte Carlo simulation..."):
                paths, mc_price, payoffs = monte_carlo_simulation(S, K, T, r, sigma, num_simulations, num_steps, option_type.lower())

                bs_price = option_price
                difference = mc_price - bs_price
                error_pct = (difference / bs_price) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Monte Carlo Price", f"${mc_price:.2f}")
                with col2:
                    st.metric("Black-Scholes Price", f"${bs_price:.2f}")
                with col3:
                    st.metric("Difference", f"${difference:.2f}", f"{error_pct:.2f}%")

                st.markdown("#### Simulated Stock Price Paths")
                fig_paths = go.Figure()

                num_paths_to_plot = min(100, num_simulations)
                time_steps = np.linspace(0, T, num_steps + 1)

                for i in range(num_paths_to_plot):
                    fig_paths.add_trace(go.Scatter(
                        x=time_steps,
                        y=paths[i],
                        mode='lines',
                        line=dict(width=0.5),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                fig_paths.add_hline(y=K, line_dash="dash", line_color="red",
                                    annotation_text=f"Strike: ${K}", annotation_position="right")
                fig_paths.add_hline(y=S, line_dash="dash", line_color="blue",
                                    annotation_text=f"Start: ${S}", annotation_position="right")

                fig_paths.update_layout(
                    title=f"Stock Price Simulation ({num_paths_to_plot} of {num_simulations} paths shown)",
                    xaxis_title="Time (Years)",
                    yaxis_title="Stock Price",
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig_paths, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Distribution of Final Stock Prices")
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=paths[:, -1],
                        nbinsx=50,
                        name='Final Prices',
                        marker_color='lightblue'
                    ))
                    fig_dist.add_vline(x=K, line_dash="dash", line_color="red",
                                       annotation_text="Strike")
                    fig_dist.add_vline(x=np.mean(paths[:, -1]), line_dash="dash",
                                       line_color="green", annotation_text="Mean")
                    fig_dist.update_layout(
                        xaxis_title="Final Stock Price",
                        yaxis_title="Frequency",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col2:
                    st.markdown("#### Distribution of Option Payoffs")
                    fig_payoff = go.Figure()
                    fig_payoff.add_trace(go.Histogram(
                        x=payoffs,
                        nbinsx=50,
                        name='Payoffs',
                        marker_color='lightgreen'
                    ))
                    fig_payoff.add_vline(x=np.mean(payoffs), line_dash="dash",
                                         line_color="red", annotation_text=f"Mean: ${np.mean(payoffs):.2f}")
                    fig_payoff.update_layout(
                        xaxis_title="Option Payoff at Expiry",
                        yaxis_title="Frequency",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_payoff, use_container_width=True)

                st.markdown("#### Simulation Statistics")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Profitable Outcomes", f"{(payoffs > 0).sum():,}")
                with stats_col2:
                    st.metric("Probability of Profit", f"{(payoffs > 0).sum() / num_simulations * 100:.1f}%")
                with stats_col3:
                    st.metric("Avg Final Price", f"${np.mean(paths[:, -1]):.2f}")
                with stats_col4:
                    st.metric("Avg Payoff", f"${np.mean(payoffs):.2f}")

        else:
            st.info("👆 Click 'Run Simulation' to see thousands of possible stock price paths and how they affect option pricing!")
            st.markdown("""
            **What is Monte Carlo Simulation?**

            Instead of using a formula (like Black-Scholes), Monte Carlo simulation:
            1. Simulates thousands of possible future stock price paths
            2. Calculates the option payoff for each path
            3. Averages the payoffs and discounts to present value

            This provides an intuitive, visual understanding of option pricing based on probability!
            """)

    with st.expander("ℹ About the Black-Scholes Model & Monte Carlo Simulation"):
        st.markdown("""
        ### Black-Scholes Model
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

        ---

        ### Monte Carlo Simulation
        **Monte Carlo simulation** is a numerical method for option pricing that uses randomness to model uncertainty.

        **How it works:**
        1. **Simulate Stock Paths**: Generate thousands of possible future stock price paths using the geometric Brownian motion model:
           - Each path represents one possible "future" for the stock
           - Uses the same volatility (σ) and drift (r) as Black-Scholes
           - Incorporates random movements to capture uncertainty

        2. **Calculate Payoffs**: For each simulated path, calculate what the option would be worth at expiration:
           - **Call option**: max(Final Stock Price - Strike Price, 0)
           - **Put option**: max(Strike Price - Final Stock Price, 0)

        3. **Average & Discount**: Take the average of all payoffs and discount back to present value using the risk-free rate

        **Why use Monte Carlo?**
        - **Visual & Intuitive**: You can actually SEE the uncertainty in stock prices
        - **Flexible**: Works for complex options where formulas don't exist (exotic options, American options with dividends)
        - **Validates Theory**: Monte Carlo prices should converge to Black-Scholes prices (for European options)

        **Key Insight**: The Monte Carlo price should match the Black-Scholes price closely (within a few percent).
        The difference comes from:
        - **Simulation error**: More simulations = more accurate (try 10,000+ for precision)
        - **Path steps**: More steps = smoother paths and better accuracy
        - **Random seed**: Different random numbers each time (can be fixed for reproducibility)

        Monte Carlo is the foundation of modern quantitative finance and is used by banks and hedge funds to price
        complex derivatives that don't have closed-form solutions!
        """)


if __name__ == "__main__":
    main()