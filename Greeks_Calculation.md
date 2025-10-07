# Option Greeks Calculation

## Overview
This function calculates the five main "Greeks" used in options trading to measure different aspects of risk and sensitivity.

## Function Definition
```python
def calc_greeks(S, K, T, r, sigma):
```

## Parameters
- **S**: Current stock price (spot price)
- **K**: Strike price of the option
- **T**: Time to expiration (in years)
- **r**: Risk-free interest rate (annual)
- **sigma**: Volatility of the underlying asset (annual)

## Intermediate Calculations

### d1 and d2
```python
d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
```
These are standard Black-Scholes intermediate values used in option pricing formulas.

## The Greeks

### 1. Delta (Δ)
**Measures**: Rate of change of option price with respect to stock price

**Call Option**: `delta = norm.cdf(d1)`
**Put Option**: `delta = norm.cdf(d1) - 1`

- Call delta ranges from 0 to 1
- Put delta ranges from -1 to 0

### 2. Gamma (Γ)
**Measures**: Rate of change of delta with respect to stock price

```python
gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
```

- Same for both calls and puts
- Highest for at-the-money options

### 3. Theta (Θ)
**Measures**: Rate of change of option price with respect to time (time decay)

**Call Option**:
```python
theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
         - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
```

**Put Option**:
```python
theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
         + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
```

- Expressed as daily decay (divided by 365)
- Usually negative for long options

### 4. Vega (ν)
**Measures**: Rate of change of option price with respect to volatility

```python
vega = S * norm.pdf(d1) * np.sqrt(T) / 100
```

- Same for both calls and puts
- Expressed as change per 1% volatility change (divided by 100)
- Higher for longer-dated options

### 5. Rho (ρ)
**Measures**: Rate of change of option price with respect to interest rate

**Call Option**:
```python
rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
```

**Put Option**:
```python
rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
```

- Expressed as change per 1% interest rate change (divided by 100)
- Positive for calls, negative for puts

## Return Value
```python
return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}
```

Returns a dictionary containing all five Greeks.

## Note
⚠️ **Indentation Issue**: The return statement appears to be indented under the `else` block for rho calculation. It should be aligned with the function body to return correctly for all cases.

## Dependencies
- `numpy` (as np): For mathematical operations
- `scipy.stats.norm`: For normal distribution functions (cdf and pdf)
- `option_type` variable: Must be defined in the calling scope (either 'call' or 'put')
