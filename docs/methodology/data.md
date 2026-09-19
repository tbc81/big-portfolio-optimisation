# Data preparation

## Adjusted prices

Adjusted prices are downloaded with `yfinance`. Corporate actions such as stock splits and distributions can otherwise introduce price changes that do not represent investment performance.

The configured universe contains assets quoted in GBP, GBX, EUR, USD and CAD. All prices are converted into GBP before returns are calculated.

## Currency conversion

For a GBP investor, the return on a foreign asset depends on both the local asset price and the exchange rate. Converting the price series first ensures that both effects enter the GBP return.

The current implementation applies the quote convention of each downloaded FX series:

- GBX prices are divided by 100 to convert pence into pounds;
- EUR prices are multiplied by EUR/GBP;
- USD prices are divided by GBP/USD;
- CAD prices are divided by GBP/CAD.

For example, a US equity can rise in USD while falling in GBP terms if sterling appreciates sufficiently over the same period. The model therefore estimates risk and return from the investor's GBP price path.

## Historical proxy for CSH2

CSH2 has a shorter observed history than the main estimation sample. Its pre-history is extended using the Bank of England SONIA Compounded Index, as specified in the configuration.

Let $P_{t_0}$ denote the first observed CSH2 market price and let $I_{t_0}$ denote the corresponding SONIA index value. The scaling factor is

$$
s = \frac{P_{t_0}}{I_{t_0}}.
$$

For dates before $t_0$, the proxy price is

$$
\widetilde{P}_t = s I_t.
$$

The scaling makes the proxy continuous with the first observed market price. Observed CSH2 prices are left unchanged. When dates do not match exactly, the implementation uses the latest proxy observation available on or before the asset date, avoiding look-ahead.

This treatment supplies a cash-like historical series for the missing period. It does not reproduce the realised trading history of CSH2, which remains a market instrument with fees and tracking effects.

## Weekly sampling

Daily prices are resampled to `W-FRI` using the last available market observation in each weekly period.

Weekly sampling retains substantially more observations than monthly data while reducing some daily noise and calendar mismatch across markets. The choice also keeps the covariance estimate based on a reasonably large joint sample. The frequency and annualisation factor remain configurable.

## Complete return scenarios

For weekly GBP price $P_t$, the log return is

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right).
$$

Log returns are additive through time. If a price moves from $P_0$ to $P_1$ and then from $P_1$ to $P_2$,

$$
\ln\left(\frac{P_1}{P_0}\right)
+
\ln\left(\frac{P_2}{P_1}\right)
=
\ln\left(\frac{P_2}{P_0}\right).
$$

Covariance requires returns for all assets in the same observation period. Weeks containing a missing return for any asset are therefore removed before moment estimation. This gives a complete joint scenario matrix. The cost is a smaller effective sample when one asset has sparse history.