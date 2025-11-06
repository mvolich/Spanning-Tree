# Spanning Tree Network Analysis - Streamlit App

Network-based portfolio optimization for European sovereign bonds using FRED data.

## Overview

This application demonstrates how network methods can inform portfolio allocation using European sovereign bond yield data. It constructs a maximum spanning tree from correlation networks and uses eigenvector centrality to guide portfolio weighting decisions.

## Features

- **Data Source**: FRED API for 10-year government bond yields (9 European countries)
- **Network Analysis**: Maximum spanning tree construction
- **Centrality Measures**: Eigenvector centrality for portfolio classification
- **Interactive**: Adjustable parameters and date ranges
- **Export**: Download correlation matrices and centrality rankings

## Deployment to Streamlit Cloud

### Prerequisites

1. Get a free FRED API key:
   - Visit https://fred.stlouisfed.org/
   - Create an account
   - Request an API key at https://fred.stlouisfed.org/docs/api/api_key.html

### Steps

1. **Push to GitHub**:
   ```bash
   git add spanning_tree_app.py requirements.txt
   git commit -m "Add spanning tree network analysis app"
   git push
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `spanning_tree_app.py`

3. **Configure Secrets**:
   - In your deployed app, go to Settings → Secrets
   - Add the following:
     ```toml
     fred_key = "your_actual_fred_api_key"
     ```
   - Click "Save"

4. **Run the App**:
   - The app will restart automatically
   - Select your parameters and click "Run Analysis"

## Local Development

To run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Create .streamlit/secrets.toml
mkdir .streamlit
echo 'fred_key = "your_api_key"' > .streamlit/secrets.toml

# Run the app
streamlit run spanning_tree_app.py
```

## Data Sources

**FRED Series Used** (Monthly 10-Year Government Bond Yields):
- Germany: IRLTLT01DEM156N
- Italy: IRLTLT01ITM156N
- Spain: IRLTLT01ESM156N
- Netherlands: IRLTLT01NLM156N
- Belgium: IRLTLT01BEM156N
- Austria: IRLTLT01ATM156N
- Portugal: IRLTLT01PTM156N
- Ireland: IRLTLT01IEM156N
- Finland: IRLTLT01FIM156N

## Methodology

### Maximum Spanning Tree
Selects a subset of correlations where:
- All nodes (bonds) are connected
- Total correlation magnitude is maximized
- No cycles exist in the network

### Eigenvector Centrality
Measures relative influence of each node, accounting for:
- Direct connections to other nodes
- Higher-order connectivity
- Quality of connections

### Portfolio Application
- **High-centrality nodes** (red): Underweight to reduce correlated risk
- **Low-centrality nodes** (blue): Overweight to increase diversification

## Notes

- FRED provides **monthly** data (vs. daily data in LSEG)
- Results may vary from daily data analysis
- The methodology remains identical to the original research

## Files

- `spanning_tree_app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `secrets.toml.template` - Template for secrets configuration
- `README.md` - This file

## References

Based on research demonstrating network-based approaches to reduce overfitting in portfolio optimization, especially for large asset universes.
