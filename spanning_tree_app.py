"""
Spanning Tree Network Analysis for European Sovereign Bonds
Using FRED API data to demonstrate network-based portfolio optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from fredapi import Fred
import os

# Page configuration
st.set_page_config(
    page_title="Sovereign Bond Network Analysis",
    page_icon="🌐",
    layout="wide"
)

# Title and description
st.title("🌐 Network-Based Portfolio Optimization")
st.markdown("""
### Reframing Financial Markets as Complex Systems

**Objective:** This application demonstrates how network methods can inform portfolio allocation 
using European sovereign bond yield data. We construct a network based on correlations between 
sovereign bonds, where network properties—such as centrality—guide portfolio weighting decisions.

**Method:** A maximum spanning tree approach constructs the network by selecting correlations 
such that all nodes are connected, total correlation magnitude is maximized, and no cycles exist. 
Eigenvector centrality measures are then used to classify bonds into portfolio weight groups.

**Use Case:** This approach helps portfolio managers reduce overfitting during backtesting, 
especially with large asset universes, by reducing sensitivity to small-magnitude correlations.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# FRED API Key from Streamlit secrets
try:
    fred_api_key = st.secrets["fred_key"]
except Exception as e:
    st.error("⚠️ FRED API key not found in secrets. Please add 'fred_key' to your Streamlit secrets.")
    fred_api_key = None

# Date range selection
st.sidebar.subheader("Data Parameters")
start_year = st.sidebar.selectbox("Start Year", list(range(2020, 2026)), index=4)
end_year = st.sidebar.selectbox("End Year", list(range(2020, 2026)), index=5)

# Network parameters
st.sidebar.subheader("Network Parameters")
centrality_threshold = st.sidebar.slider(
    "Centrality Threshold",
    min_value=0.3,
    max_value=0.7,
    value=0.5,
    step=0.05,
    help="Cumulative centrality threshold for node classification"
)

# Country definitions
COUNTRIES = {
    'DEU': 'Germany',
    'ITA': 'Italy',
    'ESP': 'Spain',
    'NLD': 'Netherlands',
    'BEL': 'Belgium',
    'AUT': 'Austria',
    'PRT': 'Portugal',
    'IRL': 'Ireland',
    'FIN': 'Finland'
}

# FRED series IDs for 10-year government bond yields (Monthly)
FRED_SERIES = {
    'DEU': 'IRLTLT01DEM156N',
    'ITA': 'IRLTLT01ITM156N',
    'ESP': 'IRLTLT01ESM156N',
    'NLD': 'IRLTLT01NLM156N',
    'BEL': 'IRLTLT01BEM156N',
    'AUT': 'IRLTLT01ATM156N',
    'PRT': 'IRLTLT01PTM156N',
    'IRL': 'IRLTLT01IEM156N',
    'FIN': 'IRLTLT01FIM156N'
}

@st.cache_data
def fetch_bond_data(api_key, start_year, end_year):
    """Fetch bond yield data from FRED API"""
    try:
        fred = Fred(api_key=api_key)
        
        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-31'
        
        bond_data = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (country_code, series_id) in enumerate(FRED_SERIES.items()):
            status_text.text(f"Fetching {COUNTRIES[country_code]} data...")
            try:
                data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                bond_data[country_code] = data
            except Exception as e:
                st.warning(f"Could not fetch data for {COUNTRIES[country_code]}: {str(e)}")
            progress_bar.progress((i + 1) / len(FRED_SERIES))
        
        status_text.empty()
        progress_bar.empty()
        
        # Convert to DataFrame
        df = pd.DataFrame(bond_data)
        
        if df.empty:
            raise ValueError("No data retrieved from FRED")
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def impute_missing_values(df):
    """Impute missing values using iterative linear regression"""
    imputer = IterativeImputer(
        estimator=LinearRegression(),
        max_iter=100,
        random_state=0
    )
    
    imputed_data = imputer.fit_transform(df)
    
    imputed_df = pd.DataFrame(
        imputed_data,
        index=df.index,
        columns=df.columns
    )
    
    return imputed_df

def create_correlation_heatmap(df):
    """Create correlation heatmap using Plotly with adjusted scale for high correlations"""
    corr_matrix = df.corr()
    
    # Create custom colorscale for high correlations (0.6 to 1.0)
    # This provides better visual differentiation for highly correlated data
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[
            [0.0, '#d73027'],    # Red for low correlations (0.6)
            [0.25, '#fee090'],   # Yellow-orange
            [0.5, '#ffffbf'],    # Yellow
            [0.75, '#91cf60'],   # Light green
            [1.0, '#1a9850']     # Dark green for perfect correlation (1.0)
        ],
        zmin=0.6,  # Start color scale at 0.6 instead of -1 or 0
        zmax=1.0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 11},
        colorbar=dict(
            title="Correlation",
            titleside="right",
            tickmode="linear",
            tick0=0.6,
            dtick=0.1
        )
    ))
    
    fig.update_layout(
        title='Correlation Matrix: 10-Year Sovereign Bond Yields',
        title_x=0.5,
        width=800,
        height=700,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig, corr_matrix

def create_spanning_tree_network(corr_matrix, threshold=0.5):
    """Create maximum spanning tree network with centrality-based node coloring using Plotly"""
    
    # Create graph from correlation matrix
    corr_graph = nx.Graph(corr_matrix)
    
    # Create maximum spanning tree
    span_graph = nx.maximum_spanning_tree(corr_graph)
    
    # Calculate eigenvector centrality
    centrality = nx.eigenvector_centrality(span_graph)
    cent_total = sum(centrality.values())
    centrality_normalized = {k: v/cent_total for k, v in centrality.items()}
    
    # Sort nodes by centrality
    sorted_nodes = sorted(centrality_normalized.items(), key=lambda x: x[1], reverse=True)
    
    # Identify high-centrality nodes (underweight)
    cumulative = 0.0
    highlight_nodes = set()
    for node, value in sorted_nodes:
        cumulative += value
        highlight_nodes.add(node)
        if cumulative >= threshold:
            break
    
    # Get layout positions
    pos = nx.spring_layout(span_graph, k=0.3, iterations=200, seed=2)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    edge_midpoints_x = []
    edge_midpoints_y = []
    
    for edge in span_graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Calculate midpoint for edge labels
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_midpoints_x.append(mid_x)
        edge_midpoints_y.append(mid_y)
        edge_text.append(f"{edge[2]['weight']:.2f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Edge labels trace
    edge_label_trace = go.Scatter(
        x=edge_midpoints_x,
        y=edge_midpoints_y,
        mode='text',
        text=edge_text,
        textposition='middle center',
        textfont=dict(size=10, color='#333'),
        hoverinfo='none'
    )
    
    # Create node traces (separate for each color)
    node_x_high = []
    node_y_high = []
    node_text_high = []
    
    node_x_low = []
    node_y_low = []
    node_text_low = []
    
    for node in span_graph.nodes():
        x, y = pos[node]
        if node in highlight_nodes:
            node_x_high.append(x)
            node_y_high.append(y)
            node_text_high.append(node)
        else:
            node_x_low.append(x)
            node_y_low.append(y)
            node_text_low.append(node)
    
    # High centrality nodes (underweight)
    node_trace_high = go.Scatter(
        x=node_x_high, y=node_y_high,
        mode='markers+text',
        text=node_text_high,
        textposition='middle center',
        textfont=dict(size=12, color='black', family='Arial Black'),
        hoverinfo='text',
        hovertext=node_text_high,
        marker=dict(
            size=40,
            color='lightcoral',
            line=dict(width=2, color='darkred')
        ),
        name='Underweight (High Centrality)'
    )
    
    # Low centrality nodes (overweight)
    node_trace_low = go.Scatter(
        x=node_x_low, y=node_y_low,
        mode='markers+text',
        text=node_text_low,
        textposition='middle center',
        textfont=dict(size=12, color='black', family='Arial Black'),
        hoverinfo='text',
        hovertext=node_text_low,
        marker=dict(
            size=40,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        name='Overweight (Low Centrality)'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace_high, node_trace_low])
    
    fig.update_layout(
        title='Maximum Spanning Tree: 10-Year Sovereign Bond Correlations',
        title_x=0.5,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1000,
        height=700,
        plot_bgcolor='white'
    )
    
    return fig, span_graph, centrality_normalized, highlight_nodes

def display_centrality_table(centrality, highlight_nodes):
    """Display centrality measures in a table"""
    
    centrality_df = pd.DataFrame([
        {
            'Country': COUNTRIES[node],
            'Code': node,
            'Centrality': f"{value:.4f}",
            'Weight Group': 'Underweight' if node in highlight_nodes else 'Overweight'
        }
        for node, value in sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    ])
    
    return centrality_df

# Main application logic
if fred_api_key:
    
    if st.sidebar.button("Run Analysis", type="primary"):
        
        with st.spinner("Fetching bond yield data from FRED..."):
            bond_df = fetch_bond_data(fred_api_key, start_year, end_year)
        
        if bond_df is not None and not bond_df.empty:
            
            # Display data info
            st.success(f"✅ Successfully retrieved data for {len(bond_df.columns)} countries")
            
            with st.expander("📊 View Raw Data"):
                st.dataframe(bond_df.tail(12), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Observations", len(bond_df))
                with col2:
                    st.metric("Countries", len(bond_df.columns))
                with col3:
                    st.metric("Missing Values", bond_df.isnull().sum().sum())
            
            # Impute missing values
            with st.spinner("Imputing missing values..."):
                bond_df_imputed = impute_missing_values(bond_df)
            
            st.success("✅ Data imputation complete")
            
            # Section 1: Correlation Analysis
            st.header("1️⃣ Correlation Analysis")
            
            with st.spinner("Generating correlation heatmap..."):
                fig_corr, corr_matrix = create_correlation_heatmap(bond_df_imputed)
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            with st.expander("📈 Correlation Statistics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Correlation", f"{corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
                with col2:
                    st.metric("Max Correlation", f"{corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")
            
            # Section 2: Network Analysis
            st.header("2️⃣ Network-Based Portfolio Allocation")
            
            with st.spinner("Building spanning tree network..."):
                fig_network, span_graph, centrality, highlight_nodes = create_spanning_tree_network(
                    corr_matrix,
                    threshold=centrality_threshold
                )
            
            st.plotly_chart(fig_network, use_container_width=True)
            
            # Section 3: Centrality Analysis
            st.header("3️⃣ Centrality Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Node Centrality Rankings")
                centrality_df = display_centrality_table(centrality, highlight_nodes)
                st.dataframe(centrality_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Portfolio Groups")
                st.markdown(f"""
                **High Centrality (Underweight):**
                - {len(highlight_nodes)} bonds
                - Red nodes in network
                - Suggested: 60-80% of benchmark weight
                
                **Low Centrality (Overweight):**
                - {len(span_graph.nodes) - len(highlight_nodes)} bonds
                - Blue nodes in network
                - Suggested: 120-140% of benchmark weight
                """)
            
            # Section 4: Methodology
            with st.expander("ℹ️ Methodology Details"):
                st.markdown("""
                ### Maximum Spanning Tree
                A maximum spanning tree selects a subset of correlations where:
                - All nodes (bonds) are connected
                - Total correlation magnitude is maximized
                - No cycles exist in the network
                
                ### Eigenvector Centrality
                Measures the relative influence of each node, accounting for:
                - Direct connections to other nodes
                - Higher-order connectivity (connections of connections)
                - Quality of connections (high-centrality neighbors matter more)
                
                ### Portfolio Constraints
                - **High-centrality nodes (red):** Underweight to reduce correlated risk
                - **Low-centrality nodes (blue):** Overweight to increase diversification
                
                ### Rationale
                This approach reduces overfitting by decreasing sensitivity to small-magnitude 
                correlations, improving model robustness especially with large asset universes.
                """)
            
            # Download section
            st.header("📥 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = centrality_df.to_csv(index=False)
                st.download_button(
                    label="Download Centrality Table (CSV)",
                    data=csv_data,
                    file_name="centrality_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_corr = corr_matrix.to_csv()
                st.download_button(
                    label="Download Correlation Matrix (CSV)",
                    data=csv_corr,
                    file_name="correlation_matrix.csv",
                    mime="text/csv"
                )

else:
    st.info("⚠️ FRED API key not configured. Please add to Streamlit secrets.")
    st.markdown("""
    ### Configuration Required
    
    This application requires a FRED API key stored in Streamlit secrets.
    
    **For Streamlit Cloud:**
    1. Get a free FRED API key:
       - Visit [https://fred.stlouisfed.org/](https://fred.stlouisfed.org/)
       - Create an account
       - Request an API key at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
    
    2. Add to your app's secrets (Settings → Secrets):
       ```toml
       fred_key = "your_api_key_here"
       ```
    
    3. Restart the app and click "Run Analysis"
    
    ### Data Source
    This application uses monthly 10-year government bond yield data from FRED for:
    - Germany, Italy, Spain, Netherlands, Belgium
    - Austria, Portugal, Ireland, Finland
    
    ### Note
    FRED provides monthly data (vs. daily in the original LSEG version). 
    Results may differ from daily data analysis but the methodology remains the same.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Data Source: Federal Reserve Economic Data (FRED) | St. Louis Fed</p>
    <p>Methodology based on complex systems theory applied to financial markets</p>
</div>
""", unsafe_allow_html=True)
