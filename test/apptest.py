import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mesa_economic_model import EconomicModel

# Historical Scenario Definition
SCENARIO_2008 = {
    "name": "2008_crisis",
    "crisis_start_month": 24,
    "pre_crisis_unemployment": 0.045,
    "pre_crisis_interest_rate": 0.03,
    "crisis_financial_shock_severity": 2.5,
    "crisis_demand_shock_severity": 0.7,
    "duration": 36
}

VALIDATION_METRICS_2008 = {
    'unemployment_peak': (0.09, 0.11),
    'gdp_drop_percent': (-0.05, -0.03),
    'recovery_duration_months': (48, 72),
    'house_price_drop': (-0.25, -0.15),
    'trade_balance_change': (-0.03, -0.01),
    'gov_debt_increase': (0.15, 0.25)
}

def run_simulation(params):
    """Run the economic model simulation"""
    model = EconomicModel(**params)
    for _ in range(params['months']):
        model.step()
    return model.datacollector.get_model_vars_dataframe()

def validate_results(results, scenario_name):
    """Validate simulation results against historical benchmarks"""
    if scenario_name != "2008_crisis":
        return {}
    
    validation_report = {}
    metrics = VALIDATION_METRICS_2008
    
    # Unemployment validation
    peak_unemployment = results['Unemployment'].max()
    min_exp, max_exp = metrics['unemployment_peak']
    validation_report['Unemployment Peak'] = {
        'Simulated': f"{peak_unemployment:.1%}",
        'Target': f"{min_exp:.0%}-{max_exp:.0%}",
        'Pass': min_exp <= peak_unemployment <= max_exp
    }
    
    # GDP validation
    pre_crisis_gdp = results['GDP'].iloc[:SCENARIO_2008['crisis_start_month']].max()
    trough_gdp = results['GDP'].iloc[SCENARIO_2008['crisis_start_month']:].min()
    gdp_drop = (trough_gdp - pre_crisis_gdp) / pre_crisis_gdp
    min_exp, max_exp = metrics['gdp_drop_percent']
    validation_report['GDP Drop'] = {
        'Simulated': f"{gdp_drop:.1%}",
        'Target': f"{min_exp:.0%}-{max_exp:.0%}",
        'Pass': min_exp <= gdp_drop <= max_exp
    }
    
    # Housing market validation
    peak_hpi = results['House_Price_Index'].iloc[:SCENARIO_2008['crisis_start_month']].max()
    trough_hpi = results['House_Price_Index'].iloc[SCENARIO_2008['crisis_start_month']:].min()
    hpi_drop = (trough_hpi - peak_hpi) / peak_hpi
    min_exp, max_exp = metrics['house_price_drop']
    validation_report['House Price Drop'] = {
        'Simulated': f"{hpi_drop:.1%}",
        'Target': f"{min_exp:.0%}-{max_exp:.0%}",
        'Pass': min_exp <= hpi_drop <= max_exp
    }
    
    return validation_report

# Streamlit App Configuration
st.set_page_config(layout="wide", page_title="Advanced Economic Simulator")
st.title("🏛️ Advanced Economic Simulator with Validation")

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'last_model' not in st.session_state:
    st.session_state.last_model = None

# Control Panel Sidebar
with st.sidebar:
    st.header("⚙️ Simulation Mode")
    run_mode = st.radio("Select Mode", ["Freestyle", "Validate: 2008 Crisis"])
    
    params = {'scenario': None}
    if run_mode == "Validate: 2008 Crisis":
        st.info("Running in validation mode. Parameters are calibrated to the 2008 crisis.")
        params['scenario'] = SCENARIO_2008
        params['months'] = 72
        params['num_households'] = 1000
        params['num_firms'] = 100
        params['network_density'] = 0.05
        params['inflation_target'] = 0.02
        params['supply_shock_prob'] = 0
        params['demand_shock_prob'] = 0
        params['financial_shock_prob'] = 0
        params['policy_shock_prob'] = 0
        params['simulation_name'] = "2008 Crisis Validation"
    else:
        st.header("⚙️ Simulation Parameters")
        params['simulation_name'] = st.text_input("Simulation Name", f"Run {len(st.session_state.results_history) + 1}")
        params['months'] = st.slider("Duration (months)", 12, 360, 180)
        params['num_households'] = st.slider("Number of Households", 100, 5000, 1000)
        params['num_firms'] = st.slider("Number of Firms", 10, 500, 100)
        params['network_density'] = st.slider("Social Network Density", 0.0, 0.2, 0.05, 0.01)
        params['inflation_target'] = st.slider("Inflation Target (%)", 0.0, 5.0, 2.0, 0.1) / 100
        st.header("⚡️ Shock Probabilities")
        params['supply_shock_prob'] = st.slider("Supply Shock", 0.0, 0.1, 0.02, 0.005)
        params['demand_shock_prob'] = st.slider("Demand Shock", 0.0, 0.1, 0.02, 0.005)
        params['financial_shock_prob'] = st.slider("Financial Crisis", 0.0, 0.1, 0.01, 0.005)
        params['policy_shock_prob'] = st.slider("Policy Uncertainty", 0.0, 0.1, 0.01, 0.005)
    
    col1, col2 = st.columns(2)
    with col1: 
        run_button = st.button("▶️ Run Simulation", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.results_history = []
            st.session_state.last_model = None
            st.rerun()
    
    # Policy Tools Section
    if st.session_state.last_model is not None:
        st.header("🏛️ Policy Tools")
        if st.button("💵 Implement QE ($1B)"):
            st.session_state.last_model.policy_tools.implement_quantitative_easing(1e9)
            st.success("Quantitative Easing implemented!")
            
        if st.button("🧾 Implement UBI ($1000/month)"):
            st.session_state.last_model.policy_tools.implement_ubi(1000)
            st.success("Universal Basic Income implemented!")
            
        new_tax_rate = st.slider("Progressive Tax Rate", 0.0, 0.5, 
                                st.session_state.last_model.policy_tools.progressive_tax_rate, 
                                0.01)
        st.session_state.last_model.policy_tools.progressive_tax_rate = new_tax_rate

# Main App Logic
if run_button:
    with st.spinner(f"Running simulation: {params['simulation_name']}..."):
        model = EconomicModel(**params)
        results = run_simulation(params)
        results['name'] = params['simulation_name']
        
        if params.get('scenario'):
            results['validation'] = validate_results(results, params['scenario']['name'])
        
        st.session_state.results_history.append(results)
        st.session_state.last_model = model
        st.success("Simulation completed!")

# Display Results
if st.session_state.results_history:
    latest_results = st.session_state.results_history[-1]
    
    # Validation Report
    if 'validation' in latest_results and isinstance(latest_results['validation'], dict):
        st.header("✅ Validation Report")
        report = latest_results['validation']
        cols = st.columns(len(report))
        for i, (metric, data) in enumerate(report.items()):
            with cols[i]:
                st.metric(
                    label=metric,
                    value=data['Simulated'],
                    help=f"Target range: {data['Target']}"
                )
                st.markdown(f"**Validation:** {'✔️ Passed' if data['Pass'] else '❌ Failed'}")
        st.markdown("---")
    
    # Dashboard
    st.header("📈 Economic Dashboard")
    
    # Run selection
    all_runs = [df['name'].iloc[0] for df in st.session_state.results_history]
    selected_runs = st.multiselect(
        "Compare runs:",
        options=all_runs,
        default=[all_runs[-1]] if len(all_runs) == 1 else all_runs
    )
    display_data = [df for df in st.session_state.results_history 
                   if df['name'].iloc[0] in selected_runs]
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Macro Indicators", "Sector Dynamics", "Raw Data"])
    
    with tab1:
        # GDP and Unemployment
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        for df in display_data:
            name = df['name'].iloc[0]
            fig1.add_trace(
                go.Scatter(x=df.index, y=df['GDP'], name=f'GDP ({name})'),
                secondary_y=False
            )
            fig1.add_trace(
                go.Scatter(x=df.index, y=df['Unemployment']*100, 
                          name=f'Unemployment ({name})', line=dict(dash='dash')),
                secondary_y=True
            )
        fig1.update_layout(
            title="GDP and Unemployment Rate",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig1.update_yaxes(title_text="GDP", secondary_y=False)
        fig1.update_yaxes(title_text="Unemployment Rate (%)", secondary_y=True)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Inflation and Interest Rates
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        for df in display_data:
            name = df['name'].iloc[0]
            fig2.add_trace(
                go.Scatter(x=df.index, y=df['CPI'], name=f'CPI ({name})'),
                secondary_y=False
            )
            fig2.add_trace(
                go.Scatter(x=df.index, y=df['Interest_Rate']*100, 
                          name=f'Interest Rate ({name})', line=dict(dash='dash')),
                secondary_y=True
            )
        fig2.update_layout(
            title="Inflation (CPI) and Interest Rates",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig2.update_yaxes(title_text="Consumer Price Index", secondary_y=False)
        fig2.update_yaxes(title_text="Interest Rate (%)", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Housing Market
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        for df in display_data:
            name = df['name'].iloc[0]
            fig3.add_trace(
                go.Scatter(x=df.index, y=df['House_Price_Index'], 
                          name=f'House Prices ({name})'),
                secondary_y=False
            )
            fig3.add_trace(
                go.Scatter(x=df.index, y=df['Affordability_Index'], 
                          name=f'Affordability ({name})', line=dict(dash='dash')),
                secondary_y=True
            )
        fig3.update_layout(
            title="Housing Market Dynamics",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig3.update_yaxes(title_text="House Price Index", secondary_y=False)
        fig3.update_yaxes(title_text="Affordability Index", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        # Credit Market
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        for df in display_data:
            name = df['name'].iloc[0]
            fig4.add_trace(
                go.Scatter(x=df.index, y=df['Total_Loans'], name=f'Total Loans ({name})'),
                secondary_y=False
            )
            fig4.add_trace(
                go.Scatter(x=df.index, y=df['Active_Firms'], 
                          name=f'Active Firms ({name})', line=dict(dash='dash')),
                secondary_y=True
            )
        fig4.update_layout(
            title="Credit Market and Firm Dynamics",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig4.update_yaxes(title_text="Total Loans Outstanding", secondary_y=False)
        fig4.update_yaxes(title_text="Number of Active Firms", secondary_y=True)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Inequality and Savings
        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        for df in display_data:
            name = df['name'].iloc[0]
            fig5.add_trace(
                go.Scatter(x=df.index, y=df['Gini'], name=f'Gini Index ({name})'),
                secondary_y=False
            )
            fig5.add_trace(
                go.Scatter(x=df.index, y=df['Total_Deposits'], 
                          name=f'Total Deposits ({name})', line=dict(dash='dash')),
                secondary_y=True
            )
        fig5.update_layout(
            title="Income Inequality (Gini) and Household Savings",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig5.update_yaxes(title_text="Gini Coefficient", secondary_y=False)
        fig5.update_yaxes(title_text="Total Household Deposits", secondary_y=True)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Trade Balance
        fig6 = make_subplots(specs=[[{"secondary_y": True}]])
        for df in display_data:
            name = df['name'].iloc[0]
            fig6.add_trace(
                go.Scatter(x=df.index, y=df['Trade_Balance'], name=f'Trade Balance ({name})'),
                secondary_y=False
            )
            fig6.add_trace(
                go.Scatter(x=df.index, y=df['Exchange_Rate'], 
                          name=f'Exchange Rate ({name})', line=dict(dash='dash')),
                secondary_y=True
            )
        fig6.update_layout(
            title="Trade Balance and Exchange Rate",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig6.update_yaxes(title_text="Trade Balance", secondary_y=False)
        fig6.update_yaxes(title_text="Exchange Rate", secondary_y=True)
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab3:
        st.header("📊 Simulation Data")
        selected_run = st.selectbox(
            "Select run to view data:",
            options=[df['name'].iloc[0] for df in st.session_state.results_history]
        )
        selected_df = next(df for df in st.session_state.results_history 
                          if df['name'].iloc[0] == selected_run)
        
        st.dataframe(selected_df.drop(columns=['name', 'validation'], errors='ignore'))
        
        csv = selected_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"economic_simulation_{selected_run}.csv",
            mime="text/csv"
        )