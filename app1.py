import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
st.set_page_config(
    page_title="ANZHFR Clinical Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# --------------------------------------------------------
# COMPACT CUSTOM CSS
# --------------------------------------------------------
st.markdown("""
<style>
    /* Base styles */
    html, body, [class*="css"] {
        font-size: 14px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Compact header */
    .main-header {
        font-size: 1.8rem !important;
        color: #1E3A8A;
        margin-bottom: 0.25rem !important;
        font-weight: 600;
        padding-top: 0.5rem;
    }
    
    /* Year selector styling */
    .year-selector {
        background: #F3F4F6;
        padding: 0.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    
    /* Compact metric cards */
    .metric-compact {
        background: #F8FAFC;
        padding: 0.75rem !important;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 0.5rem;
        height: 85px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 1.5rem !important;
        font-weight: 700;
        color: #1F2937;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem !important;
        color: #6B7280;
        margin-bottom: 0.25rem;
    }
    
    .metric-delta {
        font-size: 0.8rem !important;
        margin-top: 0.25rem;
    }
    
    /* Benchmark badges */
    .benchmark-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem !important;
        border-radius: 12px;
        font-size: 0.75rem !important;
        font-weight: 600;
        margin: 0.1rem;
    }
    
    .benchmark-good { background-color: #D1FAE5; color: #065F46; border: 1px solid #10B981; }
    .benchmark-fair { background-color: #FEF3C7; color: #92400E; border: 1px solid #F59E0B; }
    .benchmark-poor { background-color: #FEE2E2; color: #991B1B; border: 1px solid #EF4444; }
    
    /* Reduce spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 2rem !important;
        padding: 0.25rem 0.75rem !important;
        font-size: 0.9rem !important;
    }
    
    /* Compact sidebar */
    section[data-testid="stSidebar"] {
        min-width: 220px !important;
        max-width: 220px !important;
    }
    
    /* Plotly chart sizing */
    .js-plotly-plot {
        margin: 0 !important;
    }
    
    /* Reduce element spacing */
    div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact column spacing */
    .stColumn {
        padding: 0.25rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# DATA LOADING & PROCESSING
# --------------------------------------------------------
@st.cache_data(ttl=300)
def load_and_process_data():
    """Load and preprocess the ANZHFR data"""
    try:
        # Load the main dataset
        df = pd.read_csv("unsw_datathon_2025.csv")
        
        # List of columns to keep (based on your specification)
        columns_to_keep = [
            'tarrdatetime_datediff', 'arrdatetime_datediff', 'depdatetime_datediff',
            'admdatetimeop_datediff', 'sdatetime_datediff', 'gdate_datediff',
            'wdisch_datediff', 'hdisch_datediff', 'age', 'sex', 'ptype',
            'uresidence', 'e_dadmit', 'painassess', 'painmanage', 'tfanalges',
            'ward', 'walk', 'cogassess', 'cogstat', 'addelassess', 'bonemed',
            'passess', 'side', 'afracture', 'ftype', 'asa', 'frailty', 'delay',
            'anaesth', 'analges', 'consult', 'wbear', 'mobil', 'pulcers', 'fassess',
            'dbonemed1', 'delassess', 'malnutrition', 'mobil2', 'wdest',
            'dresidence', 'fresidence2', 'fop2', 'ahos_code', 'surg', 'gerimed',
            'mort30d', 'mort90d', 'mort120d', 'mort365d'
        ]
        
        # Filter to keep only specified columns that exist in the dataframe
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[existing_columns].copy()
        
        # Create fracture time (earliest available time)
        df["fracture_time"] = df[["admdatetimeop_datediff", "tarrdatetime_datediff", "arrdatetime_datediff"]].bfill(axis=1).iloc[:, 0]
        
        # Fill missing time values using forward fill logic
        df['tarrdatetime_datediff'] = df['tarrdatetime_datediff'].fillna(df['admdatetimeop_datediff'])
        df['arrdatetime_datediff'] = df['arrdatetime_datediff'].fillna(df['tarrdatetime_datediff'])
        df['sdatetime_datediff'] = df['sdatetime_datediff'].fillna(df['arrdatetime_datediff'])
        df['depdatetime_datediff'] = df['depdatetime_datediff'].fillna(df['sdatetime_datediff'])
        df['gdate_datediff'] = df['gdate_datediff'].fillna(df['sdatetime_datediff'])
        df['wdisch_datediff'] = df['wdisch_datediff'].fillna(df['sdatetime_datediff'])
        df['hdisch_datediff'] = df['hdisch_datediff'].fillna(df['wdisch_datediff'])
        
        # Calculate key time metrics (in days)
        df['ed_time_days'] = (df['depdatetime_datediff'] - df['fracture_time']).fillna(0)
        df['ward_time_days'] = (df['wdisch_datediff'] - df['fracture_time']).fillna(0)
        df['time_to_surgery_days'] = (df['sdatetime_datediff'] - df['fracture_time']).fillna(0)
        df['gdate_time_days'] = (df['gdate_datediff'] - df['fracture_time']).fillna(0)
        df['discharge_time_days'] = (df['hdisch_datediff'] - df['fracture_time']).fillna(0)
        
        # Convert to hours for display
        df['time_to_surgery_hours'] = df['time_to_surgery_days'] * 24
        df['ed_time_hours'] = df['ed_time_days'] * 24
        df['acute_los_days'] = df['wdisch_datediff'].fillna(0)  # Acute length of stay
        
        # ACSQHC Quality Indicators
        # QS1: Cognitive assessment prior to surgery
        if 'cogassess' in df.columns:
            # Values: 1=Not assessed, 2=Assessed and normal, 3=Assessed and abnormal
            df['qs1_cognitive_assessed'] = df['cogassess'].isin([2, 3]).astype(float)
        
        # QS2: Pain management
        if 'painmanage' in df.columns:
            # Values: 1=Given within 30 min, 2=Given after 30 min, 3=Not required, 4=Not required
            df['qs2_pain_managed'] = df['painmanage'].isin([1, 2, 3, 4]).astype(float)
        
        # QS3: Geriatric assessment
        if 'gerimed' in df.columns:
            # Values: 0=No, 1=Yes, 8=No service, 9=Not known
            df['qs3_geriatric_assessed'] = (df['gerimed'] == 1).astype(float)
        
        # QS4: Surgery within 48 hours
        df['qs4_surgery_within_48h'] = (df['time_to_surgery_hours'] <= 48).astype(float)
        
        # QS5: First day walking
        if 'mobil2' in df.columns:
            # Values: 1=No, 2=Yes
            df['qs5_day1_walking'] = (df['mobil2'] == 2).astype(float)
        
        # QS6: Bone protection medication at discharge
        if 'dbonemed1' in df.columns:
            # Values: 1=No, 2=Calcium/VitD only, 3=Bisphosphonates, 4=Prescription
            df['qs6_bone_medication'] = df['dbonemed1'].isin([2, 3, 4]).astype(float)
        
        # QS7: Return to private residence
        if 'dresidence' in df.columns:
            # Values: 1=Private residence, 2=Residential care, 3=Deceased, 4=Other
            df['qs7_discharge_to_private_residence'] = (df['dresidence'] == 1).astype(float)
        
        # Create year column for trending (using surgery year if available, otherwise fracture year)
        if 'sdatetime_year' in df.columns:
            df['year'] = df['sdatetime_year'].fillna(df['admdatetimeop_year'] if 'admdatetimeop_year' in df.columns else datetime.now().year)
        else:
            # Create synthetic year based on index if no year column exists
            df['year'] = 2022 + (df.index % 5)  # Example: spread across 5 years
        
        # Convert year to integer
        df['year'] = df['year'].astype(int)
        
        # Create age groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 65, 75, 85, 120], 
            labels=['<65', '65-74', '75-84', '85+'],
            right=False
        )
        
        # Sex mapping
        sex_mapping = {1: 'Male', 2: 'Female', 3: 'Intersex'}
        if 'sex' in df.columns:
            df['sex_label'] = df['sex'].map(sex_mapping).fillna('Unknown')
        
        # Fracture type mapping
        if 'ftype' in df.columns:
            fracture_mapping = {
                1: "Intracapsular",
                2: "Intracapsular Displaced", 
                3: "Per/intertrochanteric",
                4: "Subtrochanteric"
            }
            df['fracture_type_label'] = df['ftype'].map(fracture_mapping)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# --------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------
def calculate_metric(df, column, metric_type='mean'):
    """Calculate metrics with error handling"""
    if column not in df.columns or df[column].isna().all():
        return None
    
    if metric_type == 'mean':
        return df[column].mean()
    elif metric_type == 'median':
        return df[column].median()
    elif metric_type == 'count':
        return len(df)
    elif metric_type == 'percentage':
        # For binary columns (0/1)
        return df[column].mean() * 100

def get_benchmark_status(value, target):
    """Determine if metric meets benchmark"""
    if pd.isna(value) or pd.isna(target):
        return "no_data"
    
    if value >= target:
        return "good"
    elif value >= target * 0.9:
        return "fair"
    else:
        return "poor"

def create_year_selector(df, selected_hospital):
    """Create year selection interface"""
    years = sorted(df['year'].unique())
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="year-selector">', unsafe_allow_html=True)
        selected_years = st.multiselect(
            "Select Years to Compare:",
            options=years,
            default=years[-3:] if len(years) >= 3 else years,  # Default to last 3 years
            key="year_selector"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        compare_mode = st.radio(
            "View Mode:",
            ["Single Year", "Multi-Year Compare"],
            horizontal=True,
            key="compare_mode"
        )
    
    with col3:
        if compare_mode == "Single Year":
            selected_year = st.selectbox(
                "Select Year:",
                options=years,
                index=len(years)-1,  # Default to most recent year
                key="single_year"
            )
            selected_years = [selected_year]
    
    return selected_years, compare_mode

# --------------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------------
def main():
    # Load data
    df = load_and_process_data()
    
    if df is None:
        st.error("Unable to load data. Please check the data file.")
        st.stop()
    
    # ----------------------------------------------------
    # SIDEBAR - FILTERS
    # ----------------------------------------------------
    with st.sidebar:
        st.markdown("### 🏥 Hospital Selection")
        
        # Hospital selection
        if 'ahos_code' in df.columns:
            hospitals = sorted(df['ahos_code'].unique())
            selected_hospital = st.selectbox(
                "Select Hospital",
                hospitals,
                index=0 if len(hospitals) > 0 else 0
            )
        else:
            st.error("Hospital code column not found")
            st.stop()
        
        st.markdown("---")
        st.markdown("### 🔍 Patient Filters")
        
        # Age filter
        if 'age_group' in df.columns:
            age_groups = st.multiselect(
                "Age Groups",
                options=df['age_group'].dropna().unique().tolist(),
                default=df['age_group'].dropna().unique().tolist()
            )
        
        # Sex filter
        if 'sex_label' in df.columns:
            sexes = st.multiselect(
                "Sex",
                options=df['sex_label'].dropna().unique().tolist(),
                default=df['sex_label'].dropna().unique().tolist()
            )
        
        # Fracture type filter
        if 'fracture_type_label' in df.columns:
            fracture_options = st.multiselect(
                "Fracture Type",
                options=df['fracture_type_label'].dropna().unique().tolist(),
                default=df['fracture_type_label'].dropna().unique().tolist()
            )
        
        # Data quality info
        st.markdown("---")
        st.markdown("### 📊 Data Summary")
        hospital_cases = len(df[df['ahos_code'] == selected_hospital])
        st.metric(f"{selected_hospital} Cases", hospital_cases)
    
    # ----------------------------------------------------
    # FILTER DATA
    # ----------------------------------------------------
    df_filtered = df[df['ahos_code'] == selected_hospital].copy()
    
    # Apply other filters
    if 'age_groups' in locals() and age_groups:
        df_filtered = df_filtered[df_filtered['age_group'].isin(age_groups)]
    
    if 'sexes' in locals() and sexes:
        df_filtered = df_filtered[df_filtered['sex_label'].isin(sexes)]
    
    if 'fracture_options' in locals() and fracture_options:
        df_filtered = df_filtered[df_filtered['fracture_type_label'].isin(fracture_options)]
    
    # Get national data for comparison
    df_national = df
    
    # ----------------------------------------------------
    # YEAR SELECTION
    # ----------------------------------------------------
    selected_years, compare_mode = create_year_selector(df_filtered, selected_hospital)
    
    if not selected_years:
        st.warning("Please select at least one year")
        st.stop()
    
    # Filter data for selected years
    df_selected_years = df_filtered[df_filtered['year'].isin(selected_years)]
    
    # ----------------------------------------------------
    # MAIN DASHBOARD HEADER
    # ----------------------------------------------------
    st.markdown(f'<div class="main-header">🏥 {selected_hospital} Performance Dashboard</div>', unsafe_allow_html=True)
    
    # Year info
    year_text = f"{selected_years[0]}" if len(selected_years) == 1 else f"{len(selected_years)} years ({min(selected_years)}-{max(selected_years)})"
    st.caption(f"📅 Showing data for: {year_text} | 📊 {len(df_selected_years):,} records")
    
    # ----------------------------------------------------
    # KEY PERFORMANCE METRICS - YEAR FOCUS
    # ----------------------------------------------------
    st.markdown("## 📈 Key Performance Metrics")
    
    if compare_mode == "Single Year":
        # Single year view - show metrics with national comparison
        selected_year = selected_years[0]
        df_year = df_selected_years[df_selected_years['year'] == selected_year]
        df_national_year = df_national[df_national['year'] == selected_year]
        
        kpi_cols = st.columns(6)
        
        with kpi_cols[0]:
            total_cases = len(df_year)
            nat_cases = len(df_national_year)
            st.metric("Total Cases", f"{total_cases:,}", delta=f"{(total_cases/len(df_national)*100):.1f}% of national")
        
        with kpi_cols[1]:
            median_age = df_year['age'].median() if not df_year.empty else 0
            nat_age = df_national_year['age'].median() if not df_national_year.empty else 0
            st.metric("Median Age", f"{median_age:.0f}", delta=f"{(median_age - nat_age):+.0f}")
        
        with kpi_cols[2]:
            if 'time_to_surgery_hours' in df_year.columns:
                median_time = df_year['time_to_surgery_hours'].median()
                nat_time = df_national_year['time_to_surgery_hours'].median() if 'time_to_surgery_hours' in df_national_year.columns else 0
                status = "✅" if median_time <= 48 else "⚠️"
                st.metric(f"{status} Surgery Time", f"{median_time:.1f}h", delta=f"{(median_time - nat_time):+.1f}h")
        
        with kpi_cols[3]:
            if 'acute_los_days' in df_year.columns:
                median_los = df_year['acute_los_days'].median()
                nat_los = df_national_year['acute_los_days'].median() if 'acute_los_days' in df_national_year.columns else 0
                st.metric("Acute LOS", f"{median_los:.1f}d", delta=f"{(median_los - nat_los):+.1f}d")
        
        with kpi_cols[4]:
            if 'mort30d' in df_year.columns:
                mortality_rate = (df_year['mort30d'] == 2).mean() * 100
                nat_mortality = (df_national_year['mort30d'] == 2).mean() * 100 if 'mort30d' in df_national_year.columns else 0
                delta_color = "inverse" if mortality_rate > nat_mortality else "normal"
                st.metric("30-day Mortality", f"{mortality_rate:.1f}%", 
                         delta=f"{(mortality_rate - nat_mortality):+.1f}%",
                         delta_color=delta_color)
        
        with kpi_cols[5]:
            if 'qs4_surgery_within_48h' in df_year.columns:
                surgery_48h = df_year['qs4_surgery_within_48h'].mean() * 100
                nat_48h = df_national_year['qs4_surgery_within_48h'].mean() * 100 if 'qs4_surgery_within_48h' in df_national_year.columns else 0
                status = "✅" if surgery_48h >= 80 else "⚠️"
                st.metric(f"{status} Surgery <48h", f"{surgery_48h:.1f}%", delta=f"{(surgery_48h - nat_48h):+.1f}%")
    
    else:
        # Multi-year comparison
        st.markdown("#### Year-over-Year Comparison")
        
        # Create comparison table
        year_metrics = []
        for year in selected_years:
            df_year = df_selected_years[df_selected_years['year'] == year]
            if len(df_year) > 0:
                metrics = {
                    'Year': year,
                    'Cases': len(df_year),
                    'Median Age': df_year['age'].median(),
                    'Surgery Time (h)': df_year['time_to_surgery_hours'].median() if 'time_to_surgery_hours' in df_year.columns else None,
                    'Acute LOS (d)': df_year['acute_los_days'].median() if 'acute_los_days' in df_year.columns else None,
                    '30-day Mortality %': (df_year['mort30d'] == 2).mean() * 100 if 'mort30d' in df_year.columns else None,
                    'Surgery <48h %': df_year['qs4_surgery_within_48h'].mean() * 100 if 'qs4_surgery_within_48h' in df_year.columns else None
                }
                year_metrics.append(metrics)
        
        if year_metrics:
            metrics_df = pd.DataFrame(year_metrics)
            st.dataframe(metrics_df.set_index('Year'), use_container_width=True)
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # ACSQHC QUALITY INDICATORS - YEAR TRENDS
    # ----------------------------------------------------
    st.markdown("## 📊 ACSQHC Quality Indicators - Yearly Trends")
    
    # Define quality indicators
    quality_indicators = [
        {'code': 'QS1', 'title': 'Cognitive Assessment', 'metric': 'qs1_cognitive_assessed', 'target': 0.95},
        {'code': 'QS2', 'title': 'Pain Management', 'metric': 'qs2_pain_managed', 'target': 0.95},
        {'code': 'QS3', 'title': 'Geriatric Assessment', 'metric': 'qs3_geriatric_assessed', 'target': 1.0},
        {'code': 'QS4', 'title': 'Surgery <48h', 'metric': 'qs4_surgery_within_48h', 'target': 0.80},
        {'code': 'QS5', 'title': 'Day 1 Walking', 'metric': 'qs5_day1_walking', 'target': 0.60},
        {'code': 'QS6', 'title': 'Bone Medication', 'metric': 'qs6_bone_medication', 'target': 0.70},
        {'code': 'QS7', 'title': 'Home Discharge', 'metric': 'qs7_discharge_to_private_residence', 'target': 0.50}
    ]
    
    # Create tabs for different views
    summary_tab, trend_tab, detail_tab = st.tabs(["Summary", "Trends", "Details"])
    
    with summary_tab:
        # Current year performance
        if compare_mode == "Single Year":
            current_year = selected_years[0]
            df_current = df_selected_years[df_selected_years['year'] == current_year]
            
            cols = st.columns(7)
            for idx, qs in enumerate(quality_indicators):
                with cols[idx]:
                    if qs['metric'] in df_current.columns:
                        value = df_current[qs['metric']].mean() * 100
                        status = get_benchmark_status(df_current[qs['metric']].mean(), qs['target'])
                        
                        st.markdown(f"**{qs['code']}**")
                        st.markdown(f"### {value:.1f}%")
                        
                        # Progress bar
                        progress = min(value / 100, 1.0)
                        st.progress(progress)
                        
                        # Status badge
                        badge_class = f"benchmark-badge benchmark-{status}"
                        st.markdown(f'<span class="{badge_class}">{status.upper()}</span>', unsafe_allow_html=True)
    
    with trend_tab:
        # Yearly trends for key indicators
        st.markdown("#### Yearly Performance Trends")
        
        trend_metric = st.selectbox(
            "Select Metric to View Trend:",
            options=['qs4_surgery_within_48h', 'qs5_day1_walking', 'qs6_bone_medication'],
            format_func=lambda x: {
                'qs4_surgery_within_48h': 'Surgery within 48 hours',
                'qs5_day1_walking': 'First Day Walking',
                'qs6_bone_medication': 'Bone Medication at Discharge'
            }[x]
        )
        
        if trend_metric in df_selected_years.columns:
            # Calculate yearly performance
            yearly_perf = df_selected_years.groupby('year')[trend_metric].mean() * 100
            yearly_perf = yearly_perf.reset_index()
            
            # Get national comparison
            national_perf = df_national.groupby('year')[trend_metric].mean() * 100
            national_perf = national_perf.reset_index()
            
            fig = go.Figure()
            
            # Hospital trend
            fig.add_trace(go.Scatter(
                x=yearly_perf['year'],
                y=yearly_perf[trend_metric],
                mode='lines+markers',
                name=f'{selected_hospital}',
                line=dict(color='#3B82F6', width=3)
            ))
            
            # National trend
            fig.add_trace(go.Scatter(
                x=national_perf['year'],
                y=national_perf[trend_metric],
                mode='lines+markers',
                name='National Average',
                line=dict(color='#9CA3AF', width=2, dash='dash')
            ))
            
            # Add target line
            target_value = next((qs['target'] * 100 for qs in quality_indicators if qs['metric'] == trend_metric), 0)
            fig.add_hline(
                y=target_value,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Target: {target_value:.0f}%"
            )
            
            fig.update_layout(
                title=f"{trend_metric.replace('_', ' ').title()} - Yearly Trend",
                xaxis_title="Year",
                yaxis_title="Percentage (%)",
                yaxis_range=[0, 100],
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with detail_tab:
        # Detailed comparison table
        st.markdown("#### Detailed Yearly Comparison")
        
        comparison_data = []
        for year in selected_years:
            df_year = df_selected_years[df_selected_years['year'] == year]
            if len(df_year) > 0:
                year_data = {'Year': year, 'Cases': len(df_year)}
                for qs in quality_indicators:
                    if qs['metric'] in df_year.columns:
                        year_data[qs['code']] = df_year[qs['metric']].mean() * 100
                comparison_data.append(year_data)
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df.set_index('Year'), use_container_width=True)
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # TIME METRICS ANALYSIS - YEAR FOCUS
    # ----------------------------------------------------
    st.markdown("## ⏱ Time Metrics Analysis")
    
    time_tab1, time_tab2, time_tab3 = st.tabs(["Surgery Timing", "Length of Stay", "Process Times"])
    
    with time_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Time to surgery distribution by year
            if 'time_to_surgery_hours' in df_selected_years.columns:
                fig = px.box(
                    df_selected_years,
                    x='year',
                    y='time_to_surgery_hours',
                    title="Time to Surgery by Year (hours)",
                    points=False
                )
                fig.add_hline(
                    y=48,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="48-hour target"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # % Surgery within 48h by year
            if 'qs4_surgery_within_48h' in df_selected_years.columns:
                yearly_48h = df_selected_years.groupby('year')['qs4_surgery_within_48h'].mean() * 100
                yearly_48h = yearly_48h.reset_index()
                
                fig = px.bar(
                    yearly_48h,
                    x='year',
                    y='qs4_surgery_within_48h',
                    title="% Surgery within 48h by Year",
                    labels={'qs4_surgery_within_48h': 'Percentage (%)'}
                )
                fig.add_hline(
                    y=80,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="80% target"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
    
    with time_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Acute LOS by year
            if 'acute_los_days' in df_selected_years.columns:
                fig = px.box(
                    df_selected_years,
                    x='year',
                    y='acute_los_days',
                    title="Acute Length of Stay by Year (days)",
                    points=False
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Total hospital LOS by year
            if 'discharge_time_days' in df_selected_years.columns:
                fig = px.box(
                    df_selected_years,
                    x='year',
                    y='discharge_time_days',
                    title="Total Hospital Stay by Year (days)",
                    points=False
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
    
    with time_tab3:
        # Process time metrics
        time_metrics = ['ed_time_days', 'ward_time_days', 'gdate_time_days']
        time_labels = ['ED Time', 'Ward Time', 'Geriatric Assessment Time']
        
        yearly_times = []
        for year in selected_years:
            df_year = df_selected_years[df_selected_years['year'] == year]
            year_data = {'Year': year}
            for metric, label in zip(time_metrics, time_labels):
                if metric in df_year.columns:
                    year_data[label] = df_year[metric].median()
            yearly_times.append(year_data)
        
        if yearly_times:
            times_df = pd.DataFrame(yearly_times).set_index('Year')
            
            # Create heatmap style visualization
            fig = go.Figure(data=go.Heatmap(
                z=times_df.values.T,
                x=times_df.index,
                y=times_df.columns,
                colorscale='Viridis',
                colorbar=dict(title="Days")
            ))
            
            fig.update_layout(
                title="Median Process Times by Year (days)",
                height=300,
                xaxis_title="Year",
                yaxis_title="Process"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # PATIENT CHARACTERISTICS - YEAR COMPARISON
    # ----------------------------------------------------
    st.markdown("## 👥 Patient Characteristics - Yearly Comparison")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        # Age distribution by year
        if 'age' in df_selected_years.columns:
            fig = px.violin(
                df_selected_years,
                x='year',
                y='age',
                box=True,
                points=False,
                title="Age Distribution by Year"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sex distribution trend
        if 'sex_label' in df_selected_years.columns:
            yearly_sex = df_selected_years.groupby(['year', 'sex_label']).size().reset_index(name='count')
            yearly_sex_pivot = yearly_sex.pivot(index='year', columns='sex_label', values='count').fillna(0)
            
            fig = px.bar(
                yearly_sex_pivot,
                title="Sex Distribution by Year",
                barmode='stack'
            )
            fig.update_layout(height=250, xaxis_title="Year", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    with demo_col2:
        # Fracture type by year
        if 'fracture_type_label' in df_selected_years.columns:
            yearly_fracture = df_selected_years.groupby(['year', 'fracture_type_label']).size().reset_index(name='count')
            
            fig = px.bar(
                yearly_fracture,
                x='year',
                y='count',
                color='fracture_type_label',
                title="Fracture Types by Year",
                barmode='stack'
            )
            fig.update_layout(height=300, xaxis_title="Year", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # ASA grade trend
        if 'asa' in df_selected_years.columns:
            yearly_asa = df_selected_years.groupby(['year', 'asa']).size().reset_index(name='count')
            
            fig = px.line(
                yearly_asa,
                x='year',
                y='count',
                color='asa',
                title="ASA Grade Distribution Trend",
                markers=True
            )
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # MORTALITY OUTCOMES - YEAR TRENDS
    # ----------------------------------------------------
    st.markdown("## 📉 Mortality Outcomes - Yearly Trends")
    
    mortality_cols = st.columns(4)
    
    mortality_metrics = [
        ('mort30d', '30-day'),
        ('mort90d', '90-day'), 
        ('mort120d', '120-day'),
        ('mort365d', '365-day')
    ]
    
    for idx, (col_name, period) in enumerate(mortality_metrics):
        if col_name in df_selected_years.columns:
            with mortality_cols[idx]:
                # Calculate yearly mortality rates
                yearly_mortality = df_selected_years.groupby('year')[col_name].apply(
                    lambda x: (x == 2).mean() * 100
                ).reset_index(name='mortality_rate')
                
                if not yearly_mortality.empty:
                    fig = px.line(
                        yearly_mortality,
                        x='year',
                        y='mortality_rate',
                        title=f"{period} Mortality Trend",
                        markers=True
                    )
                    fig.update_layout(
                        height=200,
                        margin=dict(t=40, b=20, l=40, r=20),
                        yaxis_title="Mortality Rate (%)",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # ----------------------------------------------------
    # YEARLY SUMMARY & INSIGHTS
    # ----------------------------------------------------
    with st.expander("📋 Yearly Summary & Insights"):
        st.markdown("### Key Insights by Year")
        
        insights_cols = st.columns(3)
        
        with insights_cols[0]:
            st.markdown("#### 🎯 Performance Highlights")
            for year in selected_years:
                df_year = df_selected_years[df_selected_years['year'] == year]
                if len(df_year) > 0:
                    if 'qs4_surgery_within_48h' in df_year.columns:
                        surgery_48h = df_year['qs4_surgery_within_48h'].mean() * 100
                        status = "✅" if surgery_48h >= 80 else "⚠️"
                        st.write(f"**{year}**: {status} {surgery_48h:.1f}% surgery within 48h")
        
        with insights_cols[1]:
            st.markdown("#### 📈 Improvement Areas")
            # Identify metrics below target
            if compare_mode == "Single Year":
                current_year = selected_years[0]
                df_current = df_selected_years[df_selected_years['year'] == current_year]
                
                for qs in quality_indicators:
                    if qs['metric'] in df_current.columns:
                        value = df_current[qs['metric']].mean()
                        if value < qs['target']:
                            gap = (qs['target'] - value) * 100
                            st.write(f"**{qs['code']}**: {gap:.1f}% below target")
        
        with insights_cols[2]:
            st.markdown("#### 📊 Volume Trends")
            yearly_volume = df_selected_years['year'].value_counts().sort_index()
            for year, count in yearly_volume.items():
                prev_year = year - 1
                prev_count = len(df_selected_years[df_selected_years['year'] == prev_year]) if prev_year in df_selected_years['year'].unique() else None
                
                if prev_count:
                    change = ((count - prev_count) / prev_count) * 100
                    arrow = "↑" if change > 0 else "↓"
                    st.write(f"**{year}**: {count} cases ({arrow}{abs(change):.1f}%)")
                else:
                    st.write(f"**{year}**: {count} cases")
        
        # Export option
        st.markdown("---")
        st.markdown("#### 📥 Export Yearly Data")
        
        export_cols = st.columns(2)
        with export_cols[0]:
            # Summary export
            summary_data = []
            for year in selected_years:
                df_year = df_selected_years[df_selected_years['year'] == year]
                summary = {
                    'Year': year,
                    'Total Cases': len(df_year),
                    'Median Age': df_year['age'].median() if not df_year.empty else None,
                    'Median Surgery Time (h)': df_year['time_to_surgery_hours'].median() if 'time_to_surgery_hours' in df_year.columns else None,
                    '30-day Mortality %': (df_year['mort30d'] == 2).mean() * 100 if 'mort30d' in df_year.columns else None
                }
                summary_data.append(summary)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Yearly Summary",
                    data=csv_summary,
                    file_name=f"{selected_hospital}_yearly_summary.csv",
                    mime="text/csv"
                )
        
        with export_cols[1]:
            # Raw data export (filtered)
            if not df_selected_years.empty:
                csv_raw = df_selected_years.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Data",
                    data=csv_raw,
                    file_name=f"{selected_hospital}_filtered_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()