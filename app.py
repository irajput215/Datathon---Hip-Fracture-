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
# CUSTOM STYLING
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
    
    /* Compact badges */
    .kpi-badge-compact {
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
    
    /* Smaller text in charts */
    .plotly .main-svg {
        font-size: 12px !important;
    }
</style>
""", unsafe_allow_html=True)
# st.markdown("""
# <style>
#     /* Ultra compact mode */
#     .main-header { font-size: 1.5rem !important; }
#     .metric-compact { height: 70px !important; padding: 0.5rem !important; }
#     .metric-value { font-size: 1.3rem !important; }
#     .stTabs [data-baseweb="tab"] { height: 1.8rem !important; padding: 0.2rem 0.5rem !important; }
#     section[data-testid="stSidebar"] { min-width: 200px !important; max-width: 200px !important; }
#     .block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
# </style>
# """, unsafe_allow_html=True)

# --------------------------------------------------------
# DATA LOADING & PROCESSING
# --------------------------------------------------------
@st.cache_data(ttl=300)
def load_and_process_data():
    """Load and preprocess the ANZHFR data"""
    try:
        # Load the main dataset
        df = pd.read_csv("unsw_datathon_2025.csv")
        
        # Basic cleaning
        # Convert categorical columns with value labels from data dictionary
        # Sex mapping
        sex_mapping = {1: 'Male', 2: 'Female', 3: 'Intersex'}
        if 'sex' in df.columns:
            df['sex_label'] = df['sex'].map(sex_mapping).fillna('Unknown')
        
        # Create derived time metrics - using _datediff columns which are in days
        if 'sdatetime_datediff' in df.columns:
            df['time_to_surgery_hours'] = df['sdatetime_datediff'] * 24
        
        if 'arrdatetime_datediff' in df.columns:
            df['ed_time_hours'] = df['arrdatetime_datediff'] * 24
        
        # Length of stay - using datediff columns
        if 'wdisch_datediff' in df.columns:
            df['acute_los_days'] = df['wdisch_datediff']
        
        if 'hdisch_datediff' in df.columns:
            df['hospital_los_days'] = df['hdisch_datediff']
        
        # Create binary indicators for ACSQHC quality indicators
        # Based on data dictionary mappings
        
        # QS1: Cognitive assessment prior to surgery
        if 'cogassess' in df.columns:
            # Values: 1=Not assessed, 2=Assessed and normal, 3=Assessed and abnormal
            df['qs1_cognitive_assessed'] = df['cogassess'].isin([2, 3]).astype(float)
        
        # QS2: Pain management
        if 'painmanage' in df.columns:
            # Values: 1=Given within 30 min, 2=Given after 30 min, 3=Not required - provided by paramedics, 4=Not required - no pain
            df['qs2_pain_managed'] = df['painmanage'].isin([1, 2, 3, 4]).astype(float)
        
        # QS3: Geriatric assessment
        if 'gerimed' in df.columns:
            # Values: 0=No, 1=Yes, 8=No service available, 9=Not known
            df['qs3_geriatric_assessed'] = (df['gerimed'] == 1).astype(float)
        
        # QS4: Surgery within 48 hours
        if 'sdatetime_datediff' in df.columns:
            df['qs4_surgery_within_48h'] = (df['sdatetime_datediff'] <= 2).astype(float)
        
        # QS5: First day walking
        if 'mobil2' in df.columns:
            # Values: 1=No, 2=Yes
            df['qs5_day1_walking'] = (df['mobil2'] == 2).astype(float)
        
        # QS6: Bone protection medication at discharge
        if 'dbonemed1' in df.columns:
            # Values: 1=No, 2=Calcium/VitD only, 3=Bisphosphonates/etc, 4=Prescription given
            df['qs6_bone_medication'] = df['dbonemed1'].isin([2, 3, 4]).astype(float)
        
        # QS7: Return to private residence
        if 'dresidence' in df.columns:
            # Values: 1=Private residence, 2=Residential care, 3=Deceased, 4=Other
            df['qs7_discharge_to_private_residence'] = (df['dresidence'] == 1).astype(float)
        
        # Create month-year for trending - handle missing dates
        if 'sdatetime_year' in df.columns and 'sdatetime_month' in df.columns:
            # Filter out rows with missing year or month
            mask = df['sdatetime_year'].notna() & df['sdatetime_month'].notna()
            df_valid_dates = df[mask].copy()
            
            # Create date strings safely
            date_strings = []
            for idx, row in df_valid_dates.iterrows():
                try:
                    year = int(row['sdatetime_year'])
                    month = int(row['sdatetime_month'])
                    date_strings.append(f"{year}-{month:02d}-01")
                except:
                    date_strings.append(None)
            
            df_valid_dates['surgery_date_str'] = date_strings
            df_valid_dates['surgery_date'] = pd.to_datetime(
                df_valid_dates['surgery_date_str'],
                errors='coerce'
            )
            df_valid_dates['surgery_month'] = df_valid_dates['surgery_date'].dt.to_period('M').dt.to_timestamp()
            
            # Merge back to original dataframe
            df = df.merge(
                df_valid_dates[['surgery_date', 'surgery_month']],
                left_index=True,
                right_index=True,
                how='left'
            )
        else:
            df['surgery_date'] = pd.NaT
            df['surgery_month'] = pd.NaT
        
        # Create age groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 65, 75, 85, 120], 
            labels=['<65', '65-74', '75-84', '85+'],
            right=False
        )
        
        # Map fracture types
        if 'ftype' in df.columns:
            fracture_mapping = {
                1: "Intracapsular undisplaced",
                2: "Intracapsular displaced",
                3: "Per/intertrochanteric",
                4: "Subtrochanteric"
            }
            df['fracture_type_label'] = df['ftype'].map(fracture_mapping)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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

def get_benchmark_status(value, target, threshold=0.05):
    """Determine if metric meets benchmark"""
    if pd.isna(value) or pd.isna(target):
        return "no_data"
    
    if value >= target:
        return "good"
    elif value >= target * 0.9:
        return "fair"
    else:
        return "poor"

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
        
        st.markdown("---")
        st.markdown("### 📅 Time Period")
        
        # Time period filter - handle missing dates
        if 'surgery_date' in df.columns:
            # Filter out NaT values for date range calculation
            valid_dates = df['surgery_date'].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                
                # Use today's date as fallback if no valid dates
                today = datetime.now().date()
                default_start = min_date if not pd.isna(min_date) else today - timedelta(days=365)
                default_end = max_date if not pd.isna(max_date) else today
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("From", value=default_start, min_value=default_start)
                with col2:
                    end_date = st.date_input("To", value=default_end, max_value=default_end)
            else:
                st.warning("No valid surgery dates found in data")
                # Set default dates
                today = datetime.now().date()
                start_date = today - timedelta(days=365)
                end_date = today
        else:
            st.warning("Surgery date column not found")
            today = datetime.now().date()
            start_date = today - timedelta(days=365)
            end_date = today
        
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
        st.markdown("### 📊 Data Quality")
        total_records = len(df[df['ahos_code'] == selected_hospital]) if selected_hospital else len(df)
        st.metric("Hospital Records", total_records)
    
    # ----------------------------------------------------
    # FILTER DATA
    # ----------------------------------------------------
    df_filtered = df.copy()
    
    # Apply hospital filter
    if selected_hospital:
        df_filtered = df_filtered[df_filtered['ahos_code'] == selected_hospital]
    
    # Apply date filter if surgery_date exists
    if 'surgery_date' in df_filtered.columns and 'start_date' in locals() and 'end_date' in locals():
        # Filter out NaT values first
        date_mask = df_filtered['surgery_date'].notna()
        if date_mask.any():
            date_filtered = df_filtered[date_mask].copy()
            df_filtered = date_filtered[
                (date_filtered['surgery_date'].dt.date >= start_date) & 
                (date_filtered['surgery_date'].dt.date <= end_date)
            ]
    
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
    # MAIN DASHBOARD
    # ----------------------------------------------------
    st.markdown(f'<div class="main-header">🏥 {selected_hospital} Clinical Dashboard</div>', unsafe_allow_html=True)
    
    # Display filtered record count
    st.info(f"Showing **{len(df_filtered)}** records for {selected_hospital}")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = len(df_filtered)
        st.metric("Total Cases", total_cases)
    
    with col2:
        median_age = df_filtered['age'].median() if not df_filtered.empty else 0
        st.metric("Median Age", f"{median_age:.1f}")
    
    with col3:
        if 'time_to_surgery_hours' in df_filtered.columns and not df_filtered['time_to_surgery_hours'].isna().all():
            median_time = df_filtered['time_to_surgery_hours'].median()
            st.metric("Time to Surgery", f"{median_time:.1f} hrs")
        else:
            st.metric("Time to Surgery", "N/A")
    
    with col4:
        if 'acute_los_days' in df_filtered.columns and not df_filtered['acute_los_days'].isna().all():
            median_los = df_filtered['acute_los_days'].median()
            st.metric("Acute LOS", f"{median_los:.1f} days")
        else:
            st.metric("Acute LOS", "N/A")
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # ACSQHC QUALITY INDICATORS
    # ----------------------------------------------------
    st.markdown("## 📋 ACSQHC Clinical Quality Indicators")
    
    # Define quality indicators
    quality_indicators = [
        {
            'code': 'QS1',
            'title': 'Cognitive Assessment',
            'metric': 'qs1_cognitive_assessed',
            'target': 0.95,
            'description': 'Cognitive assessment prior to surgery'
        },
        {
            'code': 'QS2',
            'title': 'Pain Management',
            'metric': 'qs2_pain_managed',
            'target': 0.95,
            'description': 'Pain assessment & management'
        },
        {
            'code': 'QS3',
            'title': 'Geriatric Assessment',
            'metric': 'qs3_geriatric_assessed',
            'target': 1.0,
            'description': 'Assessed by geriatric medicine'
        },
        {
            'code': 'QS4',
            'title': 'Timely Surgery',
            'metric': 'qs4_surgery_within_48h',
            'target': 0.80,
            'description': 'Surgery within 48 hours'
        },
        {
            'code': 'QS5',
            'title': 'Early Mobilisation',
            'metric': 'qs5_day1_walking',
            'target': 0.60,
            'description': 'First day walking'
        },
        {
            'code': 'QS6',
            'title': 'Fracture Prevention',
            'metric': 'qs6_bone_medication',
            'target': 0.70,
            'description': 'Bone medication on discharge'
        },
        {
            'code': 'QS7',
            'title': 'Discharge Planning',
            'metric': 'qs7_discharge_to_private_residence',
            'target': 0.50,
            'description': 'Return to private residence'
        }
    ]
    
    # Display quality indicators in rows
    for i in range(0, len(quality_indicators), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(quality_indicators):
                indicator = quality_indicators[i + j]
                with cols[j]:
                    if indicator['metric'] in df_filtered.columns:
                        # Calculate hospital performance
                        hosp_value = calculate_metric(df_filtered, indicator['metric'], 'mean')
                        nat_value = calculate_metric(df_national, indicator['metric'], 'mean')
                        
                        if hosp_value is not None:
                            hosp_pct = hosp_value * 100
                            nat_pct = nat_value * 100 if nat_value else 0
                            
                            # Determine status
                            status = get_benchmark_status(hosp_value, indicator['target'])
                            
                            # Display metric
                            st.metric(
                                f"{indicator['code']}",
                                f"{hosp_pct:.1f}%",
                                delta=f"{hosp_pct - nat_pct:+.1f}%",
                                help=indicator['description']
                            )
                            
                            # Status badge
                            if status == "good":
                                st.markdown('<span class="kpi-badge benchmark-good">✓ Target</span>', unsafe_allow_html=True)
                            elif status == "fair":
                                st.markdown('<span class="kpi-badge benchmark-fair">⚠ Near</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="kpi-badge benchmark-poor">✗ Below</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # ACUTE CARE METRICS
    # ----------------------------------------------------
    st.markdown("## ⚡ Acute Care Outcome Measures")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Time Metrics", "Length of Stay", "Patient Volume"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Time to surgery distribution
            if 'time_to_surgery_hours' in df_filtered.columns and not df_filtered['time_to_surgery_hours'].isna().all():
                # Remove outliers for better visualization
                q1 = df_filtered['time_to_surgery_hours'].quantile(0.25)
                q3 = df_filtered['time_to_surgery_hours'].quantile(0.75)
                iqr = q3 - q1
                df_plot = df_filtered[
                    (df_filtered['time_to_surgery_hours'] >= q1 - 1.5*iqr) & 
                    (df_filtered['time_to_surgery_hours'] <= q3 + 1.5*iqr)
                ]
                
                fig = px.histogram(
                    df_plot,
                    x='time_to_surgery_hours',
                    nbins=30,
                    title="Time to Surgery Distribution",
                    labels={'time_to_surgery_hours': 'Hours', 'count': 'Patients'}
                )
                fig.add_vline(
                    x=48,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="48-hour target"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Surgery within 48 hours over time
            if 'surgery_month' in df_filtered.columns and 'qs4_surgery_within_48h' in df_filtered.columns:
                # Filter out NaT values
                valid_months = df_filtered['surgery_month'].notna()
                if valid_months.any():
                    df_valid = df_filtered[valid_months].copy()
                    monthly_performance = df_valid.groupby('surgery_month')['qs4_surgery_within_48h'].mean() * 100
                    monthly_performance = monthly_performance.reset_index()
                    
                    if not monthly_performance.empty:
                        fig = px.line(
                            monthly_performance,
                            x='surgery_month',
                            y='qs4_surgery_within_48h',
                            title="Monthly % Surgery within 48 hours",
                            markers=True
                        )
                        fig.add_hline(
                            y=80,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="80% target"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Acute LOS distribution
            if 'acute_los_days' in df_filtered.columns and not df_filtered['acute_los_days'].isna().all():
                fig = px.box(
                    df_filtered,
                    y='acute_los_days',
                    title="Acute Length of Stay Distribution",
                    labels={'acute_los_days': 'Days'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hospital LOS distribution
            if 'hospital_los_days' in df_filtered.columns and not df_filtered['hospital_los_days'].isna().all():
                fig = px.box(
                    df_filtered,
                    y='hospital_los_days',
                    title="Hospital Length of Stay Distribution",
                    labels={'hospital_los_days': 'Days'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Monthly case volume
        if 'surgery_month' in df_filtered.columns:
            # Filter out NaT values
            valid_months = df_filtered['surgery_month'].notna()
            if valid_months.any():
                df_valid = df_filtered[valid_months].copy()
                monthly_cases = df_valid.groupby('surgery_month').size().reset_index(name='cases')
                
                if not monthly_cases.empty:
                    fig = px.bar(
                        monthly_cases,
                        x='surgery_month',
                        y='cases',
                        title="Monthly Case Volume",
                        labels={'surgery_month': 'Month', 'cases': 'Number of Cases'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # PATIENT DEMOGRAPHICS
    # ----------------------------------------------------
    st.markdown("## 👥 Patient Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        if 'age' in df_filtered.columns:
            fig = px.histogram(
                df_filtered,
                x='age',
                nbins=20,
                title="Age Distribution",
                labels={'age': 'Age', 'count': 'Number of Patients'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sex distribution
        if 'sex_label' in df_filtered.columns:
            sex_dist = df_filtered['sex_label'].value_counts()
            if not sex_dist.empty:
                fig = px.pie(
                    values=sex_dist.values,
                    names=sex_dist.index,
                    title="Sex Distribution",
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fracture type distribution
        if 'fracture_type_label' in df_filtered.columns:
            fracture_counts = df_filtered['fracture_type_label'].value_counts()
            if not fracture_counts.empty:
                fig = px.bar(
                    x=fracture_counts.index,
                    y=fracture_counts.values,
                    title="Fracture Type Distribution",
                    labels={'x': 'Fracture Type', 'y': 'Count'},
                    color=fracture_counts.values,
                    color_continuous_scale='viridis'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # ASA Grade distribution
        if 'asa' in df_filtered.columns:
            asa_counts = df_filtered['asa'].value_counts().sort_index()
            if not asa_counts.empty:
                fig = px.bar(
                    x=asa_counts.index.astype(str),
                    y=asa_counts.values,
                    title="ASA Grade Distribution",
                    labels={'x': 'ASA Grade', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # MORTALITY OUTCOMES
    # ----------------------------------------------------
    st.markdown("## 📊 Mortality Outcomes")
    
    mortality_cols = st.columns(4)
    
    mortality_metrics = [
        ('mort30d', '30-day', 2),  # 2 = Deceased in data dictionary
        ('mort90d', '90-day', 2),
        ('mort120d', '120-day', 2),
        ('mort365d', '365-day', 2)
    ]
    
    for idx, (col_name, period, deceased_code) in enumerate(mortality_metrics):
        if col_name in df_filtered.columns:
            with mortality_cols[idx]:
                # Filter out NaN values
                valid_data = df_filtered[col_name].dropna()
                if not valid_data.empty:
                    # Calculate mortality rate
                    mortality_rate = (valid_data == deceased_code).mean() * 100
                    
                    # National comparison
                    nat_valid = df_national[col_name].dropna()
                    if not nat_valid.empty:
                        nat_mortality = (nat_valid == deceased_code).mean() * 100
                        delta_value = mortality_rate - nat_mortality
                    else:
                        nat_mortality = 0
                        delta_value = 0
                    
                    st.metric(
                        f"{period} Mortality",
                        f"{mortality_rate:.1f}%",
                        delta=f"{delta_value:+.1f}%",
                        delta_color="inverse" if delta_value > 0 else "normal"
                    )
                else:
                    st.metric(f"{period} Mortality", "N/A")
    
    # ----------------------------------------------------
    # DATA EXPLORER
    # ----------------------------------------------------
    with st.expander("📁 Data Explorer & Export"):
        st.markdown("### Raw Data Preview")
        
        # Select columns to display
        available_cols = df_filtered.columns.tolist()
        # Remove complex columns for cleaner display
        simple_cols = [col for col in available_cols if not col.startswith('qs') and col not in ['surgery_date', 'surgery_month']]
        default_cols = ['ahos_code', 'age', 'sex_label', 'fracture_type_label', 
                       'time_to_surgery_hours', 'acute_los_days', 'asa']
        
        # Filter to only include columns that exist
        default_cols = [col for col in default_cols if col in simple_cols]
        
        selected_cols = st.multiselect(
            "Select columns to display",
            options=simple_cols,
            default=default_cols[:5]  # First 5 valid columns
        )
        
        if selected_cols:
            # Display data
            display_df = df_filtered[selected_cols].head(100)  # Limit to 100 rows for performance
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300
            )
            
            # Export button
            csv_data = df_filtered[selected_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"anzhfr_{selected_hospital}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Data quality info
        st.markdown("### 📊 Data Quality Summary")
        quality_cols = st.columns(3)
        
        with quality_cols[0]:
            total_records = len(df_filtered)
            st.metric("Total Records", total_records)
        
        with quality_cols[1]:
            complete_cases = df_filtered.notna().all(axis=1).sum()
            st.metric("Complete Cases", f"{complete_cases} ({complete_cases/total_records*100:.1f}%)")
        
        with quality_cols[2]:
            if 'time_to_surgery_hours' in df_filtered.columns:
                missing_time = df_filtered['time_to_surgery_hours'].isna().sum()
                st.metric("Missing Surgery Time", f"{missing_time} ({missing_time/total_records*100:.1f}%)")

if __name__ == "__main__":
    main()