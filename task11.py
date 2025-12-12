import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib

# Page configuration
st.set_page_config(
    page_title="ANZHFR Hip Fracture Care Dashboard",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #e8f4fc;
        border-left: 4px solid #3498db;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    .step-box {
        background-color: #f0f7ff;
        border: 1px solid #d1e3ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data using your exact data structure
@st.cache_data
def load_data():
    # Load the actual CSV file
    df = pd.read_csv("unsw_datathon_2025.csv")
    df["year"] = df[[ "admdatetimeop_year","tarrdatetime_year", "arrdatetime_year",'depdatetime_year',"sdatetime_year",'gdate_year','wdisch_year','hdisch_year' ]].bfill(axis=1).iloc[:, 0]
    df["month"] = df[[ "admdatetimeop_month","tarrdatetime_month", "arrdatetime_month",'depdatetime_month',"sdatetime_month",'gdate_month','wdisch_month','hdisch_month' ]].bfill(axis=1).iloc[:, 0]
        
    # Calculate fracture_time as you did before
    df["fracture_time"] = df[["admdatetimeop_datediff", "tarrdatetime_datediff", "arrdatetime_datediff"]].bfill(axis=1).iloc[:, 0]
    
    # Fill missing time values using forward fill
    df['tarrdatetime_datediff'] = df['tarrdatetime_datediff'].fillna(df['admdatetimeop_datediff'])
    df['arrdatetime_datediff'] = df['arrdatetime_datediff'].fillna(df['tarrdatetime_datediff'])
    df['sdatetime_datediff'] = df['sdatetime_datediff'].fillna(df['arrdatetime_datediff'])
    df['depdatetime_datediff'] = df['depdatetime_datediff'].fillna(df['sdatetime_datediff'])
    df['gdate_datediff'] = df['gdate_datediff'].fillna(df['sdatetime_datediff'])
    df['wdisch_datediff'] = df['wdisch_datediff'].fillna(df['sdatetime_datediff'])
    df['hdisch_datediff'] = df['hdisch_datediff'].fillna(df['wdisch_datediff'])
    
    # Calculate time metrics
    df['ed_time'] = df['depdatetime_datediff'] - df['fracture_time']
    df['ward_time'] = df['wdisch_datediff'] - df['fracture_time']
    df['time_to_surgery'] = df['sdatetime_datediff'] - df['fracture_time']
    df['gdate_time'] = df['gdate_datediff'] - df['fracture_time']
    df['discharge_time'] = df['hdisch_datediff'] - df['fracture_time']
    
    
    
    # Create new residential care indicator
    # First map the numeric codes to meaningful values
    # Based on data dictionary: 1=Home, 2=Residential Aged Care, 3=Other, 4=Unknown
    def map_residence(code):
        if code == 1:
            return 'Home'
        elif code == 2:
            return 'Residential Care'
        elif code == 3:
            return 'Other'
        elif code == 4:
            return 'Unknown'
        else:
            return 'Unknown'
    
    def map_wdest(code):
        # Map wdest codes: 1=Home, 2=Rehab, 3=Residential Care, 4=Other hospital, 5=Died, 6=Other, 7=Unknown
        if code == 1:
            return 'Home'
        elif code == 2:
            return 'Rehab'
        elif code == 3:
            return 'Residential Care'
        elif code == 4:
            return 'Other hospital'
        elif code == 5:
            return 'Died'
        elif code == 6:
            return 'Other'
        elif code == 7:
            return 'Unknown'
        else:
            return 'Unknown'
    
    # Apply mappings
    df['fresidence2_mapped'] = df['fresidence2'].apply(map_residence)
    df['wdest_mapped'] = df['wdest'].apply(map_wdest)
    
    # Create new residential care indicator
    df['new_residential_care'] = np.where(
        (df['fresidence2_mapped'] == 'Home') & 
        (df['wdest_mapped'] == 'Residential Care'),
        'Yes', 'No'
    )
    
    # Map other important categorical variables
    def map_sex(code):
        if code == 1:
            return 'Male'
        elif code == 2:
            return 'Female'
        else:
            return 'Unknown'
    
    def map_cogstat(code):
        if code == 1:
            return 'Normal'
        elif code == 2:
            return 'Impaired'
        elif code == 3:
            return 'Unknown'
        else:
            return 'Unknown'
    
    def map_mobil2(code):
        if code == 1:
            return 'Day 1'
        elif code == 2:
            return 'Day 2'
        elif code == 3:
            return 'Day 3+'
        elif code == 4:
            return 'Not mobilised'
        elif code == 5:
            return 'Unknown'
        else:
            return 'Unknown'
    
    def map_painmanage(code):
        if code == 1:
            return 'Yes'
        elif code == 2:
            return 'No'
        elif code == 3:
            return 'Unknown'
        else:
            return 'Unknown'
    
    def map_delay(code):
        # delay codes: 1=No delay, 2=Medical, 3=Operative, 4=Other
        if code == 1:
            return 'No delay'
        elif code == 2:
            return 'Medical'
        elif code == 3:
            return 'Operative'
        elif code == 4:
            return 'Other'
        else:
            return 'Unknown'
    
    def map_anaesth(code):
        # anaesth codes: 1=General, 2=Spinal, 3=Regional, 4=Other
        if code == 1:
            return 'General'
        elif code == 2:
            return 'Spinal'
        elif code == 3:
            return 'Regional'
        elif code == 4:
            return 'Other'
        else:
            return 'Unknown'
    
    def map_frailty(code):
        # frailty codes: 1=Fit, 2=Vulnerable, 3=Frail, 4=Severely Frail
        if code == 1:
            return 'Fit'
        elif code == 2:
            return 'Vulnerable'
        elif code == 3:
            return 'Frail'
        elif code == 4:
            return 'Severely Frail'
        else:
            return 'Unknown'
    
    def map_asa(code):
        # asa codes: 1=I, 2=II, 3=III, 4=IV, 5=V
        if code == 1:
            return 'I'
        elif code == 2:
            return 'II'
        elif code == 3:
            return 'III'
        elif code == 4:
            return 'IV'
        elif code == 5:
            return 'V'
        else:
            return 'Unknown'
    
    # Apply all mappings
    df['sex_mapped'] = df['sex'].apply(map_sex)
    df['cogstat_mapped'] = df['cogstat'].apply(map_cogstat)
    df['mobil2_mapped'] = df['mobil2'].apply(map_mobil2)
    df['painmanage_mapped'] = df['painmanage'].apply(map_painmanage)
    df['delay_mapped'] = df['delay'].apply(map_delay)
    df['anaesth_mapped'] = df['anaesth'].apply(map_anaesth)
    df['frailty_mapped'] = df['frailty'].apply(map_frailty)
    df['asa_mapped'] = df['asa'].apply(map_asa)
    
    # Convert mortality indicators
    df['mort30d_bool'] = df['mort30d'] == 1
    
    time_columns = ['ed_time', 'ward_time', 'time_to_surgery', 'gdate_time', 'discharge_time']
    df_cleaned_fixed = df.copy()
    for col in time_columns:
        # Compute Q1, Q3, IQR
        Q1 = df_cleaned_fixed[col].quantile(0.25)
        Q3 = df_cleaned_fixed[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = 0  # negative time is invalid → enforce 0 as min
        upper_bound = Q3 + 1.5 * IQR
        
        # Valid mask: within [0, upper_bound]
        valid_mask = (df_cleaned_fixed[col] >= lower_bound) & (df_cleaned_fixed[col] <= upper_bound)
        
        # Mean of valid data
        mean_valid = df_cleaned_fixed.loc[valid_mask, col].mean()
        
        # Count invalid values
        n_invalid = (~valid_mask).sum()
        
        # Replace invalid values with mean of valid ones
        df_cleaned_fixed[col] = np.where(valid_mask, df_cleaned_fixed[col], mean_valid)


    df = df_cleaned_fixed[df_cleaned_fixed['year'] >= 2022] 

    # Convert to hours/days for easier interpretation
    df['time_to_surgery_hours'] = df['time_to_surgery'] * 24
    df['ed_time_hours'] = df['ed_time'] * 24
    df['acute_los_days'] = df['ward_time']

    return df

def main():
    # Header
    st.markdown('<h1 class="main-header" >🦴 ANZHFR Hip Fracture Care Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
    <strong>Welcome</strong> - This dashboard provides information about hip fracture care in Australia and New Zealand. 
    If you or a loved one has experienced a hip fracture, this information can help you understand what to expect during the care journey.
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    dictionary = {}

    # Sidebar for filters
    with st.sidebar:
        st.markdown("## 🔍 Filter View")
        st.markdown("Adjust these filters to see specific information:")
        
        age_range = st.slider(
            "Age Range",
            min_value=int(df['age'].min()),
            max_value=int(df['age'].max()),
            value=(50, 120)
        )
        
        sex_filter = st.multiselect(
            "Gender",
            options=df['sex_mapped'].unique(),
            default=df['sex_mapped'].unique()
        )
        
        cog_filter = st.multiselect(
            "Cognitive Status",
            options=df['cogstat_mapped'].unique(),
            default=df['cogstat_mapped'].unique()
        )
        
        # Hospital filter
        hospital_options = ['All Hospitals'] + sorted(df['ahos_code'].dropna().unique().tolist())
        hospital_filter = st.selectbox(
            "Hospital Group",
            options=hospital_options
        )
        
        # Filter data
        filtered_df = df[
            (df['age'] >= age_range[0]) & 
            (df['age'] <= age_range[1]) &
            (df['sex_mapped'].isin(sex_filter)) &
            (df['cogstat_mapped'].isin(cog_filter))
        ]
        
        if hospital_filter != 'All Hospitals':
            filtered_df = filtered_df[filtered_df['ahos_code'] == hospital_filter]
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Information")
        st.metric("Patients in View", f"{len(filtered_df):,}")
        st.metric("Median Age", f"{filtered_df['age'].median():.0f} years")
        st.metric("Data Completeness", f"{(filtered_df['fracture_time'].notna().sum() / len(filtered_df) * 100):.0f}%")
    
    # Main content - Tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", 
        "🏥 Care Journey", 
        "📊 Outcomes", 
        "ℹ️ Information"
    ])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="sub-header">', unsafe_allow_html=True)
            st.metric(
                "Median Age",
                f"{filtered_df['age'].median():.0f} years",
                help="Average age of patients with hip fractures"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            dictionary['Median Age'] = filtered_df['age'].median()

            # Age distribution
            # dictionary['Age Distribution'] = filtered_df['age'].describe().to_dict()
            fig_age = px.histogram(
                filtered_df, 
                x='age',
                nbins=20,
                title="Age Distribution",
                labels={'age': 'Age (years)'},
                color_discrete_sequence=['#3498db']
            )
            fig_age.update_layout(showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.markdown('<div class="sub-header">', unsafe_allow_html=True)
            female_pct = (filtered_df['sex_mapped'] == 'Female').mean() * 100
            st.metric(
                "Female Patients",
                f"{female_pct:.0f}%",
                help="Percentage of female patients"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            dictionary['Percentage of female patients'] = (filtered_df['sex_mapped'] == 'Female').mean() * 100
            
            # Gender distribution

            gender_counts = filtered_df['sex_mapped'].value_counts()
            dictionary['gender distribution'] = filtered_df['sex_mapped'].value_counts()
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Gender Distribution",
                color_discrete_sequence=['#e74c3c', '#3498db', '#95a5a6']
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col3:
            st.markdown('<div class="sub-header">', unsafe_allow_html=True)
            cog_impairment = (filtered_df['cogstat_mapped'] == 'Impaired').mean() * 100
            st.metric(
                "Cognitive Impairment",
                f"{cog_impairment:.0f}%",
                help="Patients with cognitive impairment"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            dictionary['Percentage with cognitive impairment'] = (filtered_df['cogstat_mapped'] == 'Impaired').mean() * 100

            # Cognitive status
            cog_counts = filtered_df['cogstat_mapped'].value_counts()
            dictionary['Cognitive status distribution'] = filtered_df['cogstat_mapped'].value_counts()
            fig_cog = px.bar(
                x=cog_counts.index,
                y=cog_counts.values,
                title="Cognitive Status",
                labels={'x': 'Status', 'y': 'Count'},
                color_discrete_sequence=['#2ecc71', '#e74c3c', '#95a5a6']
            )
            st.plotly_chart(fig_cog, use_container_width=True)
        
        # Frailty and ASA Score
        st.markdown('<h3 class="sub-header">Patient Health Status</h3>', unsafe_allow_html=True)
        
        col4, col5 = st.columns(2)
        
        with col4:
            analges = filtered_df['analges'].value_counts()
            map1 = {1: 'Nerve block administered before arriving in OT', 2: 'Nerve block administered in OT', 3: 'Both',4: 'Neither'}
            analges = analges.rename(index=map1)
            dictionary['Nerve Block Status'] = analges
            fig_analges = px.pie(
                values=analges.values,
                names=analges.index,
                title="Nerve Block Status",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig_analges, use_container_width=True)
        
        with col5:
            delassess = filtered_df['delassess'].value_counts()
            map2 = {1: 'Not assessed;', 2: 'Assessed and not identified', 3: 'Assessed and identified'}
            delassess = delassess.rename(index=map2)
            dictionary['Delirium Assessment Status'] = delassess
            fig_delass = px.bar(
                x=delassess.index,
                y=delassess.values,
                title="Post-operative Delirium Assessment",
                labels={'x': 'Delirium Assesment', 'y': 'Count'},
                color_discrete_sequence=['#9b59b6']
            )
            st.plotly_chart(fig_delass, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">🏥 Hip Fracture Care Journey</h2>', unsafe_allow_html=True)
        
        # Care journey steps - using actual data
        
        steps = [
            {
                "step": 1,
                "title": "Emergency Department",
                "metric": f"{filtered_df['ed_time_hours'].mean():.1f} hours",
                "description": "Mean time spent in ED",
                "color": "#e74c3c"
            },
            {
                "step": 2,
                "title": "Time to Surgery",
                "metric": f"{filtered_df['time_to_surgery_hours'].median():.1f} hours",
                "description": "Median time from fracture to surgery",
                "color": "#3498db"
            },
            {
                "step": 3,
                "title": "Pain Management",
                "metric": f"{(filtered_df['painmanage_mapped'] == 'Yes').mean()*100:.0f}%",
                "description": "Patients receiving pain management",
                "color": "#2ecc71"
            },
            {
                "step": 4,
                "title": "Early Mobilisation",
                "metric": f"{(filtered_df['mobil2_mapped'] == 'Day 1').mean()*100:.0f}%",
                "description": "Mobilised on day 1 after surgery",
                "color": "#f39c12"
            }
        ]
        dictionary['Care Journey Steps'] = steps

        
        # Display steps
        cols = st.columns(4)
        for i, step in enumerate(steps):
            with cols[i]:
                st.markdown(f"""
                <div style="background-color: {step['color']}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="margin: 0;">Step {step['step']}</h3>
                    <h2 style="margin: 10px 0;">{step['metric']}</h2>
                    <h4 style="margin: 5px 0;">{step['title']}</h4>
                    <p style="font-size: 0.9em;">{step['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed metrics
        st.markdown('<h3 class="sub-header">Detailed Care Metrics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Time to surgery distribution (filter out extreme values for better visualization)
            surgery_data = filtered_df['time_to_surgery_hours'].clip(upper=120)  # Cap at 120 hours for better visualization
            dictionary['Time to Surgery Distribution'] = surgery_data.describe().to_dict()
            fig_surgery_time = px.histogram(
                filtered_df,
                x=surgery_data,
                nbins=30,
                title="Time to Surgery Distribution",
                labels={'x': 'Hours'},
                color_discrete_sequence=['#3498db']
            )
            median_time = filtered_df['time_to_surgery_hours'].median()
            dictionary['Median Time to Surgery (hours)'] = median_time
            fig_surgery_time.add_vline(
                x=median_time,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: {median_time:.1f}h"
            )
            dictionary['Target Time to Surgery (hours)'] = 48
            fig_surgery_time.add_vline(
                x=48,
                line_dash="dot",
                line_color="green",
                annotation_text="48h Target",
                annotation_position="top right"
            )
            fig_surgery_time.add_annotation(
                x=0,  # Position: around 12 hours
                y=0.3,  # Relative y-position (use 'paper' coordinates)
                yref='paper',
                text="<b>0 - Within a day</b>",
                showarrow=False,
                font=dict(color="black", size=12),
                bgcolor="lightyellow",
                bordercolor="gray"
            )
            st.plotly_chart(fig_surgery_time, use_container_width=True)
        
        with col2:
            # Delay reasons
            delay_counts = filtered_df['delay_mapped'].value_counts()
            dictionary['Surgery Delay Reasons'] = filtered_df['delay_mapped'].value_counts()
            fig_delay = px.pie(
                values=delay_counts.values,
                names=delay_counts.index,
                title="Reasons for Surgery Delay",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_delay, use_container_width=True)
        
        with col3:
            # Anaesthesia types
            anaesth_counts = filtered_df['anaesth_mapped'].value_counts()
            dictionary['Anaesthesia Types'] = filtered_df['anaesth_mapped'].value_counts()
            fig_anaesth = px.bar(
                x=anaesth_counts.index,
                y=anaesth_counts.values,
                title="Type of Anaesthesia",
                labels={'x': 'Type', 'y': 'Count'},
                color_discrete_sequence=['#9b59b6']
            )
            st.plotly_chart(fig_anaesth, use_container_width=True)
        
        # Mobilisation timeline
        st.markdown('<h3 class="sub-header">Mobilisation After Surgery</h3>', unsafe_allow_html=True)
        mobil_order = {1:'Patient out of bed ', 2: 'Patient not given opportunity '}
        mobil_counts = filtered_df['mobil'].value_counts()
        mobil_counts.index = mobil_counts.index.map(mobil_order)
        
        fig_mobil = px.bar(
            x=mobil_counts.index,
            y=mobil_counts.values,
            title="When Patients First Mobilise After Surgery",
            labels={'x': 'Time to Mobilisation', 'y': 'Number of Patients'},
            color=mobil_counts.index,
            color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
        )
        st.plotly_chart(fig_mobil, use_container_width=True)
        
        # ED time distribution
        st.markdown('<h3 class="sub-header">Emergency Department Stay</h3>', unsafe_allow_html=True)
        # Mapping for pain assessment categories
        p = {
            1: 'Within 30 minutes of ED presentation',
            2: 'Greater than 30 minutes of ED presentation',
            3: 'Pain assessment not done'
        }

        # Get value counts and map to readable labels
        pain = filtered_df['painassess'].value_counts()
        pain.index = pain.index.map(p)
        dictionary['Pain Assessment Timing'] = pain
        # Create bar chart
        fig_pain = px.bar(
            x=pain.index,
            y=pain.values,
            title="Pain Assessment in ED",
            labels={'x': 'Pain Assessment Timing', 'y': 'Number of Patients'},
            color=pain.index,  # color by category
            color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c'],  # green → orange → red
            text=pain.values  # show counts on bars
        )
        # Improve layout
        fig_pain.update_layout(
            xaxis_tickangle=0,  # keep labels horizontal
            showlegend=False,   # not needed when x-axis shows category
            height=500
        )
        # Optional: wrap long x-labels if needed (though these are okay)
        fig_pain.update_xaxes(tickmode='array', tickvals=pain.index, ticktext=[label.replace(' ', '<br>') for label in pain.index])

        st.plotly_chart(fig_pain, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">📊 Patient Outcomes</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="sub-header">', unsafe_allow_html=True)
            los_median = filtered_df['acute_los_days'].median()
            st.metric(
                "Acute Hospital Stay",
                f"{los_median:.0f} days",
                "median length of stay",
                delta_color="off"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            dictionary['Median Acute Hospital Stay (days)'] = los_median
            # Length of stay distribution
            los_data = filtered_df['acute_los_days'].clip(upper=30)  # Cap at 30 days for visualization
            dictionary['Acute Hospital Stay Distribution'] = los_data.describe().to_dict()
            fig_los = px.histogram(
                filtered_df,
                x=los_data,
                nbins=20,
                title="Acute Hospital Stay Duration",
                labels={'x': 'Days'},
                color_discrete_sequence=['#3498db']
            )
            fig_los.add_vline(
                x=los_median,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: {los_median:.0f} days"
            )
            st.plotly_chart(fig_los, use_container_width=True)
        
        with col2:
            st.markdown('<div class="sub-header">', unsafe_allow_html=True)
            rehab_pct = (filtered_df['wdest_mapped'] == 'Rehab').mean() * 100
            dictionary['Percentage Discharged to Rehabilitation'] = rehab_pct
            st.metric(
                "Go to Rehabilitation",
                f"{rehab_pct:.0f}%",
                "of patients",
                delta_color="off"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Discharge destination
            dest_counts = filtered_df['wdest_mapped'].value_counts()
            fig_dest = px.pie(
                values=dest_counts.values,
                names=dest_counts.index,
                title="Discharge Destination",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_dest, use_container_width=True)
        
        with col3:
            st.markdown('<div class="sub-header">', unsafe_allow_html=True)
            new_care_pct = (filtered_df['new_residential_care'] == 'Yes').mean() * 100
            dictionary['Percentage Needing New Residential Care'] = new_care_pct
            st.metric(
                "New Residential Care",
                f"{new_care_pct:.1f}%",
                "need new placement",
                delta_color="off"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        #-----------------------------------
        # dictionary.to_json("dashboard_metrics.json")
        # import json

        # # Save
        # with open('data.json', 'w') as f:
        #     json.dump(dictionary, f, indent=4)
        print(dictionary)
        #-----------------------------------

        @st.cache_resource
        def load_model():
            model = CatBoostClassifier()
            model.load_model("catboost_delirium_model.cbm")
            return model

        model = load_model()

        # Define feature names IN THE SAME ORDER as during training
        FEATURES = [
            'age', 'sex', 'ptype', 'uresidence', 'e_dadmit', 'painassess',
            'painmanage', 'ward', 'walk', 'cogassess', 'cogstat', 'addelassess',
            'bonemed', 'passess', 'side', 'afracture', 'ftype', 'asa', 'frailty',
            'delay', 'anaesth', 'analges', 'consult', 'wbear', 'mobil', 'pulcers',
            'fassess', 'dbonemed1', 'malnutrition', 'mobil2', 'fracture_time',
            'ed_time', 'ward_time', 'time_to_surgery', 'gdate_time',
            'discharge_time', 'surgery_delay_group' 
        ]

        # Categorical features (must match cat_features_eng)
        CATEGORICAL_FEATURES = ['sex', 'ptype', 'uresidence', 'e_dadmit', 'painassess', 'painmanage', 'ward', 'walk', 'cogassess', 'cogstat', 'addelassess', 'bonemed', 'passess', 'side', 'afracture', 'ftype', 'asa', 'frailty', 'delay', 'anaesth', 'analges', 'consult', 'wbear', 'mobil', 'pulcers', 'fassess', 'dbonemed1', 'malnutrition', 'mobil2', 'surgery_delay_group']

        # ==============================
        # 2. STREAMLIT UI
        # ==============================
        st.set_page_config(page_title="Delirium Risk Predictor", layout="wide")
        st.title("🩺 Post-Operative Delirium Risk Prediction")
        st.markdown("Enter patient details to predict risk of post-operative delirium.")

        # Create input form
        with st.form("prediction_form"):
            st.subheader("Patient Features")
            
            # Create multiple columns for better organization
            col1, col2, col3 = st.columns(3)
            
            # with col1:
            #     age = st.number_input("Age", min_value=50, max_value=100, value=80)
            #     sex = st.selectbox("Sex", options=["M", "F"], index=0)
            #     ptype = st.selectbox("Patient Type", options=["1", "2", "3"], index=0)  # Assuming codes 1-3
            #     uresidence = st.selectbox("Residence", options=["1", "2", "3"], index=0)
            #     e_dadmit = st.selectbox("Emergency Department Admit", options=["1", "2", "3"], index=0)
            #     painassess = st.selectbox("Pain Assessment", options=["1", "2", "3"], index=0)
            #     painmanage = st.selectbox("Pain Management", options=["1", "2", "3"], index=0)
            #     ward = st.text_input("Ward", value="A3")
            #     walk = st.selectbox("Walking Ability", options=["1", "2", "3"], index=0)
                
            # with col2:
            #     cogassess = st.selectbox("Cognitive Assessment", options=["1", "2", "3", "4"], index=1)
            #     cogstat = st.selectbox("Cognitive Status", options=["1", "2", "3"], index=0)
            #     addelassess = st.selectbox("Additional Delirium Assessment", options=["1", "2", "3"], index=0)
            #     bonemed = st.selectbox("Bone Medication", options=["1", "2"], index=0)
            #     passess = st.selectbox("Pressure Assessment", options=["1", "2"], index=0)
            #     side = st.selectbox("Fracture Side", options=["1", "2"], index=0)  # Left/Right
            #     afracture = st.selectbox("Additional Fracture", options=["1", "2"], index=0)
            #     ftype = st.selectbox("Fracture Type", options=["1", "2", "3"], index=0)
            #     asa = st.selectbox("ASA Score", options=["1", "2", "3", "4"], index=2)
                
            # with col3:
            #     frailty = st.selectbox("Frailty", options=["1", "2", "3", "4", "5"], index=2)
            #     delay = st.selectbox("Surgery Delay", options=["1", "2"], index=0)
            #     anaesth = st.selectbox("Anaesthesia Type", options=["1", "2", "3"], index=0)
            #     analges = st.selectbox("Analgesia", options=["1", "2", "3", "4"], index=0)
            #     consult = st.selectbox("Consultation", options=["1", "2"], index=0)
            #     wbear = st.selectbox("Weight Bearing", options=["1", "2"], index=0)
            #     mobil = st.selectbox("Mobilisation", options=["1", "2"], index=0)
            #     pulcers = st.selectbox("Pressure Ulcers", options=["1", "2"], index=0)
            #     fassess = st.selectbox("Falls Assessment", options=["1", "2"], index=0)
            with col1:
                age = st.number_input("Age", min_value=50, max_value=100, value=80)
            
                sex_label = st.selectbox("Sex", ["Male", "Female"])
                sex = {"Male": "1", "Female": "2"}[sex_label]
            
                ptype_label = st.selectbox(
                    "Patient Type",
                    ["Public patient", "Private patient", "Overseas patient"]
                )
                ptype = {
                    "Public patient": "1",
                    "Private patient": "2",
                    "Overseas patient": "3"
                }[ptype_label]
            
                uresidence_label = st.selectbox(
                    "Usual Residence",
                    [
                        "Private residence",
                        "Residential aged care facility",
                        "Other / Unknown"
                    ]
                )
                uresidence = {
                    "Private residence": "1",
                    "Residential aged care facility": "2",
                    "Other / Unknown": "3"
                }[uresidence_label]
            
                e_dadmit_label = st.selectbox(
                    "Emergency Department Admit",
                    ["No", "Yes", "Unknown"]
                )
                e_dadmit = {
                    "No": "1",
                    "Yes": "2",
                    "Unknown": "3"
                }[e_dadmit_label]
            
                painassess_label = st.selectbox(
                    "Pain Assessment",
                    [
                        "Not assessed",
                        "Assessed and documented",
                        "Assessed but not documented"
                    ]
                )
                painassess = {
                    "Not assessed": "1",
                    "Assessed and documented": "2",
                    "Assessed but not documented": "3"
                }[painassess_label]
            
                painmanage_label = st.selectbox(
                    "Pain Management",
                    [
                        "No analgesia",
                        "Oral analgesia",
                        "Parenteral analgesia"
                    ]
                )
                painmanage = {
                    "No analgesia": "1",
                    "Oral analgesia": "2",
                    "Parenteral analgesia": "3"
                }[painmanage_label]
            
                ward = st.text_input("Ward", value="A3")
            
                walk_label = st.selectbox(
                    "Walking Ability",
                    [
                        "Independent",
                        "With aid",
                        "With assistance",
                        "Unable to walk"
                    ]
                )
                walk = {
                    "Independent": "1",
                    "With aid": "2",
                    "With assistance": "3",
                    "Unable to walk": "4"
                }[walk_label]
            
            
            with col2:
                cogassess_label = st.selectbox(
                    "Cognitive Assessment",
                    [
                        "Not assessed",
                        "Assessed and normal",
                        "Assessed and abnormal",
                        "Unknown"
                    ]
                )
                cogassess = {
                    "Not assessed": "1",
                    "Assessed and normal": "2",
                    "Assessed and abnormal": "3",
                    "Unknown": "4"
                }[cogassess_label]
            
                cogstat_label = st.selectbox(
                    "Cognitive Status",
                    [
                        "Normal cognition",
                        "Impaired cognition",
                        "Unknown"
                    ]
                )
                cogstat = {
                    "Normal cognition": "1",
                    "Impaired cognition": "2",
                    "Unknown": "3"
                }[cogstat_label]
            
                addelassess_label = st.selectbox(
                    "Additional Delirium Assessment",
                    [
                        "Not assessed",
                        "Assessed – not identified",
                        "Assessed – identified"
                    ]
                )
                addelassess = {
                    "Not assessed": "1",
                    "Assessed – not identified": "2",
                    "Assessed – identified": "3"
                }[addelassess_label]
            
                bonemed_label = st.selectbox("Bone Medication", ["No", "Yes"])
                bonemed = {"No": "1", "Yes": "2"}[bonemed_label]
            
                passess_label = st.selectbox(
                    "Pressure Assessment",
                    [
                        "Not assessed",
                        "Low risk",
                        "Moderate risk",
                        "High risk",
                        "Very high risk"
                    ]
                )
                passess = {
                    "Not assessed": "1",
                    "Low risk": "2",
                    "Moderate risk": "3",
                    "High risk": "4",
                    "Very high risk": "5"
                }[passess_label]
            
                side_label = st.selectbox("Fracture Side", ["Left", "Right"])
                side = {"Left": "1", "Right": "2"}[side_label]
            
                afracture_label = st.selectbox(
                    "Additional Fracture",
                    ["No", "Yes", "Unknown"]
                )
                afracture = {
                    "No": "1",
                    "Yes": "2",
                    "Unknown": "3"
                }[afracture_label]
            
                ftype_label = st.selectbox(
                    "Fracture Type",
                    [
                        "Intracapsular",
                        "Intertrochanteric",
                        "Subtrochanteric",
                        "Other"
                    ]
                )
                ftype = {
                    "Intracapsular": "1",
                    "Intertrochanteric": "2",
                    "Subtrochanteric": "3",
                    "Other": "4"
                }[ftype_label]
            
                asa_label = st.selectbox(
                    "ASA Score",
                    [
                        "ASA I – Healthy",
                        "ASA II – Mild systemic disease",
                        "ASA III – Severe systemic disease",
                        "ASA IV – Severe disease, constant threat to life"
                    ]
                )
                asa = {
                    "ASA I – Healthy": "1",
                    "ASA II – Mild systemic disease": "2",
                    "ASA III – Severe systemic disease": "3",
                    "ASA IV – Severe disease, constant threat to life": "4"
                }[asa_label]
            
            
            with col3:
                frailty_label = st.selectbox(
                    "Frailty",
                    [
                        "Not frail",
                        "Mild frailty",
                        "Moderate frailty",
                        "Severe frailty",
                        "Very severe frailty"
                    ]
                )
                frailty = {
                    "Not frail": "1",
                    "Mild frailty": "2",
                    "Moderate frailty": "3",
                    "Severe frailty": "4",
                    "Very severe frailty": "5"
                }[frailty_label]
            
                delay_label = st.selectbox("Surgery Delay", ["No", "Yes"])
                delay = {"No": "1", "Yes": "2"}[delay_label]
            
                anaesth_label = st.selectbox(
                    "Anaesthesia Type",
                    [
                        "General",
                        "Spinal / Regional",
                        "Combined / Other"
                    ]
                )
                anaesth = {
                    "General": "1",
                    "Spinal / Regional": "2",
                    "Combined / Other": "3"
                }[anaesth_label]
            
                analges_label = st.selectbox(
                    "Analgesia",
                    [
                        "Nerve block before OT",
                        "Nerve block in OT",
                        "Both",
                        "Neither"
                    ]
                )
                analges = {
                    "Nerve block before OT": "1",
                    "Nerve block in OT": "2",
                    "Both": "3",
                    "Neither": "4"
                }[analges_label]
            
                consult_label = st.selectbox("Consultation", ["No", "Yes"])
                consult = {"No": "1", "Yes": "2"}[consult_label]
            
                wbear_label = st.selectbox(
                    "Weight Bearing",
                    [
                        "Unrestricted",
                        "Restricted / Non-weight bearing"
                    ]
                )
                wbear = {
                    "Unrestricted": "1",
                    "Restricted / Non-weight bearing": "2"
                }[wbear_label]
            
                mobil_label = st.selectbox("Mobilisation", ["No", "Yes"])
                mobil = {"No": "1", "Yes": "2"}[mobil_label]
            
                pulcers_label = st.selectbox("Pressure Ulcers", ["No", "Yes"])
                pulcers = {"No": "1", "Yes": "2"}[pulcers_label]
            
                fassess_label = st.selectbox(
                    "Falls Assessment",
                    [
                        "Not assessed",
                        "Low risk",
                        "Moderate risk",
                        "High risk",
                        "Very high risk"
                    ]
                )
                fassess = {
                    "Not assessed": "1",
                    "Low risk": "2",
                    "Moderate risk": "3",
                    "High risk": "4",
                    "Very high risk": "5"
                }[fassess_label]
                
            # Second row of columns for remaining features
            st.subheader("Additional Features")
            col4, col5, col6 = st.columns(3)
            
            with col4:
                dbonemed1 = st.selectbox("Discharge Bone Medication", options=["1", "2"], index=0)
                malnutrition = st.selectbox("Malnutrition", options=["1", "2"], index=0)
                mobil2 = st.selectbox("Secondary Mobilisation", options=["1", "2"], index=0)
                
            with col5:
                fracture_time = st.number_input("Fracture Time (hours)", min_value=0, max_value=500, value=24)
                ed_time = st.number_input("ED Time (hours)", min_value=0, max_value=500, value=12)
                ward_time = st.number_input("Ward Time (hours)", min_value=0, max_value=500, value=36)
                time_to_surgery = st.number_input("Time to Surgery (hours)", min_value=0, max_value=200, value=24)
                
            with col6:
                gdate_time = st.number_input("G-Date Time (hours)", min_value=0, max_value=500, value=48)
                discharge_time = st.number_input("Discharge Time (hours)", min_value=0, max_value=1000, value=120)
                surgery_delay_group = st.selectbox(
                    "Surgery Delay Group",
                    options=["<24h", "24-48h", "2-5d", ">5d"],
                    index=1
                )
            
            submitted = st.form_submit_button("Predict Delirium Risk")

        # ==============================
        # 3. MAKE PREDICTION
        # ==============================
        if submitted:
            # Build input DataFrame with ALL features in the correct order
            input_data = pd.DataFrame([{
                'age': age,
                'sex': sex,
                'ptype': ptype,
                'uresidence': uresidence,
                'e_dadmit': e_dadmit,
                'painassess': painassess,
                'painmanage': painmanage,
                'ward': ward,
                'walk': walk,
                'cogassess': cogassess,
                'cogstat': cogstat,
                'addelassess': addelassess,
                'bonemed': bonemed,
                'passess': passess,
                'side': side,
                'afracture': afracture,
                'ftype': ftype,
                'asa': asa,
                'frailty': frailty,
                'delay': delay,
                'anaesth': anaesth,
                'analges': analges,
                'consult': consult,
                'wbear': wbear,
                'mobil': mobil,
                'pulcers': pulcers,
                'fassess': fassess,
                'dbonemed1': dbonemed1,
                'malnutrition': malnutrition,
                'mobil2': mobil2,
                'fracture_time': fracture_time,
                'ed_time': ed_time,
                'ward_time': ward_time,
                'time_to_surgery': time_to_surgery,
                'gdate_time': gdate_time,
                'discharge_time': discharge_time,
                'surgery_delay_group': surgery_delay_group
            }])
            
            # Ensure column order matches training
            input_data = input_data[FEATURES]
            
            # Convert categoricals to string (as done during training)
            for col in CATEGORICAL_FEATURES:
                if col in input_data.columns:
                    input_data[col] = input_data[col].astype(str)
            
            try:
                # Predict
                pred_proba = model.predict_proba(input_data)[0][1]
                pred_class = "Delirium Likely" if pred_proba > 0.5 else "Delirium Unlikely"
                
                # Display result
                st.subheader(" Prediction Result")
                st.metric("Delirium Probability", f"{pred_proba:.1%}")
                st.write(f"**Classification**: {pred_class}")
                
                # Risk interpretation
                if pred_proba >= 0.7:
                    st.error("🚨 High Risk: Consider delirium prevention protocols.")
                elif pred_proba >= 0.4:
                    st.warning("⚠️ Moderate Risk: Monitor closely.")
                else:
                    st.success("✅ Low Risk")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.write("Check that all features match the training schema.")

        # ==============================
        # 4. HELP / INFO
        # ==============================
        with st.expander("ℹ️ About this model"):
            st.write("""
            - **Model**: Optuna-tuned CatBoost classifier
            - **Target**: Post-operative delirium (binary: assessed & identified vs. not)
            - **AUC**: ~0.81 on held-out test set
            - **Use Case**: Clinical decision support for identifying high-risk patients
            - **Accuracy**    : 0.7529
            - **AUC**         : 0.8074
            - **Recall**      : 0.7305
            """)
    
    with tab4:
        st.markdown('<h2 class="sub-header">ℹ️ About Hip Fracture Care</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### What is a Hip Fracture?
            
            A hip fracture is a break in the upper part of the thigh bone (femur). 
            It's a serious injury that often requires surgery and comprehensive rehabilitation.
            
            ### The Care Journey
            
            1. **Emergency Care**: Prompt assessment and pain management
            2. **Surgery**: Usually within 48 hours for best outcomes
            3. **Hospital Recovery**: Multidisciplinary care including physiotherapy
            4. **Rehabilitation**: To regain mobility and independence
            5. **Return Home or Ongoing Care**: Support for long-term recovery
            
            ### Why Timely Care Matters
            
            - Early surgery reduces complications
            - Good pain management improves recovery
            - Early mobilisation prevents further health issues
            - Comprehensive care improves quality of life
            
            ### Understanding the Data
            
            This dashboard shows aggregated, anonymous data from the ANZHFR registry.
            Individual experiences may vary based on personal health circumstances.
            
            ### Data Notes
            
            - **Time calculations**: Based on available date/time data
            - **Missing values**: Some time calculations may have missing data
            - **Categories**: Numeric codes mapped to meaningful labels
            - **Sample size**: Based on {:,} patient records
            """.format(len(df)))
        
        with col2:
            st.markdown("""
            ### 📞 Support Resources
            
            **Emergency**
            - Call 000 for emergency assistance
            
            **Information & Support**
            - HealthDirect: 1800 022 222
            - My Aged Care: 1800 200 422
            
            **Clinical Guidelines**
            - [ANZHFR Clinical Care Standard](https://www.safetyandquality.gov.au/standards/clinical-care-standards/hip-fracture-care-clinical-care-standard)
            
            ### 🔍 Key Statistics Explained
            
            **Time to Surgery**
            - Target: Within 48 hours
            - Earlier surgery leads to better outcomes
            
            **Mobilisation**
            - Day 1 mobilisation is ideal
            - Helps prevent complications
            
            **Rehabilitation**
            - Most patients need rehab
            - Crucial for recovery
            
            **New Residential Care**
            - Percentage needing new aged care placement
            - Important for discharge planning
            """)
            
            st.markdown("---")
            st.markdown("""
            **Last Updated**: {}
            
            **Data Source**: ANZHFR Registry
            
            **Note**: This information is for educational purposes.
            Always consult healthcare professionals for medical advice.
            """.format(datetime.now().strftime("%B %d, %Y")))
        
        # Quick stats summary
        st.markdown("---")
        st.markdown('<h3 class="sub-header">Key Statistics at a Glance</h3>', unsafe_allow_html=True)
        
        summary_cols = st.columns(5)
        summary_metrics = [
            ("Median Age", f"{filtered_df['age'].median():.0f} years"),
            ("Female Patients", f"{(filtered_df['sex_mapped'] == 'Female').mean()*100:.0f}%"),
            ("Time to Surgery", f"{filtered_df['time_to_surgery_hours'].median():.0f} hours"),
            ("Hospital Stay", f"{filtered_df['acute_los_days'].median():.0f} days"),
            ("Rehabilitation", f"{(filtered_df['wdest_mapped'] == 'Rehab').mean()*100:.0f}%"),
        ]
        
        for i, (title, value) in enumerate(summary_metrics):
            with summary_cols[i]:
                st.markdown(f"""
                <div class="sub-header" style="text-align: center;">
                    <h4 style="color: #7f8c8d; margin: 0;">{title}</h4>
                    <h2 style="color: #2c3e50; margin: 10px 0;">{value}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Data dictionary
        with st.expander("📋 Data Field Information"):
            st.markdown("""
            | Field | Description |
            |-------|-------------|
            | age | Patient age in years |
            | sex_mapped | Gender (Male/Female/Unknown) |
            | cogstat_mapped | Cognitive status (Normal/Impaired/Unknown) |
            | time_to_surgery_hours | Hours from fracture to surgery |
            | ed_time_hours | Hours spent in emergency department |
            | acute_los_days | Days in acute hospital care |
            | painmanage_mapped | Whether pain was managed (Yes/No/Unknown) |
            | mobil2_mapped | When patient first mobilised after surgery |
            | wdest_mapped | Discharge destination |
            | fresidence2_mapped | Pre-fracture residence |
            | new_residential_care | Whether patient needed new residential care |
            | mort30d_bool | 30-day mortality indicator |
            | frailty_mapped | Clinical frailty score |
            | asa_mapped | ASA physical status score |
            | delay_mapped | Reason for surgery delay |
            | anaesth_mapped | Type of anaesthesia used |
            """)

if __name__ == "__main__":
    main()