import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt

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
    .age-distribution {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    # This function would load your actual data
    # For demonstration, I'll create sample data similar to your structure
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'age': np.random.normal(82, 10, n).clip(60, 100),
        'sex': np.random.choice(['Female', 'Male'], n, p=[0.7, 0.3]),
        'cogstat': np.random.choice(['Normal', 'Impaired', 'Unknown'], n, p=[0.5, 0.45, 0.05]),
        'time_to_surgery_hours': np.random.gamma(3, 8, n).clip(0, 120),
        'ed_time_hours': np.random.exponential(4, n).clip(0, 24),
        'acute_los_days': np.random.gamma(7, 2, n).clip(1, 30),
        'delay': np.random.choice(['No delay', 'Medical', 'Operative', 'Other'], n, p=[0.6, 0.25, 0.1, 0.05]),
        'anaesth': np.random.choice(['General', 'Spinal', 'Regional'], n, p=[0.4, 0.5, 0.1]),
        'painassess': np.random.choice(['Yes', 'No', 'Unknown'], n, p=[0.8, 0.15, 0.05]),
        'painmanage': np.random.choice(['Yes', 'No', 'Unknown'], n, p=[0.85, 0.1, 0.05]),
        'mobil2': np.random.choice(['Day 1', 'Day 2', 'Day 3+', 'Not mobilised'], n, p=[0.4, 0.3, 0.25, 0.05]),
        'wdest': np.random.choice(['Home', 'Rehab', 'Residential Care', 'Other hospital'], n, p=[0.2, 0.5, 0.25, 0.05]),
        'fresidence2': np.random.choice(['Home', 'Residential Care', 'Unknown'], n, p=[0.6, 0.35, 0.05]),
        'mort30d': np.random.choice([0, 1], n, p=[0.93, 0.07]),
        'frailty': np.random.choice(['Fit', 'Vulnerable', 'Frail', 'Severely Frail'], n, p=[0.2, 0.3, 0.4, 0.1]),
        'asa': np.random.choice(['I', 'II', 'III', 'IV', 'V'], n, p=[0.05, 0.25, 0.5, 0.15, 0.05]),
    })
    
    # Add some derived metrics
    data['new_residential_care'] = np.where(
        (data['fresidence2'] == 'Home') & (data['wdest'] == 'Residential Care'), 
        'Yes', 'No'
    )
    
    return data

def main():
    # Header
    st.markdown('<h1 class="main-header">🦴 ANZHFR Hip Fracture Care Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Welcome</strong> - This dashboard provides information about hip fracture care in Australia and New Zealand. 
    If you or a loved one has experienced a hip fracture, this information can help you understand what to expect during the care journey.
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar for filters
    with st.sidebar:
        st.markdown("## 🔍 Filter View")
        st.markdown("Adjust these filters to see specific information:")
        
        age_range = st.slider(
            "Age Range",
            min_value=int(df['age'].min()),
            max_value=int(df['age'].max()),
            value=(70, 95)
        )
        
        sex_filter = st.multiselect(
            "Gender",
            options=['Female', 'Male'],
            default=['Female', 'Male']
        )
        
        cog_filter = st.multiselect(
            "Cognitive Status",
            options=df['cogstat'].unique(),
            default=df['cogstat'].unique()
        )
        
        # Filter data
        filtered_df = df[
            (df['age'] >= age_range[0]) & 
            (df['age'] <= age_range[1]) &
            (df['sex'].isin(sex_filter)) &
            (df['cogstat'].isin(cog_filter))
        ]
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Information")
        st.metric("Patients in View", f"{len(filtered_df):,}")
        st.metric("Median Age", f"{filtered_df['age'].median():.0f} years")
    
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
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Median Age",
                f"{filtered_df['age'].median():.0f} years",
                help="Average age of patients with hip fractures"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Age distribution
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
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            female_pct = (filtered_df['sex'] == 'Female').mean() * 100
            st.metric(
                "Female Patients",
                f"{female_pct:.0f}%",
                help="Percentage of female patients"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Gender distribution
            gender_counts = filtered_df['sex'].value_counts()
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Gender Distribution",
                color_discrete_sequence=['#e74c3c', '#3498db']
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            cog_impairment = (filtered_df['cogstat'] == 'Impaired').mean() * 100
            st.metric(
                "Cognitive Impairment",
                f"{cog_impairment:.0f}%",
                help="Patients with cognitive impairment"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Cognitive status
            cog_counts = filtered_df['cogstat'].value_counts()
            fig_cog = px.bar(
                x=cog_counts.index,
                y=cog_counts.values,
                title="Cognitive Status",
                labels={'x': 'Status', 'y': 'Count'},
                color_discrete_sequence=['#2ecc71']
            )
            st.plotly_chart(fig_cog, use_container_width=True)
        
        # Frailty and ASA Score
        st.markdown('<h3 class="sub-header">Patient Health Status</h3>', unsafe_allow_html=True)
        
        col4, col5 = st.columns(2)
        
        with col4:
            frailty_counts = filtered_df['frailty'].value_counts()
            fig_frailty = px.pie(
                values=frailty_counts.values,
                names=frailty_counts.index,
                title="Frailty Status",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig_frailty, use_container_width=True)
        
        with col5:
            asa_counts = filtered_df['asa'].value_counts().sort_index()
            fig_asa = px.bar(
                x=asa_counts.index,
                y=asa_counts.values,
                title="ASA Physical Status (I-V)",
                labels={'x': 'ASA Score (I=Healthy, V=Moribund)', 'y': 'Count'},
                color_discrete_sequence=['#9b59b6']
            )
            st.plotly_chart(fig_asa, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">🏥 Hip Fracture Care Journey</h2>', unsafe_allow_html=True)
        
        # Care journey steps
        steps = [
            {
                "step": 1,
                "title": "Emergency Department",
                "metric": f"{filtered_df['ed_time_hours'].median():.1f} hours",
                "description": "Median time spent in ED",
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
                "metric": f"{(filtered_df['painmanage'] == 'Yes').mean()*100:.0f}%",
                "description": "Patients receiving pain management",
                "color": "#2ecc71"
            },
            {
                "step": 4,
                "title": "Early Mobilisation",
                "metric": f"{(filtered_df['mobil2'] == 'Day 1').mean()*100:.0f}%",
                "description": "Mobilised on day 1 after surgery",
                "color": "#f39c12"
            }
        ]
        
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
            # Time to surgery distribution
            fig_surgery_time = px.histogram(
                filtered_df,
                x='time_to_surgery_hours',
                nbins=30,
                title="Time to Surgery Distribution",
                labels={'time_to_surgery_hours': 'Hours'},
                color_discrete_sequence=['#3498db']
            )
            fig_surgery_time.add_vline(
                x=filtered_df['time_to_surgery_hours'].median(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: {filtered_df['time_to_surgery_hours'].median():.1f}h"
            )
            st.plotly_chart(fig_surgery_time, use_container_width=True)
        
        with col2:
            # Delay reasons
            delay_counts = filtered_df['delay'].value_counts()
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
            anaesth_counts = filtered_df['anaesth'].value_counts()
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
        mobil_order = ['Day 1', 'Day 2', 'Day 3+', 'Not mobilised']
        mobil_counts = filtered_df['mobil2'].value_counts().reindex(mobil_order, fill_value=0)
        
        fig_mobil = px.bar(
            x=mobil_counts.index,
            y=mobil_counts.values,
            title="When Patients First Mobilise After Surgery",
            labels={'x': 'Time to Mobilisation', 'y': 'Number of Patients'},
            color=mobil_counts.index,
            color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        )
        st.plotly_chart(fig_mobil, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">📊 Patient Outcomes</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            los_median = filtered_df['acute_los_days'].median()
            st.metric(
                "Acute Hospital Stay",
                f"{los_median:.0f} days",
                "median length of stay",
                delta_color="off"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Length of stay distribution
            fig_los = px.histogram(
                filtered_df,
                x='acute_los_days',
                nbins=20,
                title="Acute Hospital Stay Duration",
                labels={'acute_los_days': 'Days'},
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
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            rehab_pct = (filtered_df['wdest'] == 'Rehab').mean() * 100
            st.metric(
                "Go to Rehabilitation",
                f"{rehab_pct:.0f}%",
                "of patients",
                delta_color="off"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Discharge destination
            dest_counts = filtered_df['wdest'].value_counts()
            fig_dest = px.pie(
                values=dest_counts.values,
                names=dest_counts.index,
                title="Discharge Destination",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_dest, use_container_width=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            new_care_pct = (filtered_df['new_residential_care'] == 'Yes').mean() * 100
            st.metric(
                "New Residential Care",
                f"{new_care_pct:.1f}%",
                "need new placement",
                delta_color="off"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Mortality
            mortality_30d = filtered_df['mort30d'].mean() * 100
            fig_mort = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mortality_30d,
                title={'text': "30-Day Mortality"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 20]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 10], 'color': "yellow"},
                        {'range': [10, 20], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': mortality_30d
                    }
                }
            ))
            fig_mort.update_layout(height=300)
            st.plotly_chart(fig_mort, use_container_width=True)
        
        # Outcome comparisons by age group
        st.markdown('<h3 class="sub-header">Outcomes by Age Group</h3>', unsafe_allow_html=True)
        
        # Create age groups
        filtered_df['age_group'] = pd.cut(
            filtered_df['age'],
            bins=[60, 70, 80, 90, 100],
            labels=['60-69', '70-79', '80-89', '90+']
        )
        
        outcome_metrics = filtered_df.groupby('age_group').agg({
            'time_to_surgery_hours': 'median',
            'acute_los_days': 'median',
            'mort30d': 'mean'
        }).reset_index()
        
        # Create comparison chart
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Time to Surgery (hours)',
            x=outcome_metrics['age_group'],
            y=outcome_metrics['time_to_surgery_hours'],
            yaxis='y',
            offsetgroup=0
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Hospital Stay (days)',
            x=outcome_metrics['age_group'],
            y=outcome_metrics['acute_los_days'],
            yaxis='y2',
            offsetgroup=1
        ))
        
        fig_comparison.update_layout(
            title='Outcomes by Age Group',
            yaxis=dict(title='Hours to Surgery'),
            yaxis2=dict(title='Days in Hospital', overlaying='y', side='right'),
            barmode='group'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
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
            """)
        
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
        
        summary_cols = st.columns(4)
        summary_metrics = [
            ("Median Age", f"{filtered_df['age'].median():.0f} years"),
            ("Time to Surgery", f"{filtered_df['time_to_surgery_hours'].median():.0f} hours"),
            ("Hospital Stay", f"{filtered_df['acute_los_days'].median():.0f} days"),
            ("Rehabilitation", f"{(filtered_df['wdest'] == 'Rehab').mean()*100:.0f}%"),
        ]
        
        for i, (title, value) in enumerate(summary_metrics):
            with summary_cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h4 style="color: #7f8c8d; margin: 0;">{title}</h4>
                    <h2 style="color: #2c3e50; margin: 10px 0;">{value}</h2>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()