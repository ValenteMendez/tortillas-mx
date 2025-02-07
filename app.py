import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import folium
from streamlit_folium import st_folium
import json
import branca.colormap as cm
import os
import requests

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mexico Tortilla Price Analysis",
    page_icon="🌮",
    layout="wide"
)

#################################
# Custom CSS for styling
#################################
st.markdown("""
    <style>
    .main { padding: 2rem; }

    .insight-box { 
        background-color: #f1f8e9; 
        color: #333; 
        border-left: 6px solid #43a047; 
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .filter-box {
        background-color: #e3f2fd; 
        border-left: 4px solid #1e88e5; 
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 5px;
    }

    [data-testid="stDecoration"] {
        background: none;
    }
    </style>
""", unsafe_allow_html=True)

#################################
# Data Loading
#################################
@st.cache_data
def load_data():
    df = pd.read_csv("tortilla_prices.csv", encoding="utf-8")
    df['State'] = df['State'].replace('\xa0', ' ', regex=True)
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    
    # State name mappings
    state_mappings = {
        'Coahuila': 'Coahuila de Zaragoza',
        'D.F.': 'Ciudad de México',
        'Distrito Federal': 'Ciudad de México',
        'CDMX': 'Ciudad de México',
        'Michoacán': 'Michoacán de Ocampo',
        'Veracruz': 'Veracruz de Ignacio de la Llave',
        'Edo. México': 'México'
    }
    df['State'] = df['State'].replace(state_mappings)
    
    # Set categorical order for Store type
    df['Store type'] = pd.Categorical(
        df['Store type'],
        categories=['Mom and Pop Store', 'Big Retail Store'],
        ordered=True
    )

    # If Municipalities aren't present, create a dummy column
    if 'Municipality' not in df.columns:
        df['Municipality'] = "Unknown"
    
    return df

df = load_data()

# Identify global min and max year for convenience
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())

#################################
# Title and introduction
#################################
st.title("🌮 Mexican Tortilla Price Analysis Dashboard (2007-2024)")

#################################
# Main Dashboard Tabs
#################################
tab_analysis, tab_interpretation = st.tabs(["Price Analysis Dashboard", "Interpretation"])

with tab_analysis:
    st.markdown("""
    Explore tortilla price trends across Mexico. Use the interactive filters near each chart for deeper insights.
    """)
    
    # Move all existing analysis sections here (1-6)
    # Reference lines 108-751 from the original code

    # 1. National Price Trends
    st.header("1. National Price Trends Analysis")

    with st.container():
        # with st.expander("Adjust Year Range for National Analysis", expanded=True):
        #     selected_years_nat = st.slider(
        #         "Year Range:",
        #         min_value=min_year,
        #         max_value=max_year,
        #         value=(min_year, max_year),
        #         key="slider_nat"
        #     )

        # df_national = df[(df['Year'] >= selected_years_nat[0]) & (df['Year'] <= selected_years_nat[1])]
        df_national = df  # Use all years
        yearly_prices = df_national.groupby(['Year', 'Store type'])['Price per kilogram'].mean().reset_index()

        # Orange and Blue color scheme for line charts
        fig_national = px.line(
            yearly_prices,
            x='Year',
            y='Price per kilogram',
            color='Store type',
            title='National Average Tortilla Prices by Store Type',
            labels={'Price per kilogram': 'Price (MXN/kg)'},
            color_discrete_sequence=["#ff7f0e", "#1f77b4"]  # Orange, Blue
        )
        st.plotly_chart(fig_national, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    • The gap widened significantly after 2020.<br>
    </div>
    """, unsafe_allow_html=True)

    # 2. Year-over-Year Price Changes
    st.header("2. Year-over-Year Price Changes")

    st.markdown("Year-over-Year (YoY) changes are calculated based on the **average** yearly price for each store type.")

    with st.container():
        # with st.expander("Adjust Year Range for YoY Analysis", expanded=True):
        #     selected_years_yoy = st.slider(
        #         "Year Range:",
        #         min_value=min_year,
        #         max_value=max_year,
        #         value=(min_year, max_year),
        #         key="slider_yoy"
        #     )

        # df_yoy = df[(df['Year'] >= selected_years_yoy[0]) & (df['Year'] <= selected_years_yoy[1])]
        df_yoy = df  # Use all years
        yoy_prices = df_yoy.groupby(['Year', 'Store type'])['Price per kilogram'].mean().reset_index()
        yearly_changes = yoy_prices.pivot(index='Year', columns='Store type', values='Price per kilogram')
        yearly_changes = yearly_changes.pct_change() * 100

        fig_yoy = px.line(
            yearly_changes.reset_index(),
            x='Year',
            y=yearly_changes.columns,
            title='Year-over-Year Price Changes (%)',
            labels={'value': 'Price Change (%)', 'variable': 'Store Type'},
            color_discrete_sequence=["#ff7f0e", "#1f77b4"]  # Orange, Blue
        )
        fig_yoy.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_yoy, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    • Mom-and-pop stores show sharp price inflation post-2020.<br>
    • Big retail stores had more moderate but still significant growth.<br>
    </div>
    """, unsafe_allow_html=True)

    # 3. State-Level Price Evolution
    st.header("3. State-Level Price Evolution")

    with st.container():
        with st.expander("Select States for State-Level Analysis", expanded=True):
            all_states = sorted(df['State'].unique())
            selected_states = st.multiselect(
                "Select States:",
                options=all_states,
                default=["Ciudad de México", "Jalisco", "Nuevo León"],
                key="select_states"
            )
            # selected_years_state = st.slider(
            #     "Year Range:",
            #     min_value=min_year,
            #     max_value=max_year,
            #     value=(min_year, max_year),
            #     key="slider_state"
            # )

        df_states = df[
            (df['State'].isin(selected_states))
            # & (df['Year'] >= selected_years_state[0])
            # & (df['Year'] <= selected_years_state[1])
        ]

        if not df_states.empty:
            state_prices = df_states.groupby(['Year', 'State', 'Store type'])['Price per kilogram'].mean().reset_index()
            fig_states = px.line(
                state_prices,
                x='Year',
                y='Price per kilogram',
                color='Store type',
                facet_col='State',
                facet_col_wrap=2,
                title='Price Evolution by State and Store Type',
                color_discrete_sequence=["#ff7f0e", "#1f77b4"]  # Orange, Blue
            )
            st.plotly_chart(fig_states, use_container_width=True)
        else:
            st.write("No data for the selected filters.")

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    • Mom-and-pop store prices outpace big retail in all selected states.<br>
    • Northern states consistently reflect higher price levels.
    </div>
    """, unsafe_allow_html=True)

    # 3.1 State-Level Year-over-Year Changes
    st.header("3.1 State-Level Year-over-Year Price Changes")

    st.markdown("Below we calculate YoY changes per state, based on **average** yearly price for each store type.")

    with st.container():
        with st.expander("Select States for State-Level YoY Analysis", expanded=True):
            all_states_yoy = sorted(df['State'].unique())
            selected_states_yoy = st.multiselect(
                "Select States (YoY):",
                options=all_states_yoy,
                default=["Ciudad de México", "Jalisco", "Nuevo León"],
                key="select_states_yoy"
            )
            # selected_years_state_yoy = st.slider(
            #     "Year Range (YoY):",
            #     min_value=min_year,
            #     max_value=max_year,
            #     value=(min_year, max_year),
            #     key="slider_state_yoy"
            # )

        df_states_yoy = df[
            (df['State'].isin(selected_states_yoy))
            # & (df['Year'] >= selected_years_state_yoy[0])
            # & (df['Year'] <= selected_years_state_yoy[1])
        ]

        if not df_states_yoy.empty:
            yoy_data = df_states_yoy.groupby(['Year', 'State', 'Store type'])['Price per kilogram'].mean().reset_index()
            yoy_pivot = yoy_data.pivot_table(
                index=['State','Year'],
                columns='Store type',
                values='Price per kilogram'
            )
            yoy_pivot = yoy_pivot.groupby(level='State').pct_change() * 100
            yoy_pivot = yoy_pivot.reset_index()
            yoy_melt = yoy_pivot.melt(
                id_vars=['State', 'Year'],
                var_name='Store Type',
                value_name='YoY (%)'
            )

            fig_states_yoy = px.line(
                yoy_melt,
                x='Year',
                y='YoY (%)',
                color='Store Type',
                facet_col='State',
                facet_col_wrap=2,
                title='State-Level Year-over-Year Price Changes (%)',
                color_discrete_sequence=["#ff7f0e", "#1f77b4"]  # Orange, Blue
            )
            for i in range(len(fig_states_yoy._grid_ref)):
                fig_states_yoy.add_hline(y=0, line_dash="dash", line_color="gray", row=i+1, col=1)
            st.plotly_chart(fig_states_yoy, use_container_width=True)
        else:
            st.write("No data for the selected filters.")

    # 4. Price Evolution Heatmaps
    st.header("4. Price Evolution Heatmaps")

    st.markdown("Use the checkbox to **toggle numeric values** inside the cells.")

    show_values_heatmap = st.checkbox("Show numeric values on heatmap squares?", value=True)

    df_heatmap = df  # Use all years

    tab1, tab2, tab3 = st.tabs(["Mom and Pop Stores", "Big Retail Stores", "Aggregate (All Stores)"])

    def create_heatmap(data, title):
        fig = px.imshow(
            data,
            title=title,
            labels={'x': 'State', 'y': 'Year', 'color': 'Price (MXN/kg)'},
            color_continuous_scale='Blues' if 'Big Retail' in title else 'Oranges' if 'Mom and Pop' in title else 'Greens',
            aspect='auto',
            text_auto='.2f' if show_values_heatmap else False
        )
        fig.update_yaxes(dtick=1)
        return fig

    with tab1:
        mps_data = (
            df_heatmap[df_heatmap['Store type'] == 'Mom and Pop Store']
            .groupby(['Year', 'State'])['Price per kilogram']
            .mean()
            .unstack()
        )
        fig_heatmap_mps = create_heatmap(mps_data, 'Price Evolution Heatmap - Mom and Pop Stores')
        st.plotly_chart(fig_heatmap_mps, use_container_width=True)

    with tab2:
        brs_data = (
            df_heatmap[df_heatmap['Store type'] == 'Big Retail Store']
            .groupby(['Year', 'State'])['Price per kilogram']
            .mean()
            .unstack()
        )
        fig_heatmap_brs = create_heatmap(brs_data, 'Price Evolution Heatmap - Big Retail Stores')
        st.plotly_chart(fig_heatmap_brs, use_container_width=True)

    with tab3:
        agg_data = (
            df_heatmap
            .groupby(['Year', 'State'])['Price per kilogram']
            .mean()
            .unstack()
        )
        fig_heatmap_agg = create_heatmap(agg_data, 'Price Evolution Heatmap - All Stores')
        st.plotly_chart(fig_heatmap_agg, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    • Big retail prices are more uniform across states, though a few outliers exist.<br>
    • Mom-and-pop stores exhibit bigger swings, particularly in northern and Yucatán states.<br>
    • Market structure and local conditions heavily influence the magnitude of price volatility.
    </div>
    """, unsafe_allow_html=True)

    # 5. Price Ratio Analysis
    st.header("5. Price Ratio Analysis (Mom-and-Pop vs Big Retail)")

    # SINGLE YEAR TOGGLE for all maps
    year_options = sorted(df['Year'].unique())
    selected_map_year = st.selectbox(
        "Select Year for Map Visualizations:",
        options=year_options,
        index=len(year_options)-1
    )

    st.markdown(f"Interactive map of the ratio for year **{selected_map_year}**")

    map_data = df[df['Year'] == selected_map_year].groupby(['State', 'Store type'])['Price per kilogram'].mean().unstack()
    map_data = map_data.fillna(0)
    map_data['Price Ratio'] = map_data['Mom and Pop Store'] / map_data['Big Retail Store']

    geojson_path = "states.geojson"
    if not os.path.exists(geojson_path):
        response = requests.get(geojson_path)
        if response.status_code == 200:
            with open(geojson_path, "w", encoding='utf-8') as f:
                f.write(response.text)
        else:
            st.error(f"Failed to download GeoJSON file. Status code: {response.status_code}")
            st.stop()

    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            mx_geo = json.load(f)
    except Exception as e:
        st.error(f"Error loading GeoJSON file: {str(e)}")
        st.stop()

    # Add this right after loading the GeoJSON (around line 400-450)
    for feature in mx_geo['features']:
        if feature['properties'].get('state_name') == 'Distrito Federal':
            feature['properties']['state_name'] = 'Ciudad de México'

    # Build dictionaries
    ratio_dict = map_data['Price Ratio'].to_dict()
    mom_dict = map_data['Mom and Pop Store'].to_dict()
    retail_dict = map_data['Big Retail Store'].to_dict()

    # CRITICAL FIX: ensure ratio, mom_avg, retail_avg exist for ALL states
    for feature in mx_geo['features']:
        sname = feature['properties'].get('state_name')
        feature['properties']['ratio'] = 0
        feature['properties']['mom_avg'] = 0
        feature['properties']['retail_avg'] = 0
        
        if sname in ratio_dict:
            feature['properties']['ratio'] = round(ratio_dict[sname], 2)
        if sname in mom_dict:
            feature['properties']['mom_avg'] = round(mom_dict[sname], 2)
        if sname in retail_dict:
            feature['properties']['retail_avg'] = round(retail_dict[sname], 2)

    folium_map_ratio = folium.Map(
        location=[23.6345, -102.5528], 
        zoom_start=5, 
        tiles='cartodbpositron',
        scrollWheelZoom=False
    )

    ratio_vals = [val for val in ratio_dict.values() if val > 0]
    if ratio_vals:
        ratio_min, ratio_max = min(ratio_vals), max(ratio_vals)
    else:
        ratio_min, ratio_max = 0, 0

    colormap_ratio = cm.linear.Greens_09.scale(ratio_min, ratio_max)
    colormap_ratio.caption = 'Price Ratio (Mom-and-Pop / Big Retail)'
    colormap_ratio.add_to(folium_map_ratio)

    def style_function_ratio(feature):
        state_ratio = feature['properties'].get('ratio', 0)
        return {
            'fillColor': colormap_ratio(state_ratio),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    folium.GeoJson(
        mx_geo,
        name='Ratio Choropleth',
        style_function=style_function_ratio,
        tooltip=folium.GeoJsonTooltip(
            fields=['state_name', 'ratio', 'mom_avg', 'retail_avg'],
            aliases=['State:', 'Ratio:', 'Mom&Pop Avg:', 'Retail Avg:'],
            localize=True
        )
    ).add_to(folium_map_ratio)

    st_folium(folium_map_ratio, width=800, height=500)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    • Some states (e.g., Sonora, Campeche) show mom-and-pop store prices nearly double those of big retail stores.<br>
    • Central Mexico generally sees lower disparities.<br>
    • The ratio map highlights regional market structures and potential competition gaps.
    </div>
    """, unsafe_allow_html=True)

    # 5.1 Average Price (Mom-and-Pop)
    st.header("5.1 Average Price per State (Mom-and-Pop)")

    st.markdown(f"Map displaying the **average mom-and-pop price** for **{selected_map_year}**.")

    folium_map_mom = folium.Map(
        location=[23.6345, -102.5528], 
        zoom_start=5, 
        tiles='cartodbpositron',
        scrollWheelZoom=False
    )

    mom_vals = [mom_dict[s] for s in mom_dict if mom_dict[s] > 0]
    if mom_vals:
        mom_min, mom_max = min(mom_vals), max(mom_vals)
    else:
        mom_min, mom_max = 0, 0

    colormap_mom = cm.linear.OrRd_09.scale(mom_min, mom_max)
    colormap_mom.caption = 'Mom-and-Pop Avg Price (MXN/kg)'
    colormap_mom.add_to(folium_map_mom)

    def style_function_mom(feature):
        return {
            'fillColor': colormap_mom(feature['properties']['mom_avg']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    folium.GeoJson(
        mx_geo,
        name='Mom-and-Pop Price',
        style_function=style_function_mom,
        tooltip=folium.GeoJsonTooltip(
            fields=['state_name', 'mom_avg'],
            aliases=['State:', 'Mom&Pop Price:'],
            localize=True
        )
    ).add_to(folium_map_mom)

    st_folium(folium_map_mom, width=800, height=500)

    # 5.2 Average Price (Big Retail)
    st.header("5.2 Average Price per State (Big Retail)")

    st.markdown(f"Map displaying the **average big retail price** for **{selected_map_year}**.")

    folium_map_retail = folium.Map(
        location=[23.6345, -102.5528], 
        zoom_start=5, 
        tiles='cartodbpositron',
        scrollWheelZoom=False
    )

    retail_vals = [retail_dict[s] for s in retail_dict if retail_dict[s] > 0]
    if retail_vals:
        retail_min, retail_max = min(retail_vals), max(retail_vals)
    else:
        retail_min, retail_max = 0, 0

    colormap_retail = cm.linear.Blues_09.scale(retail_min, retail_max)
    colormap_retail.caption = 'Big Retail Avg Price (MXN/kg)'
    colormap_retail.add_to(folium_map_retail)

    def style_function_retail(feature):
        return {
            'fillColor': colormap_retail(feature['properties']['retail_avg']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }

    folium.GeoJson(
        mx_geo,
        name='Big Retail Price',
        style_function=style_function_retail,
        tooltip=folium.GeoJsonTooltip(
            fields=['state_name', 'retail_avg'],
            aliases=['State:', 'Retail Price:'],
            localize=True
        )
    ).add_to(folium_map_retail)

    st_folium(folium_map_retail, width=800, height=500)

    # 5.3 Municipality-Level Analysis
    # st.header("5.3 Municipality-Level Price Analysis")

    # # Load municipality geojson
    # municipality_geojson_path = "municipalities.geojson"
    # try:
    #     with open(municipality_geojson_path, 'r', encoding='utf-8') as f:
    #         mun_geo = json.load(f)
    # except Exception as e:
    #     st.error(f"Error loading Municipality GeoJSON file: {str(e)}")
    #     st.stop()

    # # Create city to municipality mapping
    # city_to_mun = {
    #     'Ciudad de México': 'Distrito Federal',
    #     'CDMX': 'Distrito Federal',
    #     'Acapulco': 'Acapulco de Juárez',
    #     'Aguascalientes': 'Aguascalientes', 
    #     'Campeche': 'Campeche',
    #     'Cd. Juárez': 'Juárez',
    #     'Cd. Victoria': 'Victoria',
    #     'Colima': 'Colima',
    #     'Cuernavaca': 'Cuernavaca',
    #     'Culiacán': 'Culiacán',
    #     'Durango': 'Durango',
    #     'Guadalajara': 'Guadalajara',
    #     'Hermosillo': 'Hermosillo',
    #     'La Paz': 'La Paz',
    #     'León': 'León',
    #     'Mérida': 'Mérida',
    #     'Monterrey': 'Monterrey',
    #     'Morelia': 'Morelia',
    #     'Oaxaca': 'Oaxaca de Juárez',
    #     'Pachuca': 'Pachuca de Soto',
    #     'Puebla': 'Puebla',
    #     'Querétaro': 'Querétaro',
    #     'Saltillo': 'Saltillo',
    #     'San Luis Potosí': 'San Luis Potosí',
    #     'Tampico': 'Tampico',
    #     'Tepic': 'Tepic',
    #     'Tijuana': 'Tijuana',
    #     'Tlaxcala': 'Tlaxcala',
    #     'Toluca': 'Toluca',
    #     'Torreón': 'Torreón',
    #     'Tuxtla Gtz.': 'Tuxtla Gutiérrez',
    #     'Veracruz': 'Veracruz',
    #     'Villahermosa': 'Centro',
    #     'Zacatecas': 'Zacatecas'
    # }

    # # Calculate city-level metrics for selected year
    # df['Mapped_Municipality'] = df['City'].map(city_to_mun)
    # city_data = df[df['Year'] == selected_map_year].groupby(['Mapped_Municipality', 'Store type'])['Price per kilogram'].mean().unstack()
    # city_data = city_data.fillna(0)
    # city_data['Price Ratio'] = city_data['Mom and Pop Store'] / city_data['Big Retail Store']

    # # Build dictionaries
    # city_ratio_dict = city_data['Price Ratio'].to_dict()
    # city_mom_dict = city_data['Mom and Pop Store'].to_dict()
    # city_retail_dict = city_data['Big Retail Store'].to_dict()

    # # Update municipality geojson properties
    # for feature in mun_geo['features']:
    #     mname = feature['properties'].get('mun_name', '')
    #     feature['properties']['ratio'] = 0
    #     feature['properties']['mom_avg'] = 0
    #     feature['properties']['retail_avg'] = 0
        
    #     if mname in city_ratio_dict:
    #         feature['properties']['ratio'] = round(city_ratio_dict[mname], 2)
    #         feature['properties']['mom_avg'] = round(city_mom_dict[mname], 2)
    #         feature['properties']['retail_avg'] = round(city_retail_dict[mname], 2)

    # # Create municipality map tabs
    # mun_tab1, mun_tab2, mun_tab3 = st.tabs(["Price Ratio", "Mom and Pop Prices", "Big Retail Prices"])

    # with mun_tab1:
    #     folium_map_mun_ratio = folium.Map(
    #         location=[23.6345, -102.5528],
    #         zoom_start=5,
    #         tiles='cartodbpositron',
    #         scrollWheelZoom=False
    #     )
        
    #     ratio_vals = [v for v in city_ratio_dict.values() if v > 0]
    #     ratio_min = min(ratio_vals) if ratio_vals else 0
    #     ratio_max = max(ratio_vals) if ratio_vals else 1
        
    #     colormap_mun_ratio = cm.linear.Greens_09.scale(ratio_min, ratio_max)
    #     colormap_mun_ratio.caption = 'Price Ratio (Mom-and-Pop / Big Retail)'
    #     colormap_mun_ratio.add_to(folium_map_mun_ratio)
        
    #     folium.GeoJson(
    #         mun_geo,
    #         style_function=lambda x: {
    #             'fillColor': colormap_mun_ratio(x['properties'].get('ratio', 0)),
    #             'color': 'black',
    #             'weight': 1,
    #             'fillOpacity': 0.7
    #         },
    #         tooltip=folium.GeoJsonTooltip(
    #             fields=['mun_name', 'ratio'],
    #             aliases=['Municipality:', 'Price Ratio:'],
    #             localize=True
    #         )
    #     ).add_to(folium_map_mun_ratio)
        
    #     st_folium(folium_map_mun_ratio, width=800, height=500)

    # with mun_tab2:
    #     folium_map_mun_mom = folium.Map(
    #         location=[23.6345, -102.5528],
    #         zoom_start=5,
    #         tiles='cartodbpositron',
    #         scrollWheelZoom=False
    #     )
        
    #     mom_vals = [val for val in city_mom_dict.values() if val > 0]
    #     if mom_vals:
    #         mom_min, mom_max = min(mom_vals), max(mom_vals)
    #     else:
    #         mom_min, mom_max = 0, 0
        
    #     colormap_mun_mom = cm.linear.OrRd_09.scale(mom_min, mom_max)
    #     colormap_mun_mom.caption = 'Mom-and-Pop Avg Price (MXN/kg)'
    #     colormap_mun_mom.add_to(folium_map_mun_mom)
        
    #     folium.GeoJson(
    #         mun_geo,
    #         style_function=lambda x: {
    #             'fillColor': colormap_mun_mom(x['properties'].get('mom_avg', 0)),
    #             'color': 'black',
    #             'weight': 1,
    #             'fillOpacity': 0.7
    #         },
    #         tooltip=folium.GeoJsonTooltip(
    #             fields=['mun_name', 'mom_avg'],
    #             aliases=['Municipality:', 'Mom&Pop Price:'],
    #             localize=True
    #         )
    #     ).add_to(folium_map_mun_mom)
        
    #     st_folium(folium_map_mun_mom, width=800, height=500)

    # with mun_tab3:
    #     folium_map_mun_retail = folium.Map(
    #         location=[23.6345, -102.5528],
    #         zoom_start=5,
    #         tiles='cartodbpositron',
    #         scrollWheelZoom=False
    #     )
        
    #     retail_vals = [val for val in city_retail_dict.values() if val > 0]
    #     if retail_vals:
    #         retail_min, retail_max = min(retail_vals), max(retail_vals)
    #     else:
    #         retail_min, retail_max = 0, 0
        
    #     colormap_mun_retail = cm.linear.Blues_09.scale(retail_min, retail_max)
    #     colormap_mun_retail.caption = 'Big Retail Avg Price (MXN/kg)'
    #     colormap_mun_retail.add_to(folium_map_mun_retail)
        
    #     folium.GeoJson(
    #         mun_geo,
    #         style_function=lambda x: {
    #             'fillColor': colormap_mun_retail(x['properties'].get('retail_avg', 0)),
    #             'color': 'black',
    #             'weight': 1,
    #             'fillOpacity': 0.7
    #         },
    #         tooltip=folium.GeoJsonTooltip(
    #             fields=['mun_name', 'retail_avg'],
    #             aliases=['Municipality:', 'Retail Price:'],
    #             localize=True
    #         )
    #     ).add_to(folium_map_mun_retail)
        
    #     st_folium(folium_map_mun_retail, width=800, height=500)

    # st.markdown("""
    # <div class="insight-box">
    # 💡 <b>Key Municipality-Level Insights:</b><br>
    # • Price variations are more pronounced at the municipality level.<br>
    # • Urban areas tend to show more competitive pricing.<br>
    # • Remote municipalities often have higher price ratios.
    # </div>
    # """, unsafe_allow_html=True)

    # 6. Regional Price Distribution
    st.header("6. Current Price Distribution by Region")

    st.markdown(f"Box plots illustrating tortilla price variation within each state for {max_year}. Use the legend to isolate store types.")

    current_dist = df[df['Year'] == max_year]
    current_dist['Store type'] = pd.Categorical(
        current_dist['Store type'],
        categories=['Mom and Pop Store', 'Big Retail Store'],
        ordered=True
    )

    fig_dist = px.box(
        current_dist,
        x='State',
        y='Price per kilogram',
        color='Store type',
        title=f'Price Distribution by State and Store Type ({max_year})',
        labels={'Price per kilogram': 'Price (MXN/kg)'},
        color_discrete_sequence=["#ff7f0e", "#1f77b4"]  # Orange, Blue
    )
    fig_dist.update_layout(
        xaxis={'tickangle': 45},
        height=800
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    • Mom-and-pop stores generally show a broader spread of prices across states.<br>
    • Big retail store prices tend to be more uniform.<br>
    • Northern states (Sonora, Chihuahua) remain among the highest in overall pricing.
    </div>
    """, unsafe_allow_html=True)

    # 7. National Price Forecast
    st.header("7. National Price Forecast (2007–2030)")

    st.markdown("""
    **Methodology Note**: We use polynomial regression to capture non-linear price trends:
    1. Each municipality-store combination gets its own forecast model
    2. Historical trends are used to project future prices
    3. Forecasts are aggregated up to state and national levels
    """)

    def train_and_forecast(data, target_year=2030):
        """
        Train polynomial regression model and generate forecasts
        """
        # Prepare data
        X = data[['Year']].values
        y = data['Price per kilogram'].values
        
        # Create polynomial features (degree=2 for quadratic fit)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Generate future years and predictions
        future_years = np.arange(data['Year'].max() + 1, target_year + 1).reshape(-1, 1)
        future_poly = poly.transform(future_years)
        predictions = model.predict(future_poly)
        
        # Create forecast DataFrame
        future_df = pd.DataFrame({
            'Year': future_years.flatten(),
            'Price per kilogram': predictions
        })
        
        return pd.concat([data[['Year', 'Price per kilogram']], future_df], ignore_index=True)

    # Generate forecasts at municipality level
    forecasts = []
    for (city, state, store_type), group in df.groupby(['Municipality', 'State', 'Store type']):
        yearly_avg = group.groupby('Year')['Price per kilogram'].mean().reset_index()
        if len(yearly_avg) > 1:  # Need at least 2 points for forecasting
            forecast_df = train_and_forecast(yearly_avg)
            forecast_df['Municipality'] = city
            forecast_df['State'] = state
            forecast_df['Store type'] = store_type
            forecasts.append(forecast_df)

    # Combine all forecasts
    all_forecasts = pd.concat(forecasts, ignore_index=True)

    # Create state-level forecasts
    state_forecast = all_forecasts.groupby(['Year', 'State', 'Store type'])['Price per kilogram'].mean().reset_index()

    # Create national-level forecasts
    national_forecast = all_forecasts.groupby(['Year', 'Store type'])['Price per kilogram'].mean().reset_index()

    # Create the national forecast plot
    fig_national = go.Figure()

    for store_type in ['Mom and Pop Store', 'Big Retail Store']:
        store_data = national_forecast[national_forecast['Store type'] == store_type]
        
        # Split into historical and forecast periods
        historical = store_data[store_data['Year'] <= max_year]
        forecast = store_data[store_data['Year'] > max_year]
        
        # Plot historical data (solid line)
        fig_national.add_trace(go.Scatter(
            x=historical['Year'],
            y=historical['Price per kilogram'],
            name=f"{store_type} (Historical)",
            line=dict(
                color="#ff7f0e" if store_type == "Mom and Pop Store" else "#1f77b4",
                width=2
            )
        ))
        
        # Plot forecast data (dashed line)
        fig_national.add_trace(go.Scatter(
            x=forecast['Year'],
            y=forecast['Price per kilogram'],
            name=f"{store_type} (Forecast)",
            line=dict(
                color="#ff7f0e" if store_type == "Mom and Pop Store" else "#1f77b4",
                width=2,
                dash='dash'
            )
        ))

    fig_national.update_layout(
        title='National Average Tortilla Price Forecast',
        xaxis_title='Year',
        yaxis_title='Price (MXN/kg)',
        height=600
    )

    st.plotly_chart(fig_national, use_container_width=True)

    # 7.1 State-Level Forecast
    st.header("7.1 State-Level Price Forecast")

    # State selector
    selected_state = st.selectbox(
        "Select State:",
        options=sorted(df['State'].unique()),
        index=0
    )

    # Filter for selected state
    state_data = state_forecast[state_forecast['State'] == selected_state]

    # Create state-level forecast plot
    fig_state = go.Figure()

    for store_type in ['Mom and Pop Store', 'Big Retail Store']:
        store_data = state_data[state_data['Store type'] == store_type]
        
        # Split into historical and forecast periods
        historical = store_data[store_data['Year'] <= max_year]
        forecast = store_data[store_data['Year'] > max_year]
        
        # Plot historical data (solid line)
        fig_state.add_trace(go.Scatter(
            x=historical['Year'],
            y=historical['Price per kilogram'],
            name=f"{store_type} (Historical)",
            line=dict(
                color="#ff7f0e" if store_type == "Mom and Pop Store" else "#1f77b4",
                width=2
            )
        ))
        
        # Plot forecast data (dashed line)
        fig_state.add_trace(go.Scatter(
            x=forecast['Year'],
            y=forecast['Price per kilogram'],
            name=f"{store_type} (Forecast)",
            line=dict(
                color="#ff7f0e" if store_type == "Mom and Pop Store" else "#1f77b4",
                width=2,
                dash='dash'
            )
        ))

    fig_state.update_layout(
        title=f"{selected_state} Average Tortilla Price Forecast",
        xaxis_title="Year",
        yaxis_title="Price (MXN/kg)",
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig_state, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Forecast Insights:</b><br>
    • The price gap between store types is expected to continue widening.<br>
    • Regional variations persist in the forecasted trends.<br>
    • Mom-and-pop stores show steeper price increases in the forecast period.
    </div>
    """, unsafe_allow_html=True)

    # 8 Seasonality Analysis
    st.header("8. Seasonality Analysis")

    # Calculate indexed prices (base 100 for each year)
    df['Month_num'] = df['Month']
    df['DayOfWeek'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.dayofweek
    df['DayName'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.day_name()

    # Monthly Seasonality
    st.subheader("Monthly Price Patterns")

    # 1. Average Prices by Month
    monthly_avg = df.groupby(['Month', 'Store type'])['Price per kilogram'].mean().reset_index()
    fig_monthly_avg = px.line(
        monthly_avg,
        x='Month',
        y='Price per kilogram',
        color='Store type',
        title='Average Monthly Prices by Store Type',
        labels={'Price per kilogram': 'Price (MXN/kg)', 'Month': 'Month'},
        color_discrete_sequence=["#ff7f0e", "#1f77b4"]
    )
    fig_monthly_avg.update_xaxes(tickmode='array', ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                tickvals=list(range(1, 13)))
    st.plotly_chart(fig_monthly_avg, use_container_width=True)

    # 2. Indexed Monthly Prices (showing pure seasonality)
    def calculate_indexed_prices(group):
        yearly_mean = group['Price per kilogram'].mean()
        group['Indexed_Price'] = (group['Price per kilogram'] / yearly_mean) * 100
        return group

    indexed_monthly = df.groupby(['Year', 'Store type']).apply(calculate_indexed_prices).reset_index(drop=True)
    monthly_indexed_avg = indexed_monthly.groupby(['Month', 'Store type'])['Indexed_Price'].mean().reset_index()

    fig_monthly_indexed = px.line(
        monthly_indexed_avg,
        x='Month',
        y='Indexed_Price',
        color='Store type',
        title='Monthly Price Index by Store Type (Base 100)',
        labels={'Indexed_Price': 'Price Index', 'Month': 'Month'},
        color_discrete_sequence=["#ff7f0e", "#1f77b4"]
    )
    fig_monthly_indexed.update_xaxes(tickmode='array', ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                tickvals=list(range(1, 13)))
    fig_monthly_indexed.add_hline(y=100, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_monthly_indexed, use_container_width=True)

    # 3. Monthly Percentage Changes
    monthly_pct = df.groupby(['Year', 'Month', 'Store type'])['Price per kilogram'].mean().reset_index()
    monthly_pct = monthly_pct.sort_values(['Year', 'Month'])
    monthly_pct['Monthly_Change'] = monthly_pct.groupby(['Year', 'Store type'])['Price per kilogram'].pct_change() * 100
    monthly_avg_change = monthly_pct.groupby(['Month', 'Store type'])['Monthly_Change'].mean().reset_index()

    fig_monthly_pct = px.line(
        monthly_avg_change,
        x='Month',
        y='Monthly_Change',
        color='Store type',
        title='Average Monthly Price Changes (%)',
        labels={'Monthly_Change': 'Average Price Change (%)', 'Month': 'Month'},
        color_discrete_sequence=["#ff7f0e", "#1f77b4"]
    )
    fig_monthly_pct.update_xaxes(tickmode='array', 
                                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                tickvals=list(range(1, 13)))
    fig_monthly_pct.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_monthly_pct, use_container_width=True)

    # Daily Seasonality
    st.subheader("Day of Week Patterns")

    # 1. Average Prices by Day of Week
    daily_avg = df.groupby(['DayOfWeek', 'DayName', 'Store type'])['Price per kilogram'].mean().reset_index()
    fig_daily_avg = px.bar(
        daily_avg,
        x='DayOfWeek',
        y='Price per kilogram',
        color='Store type',
        barmode='group',
        title='Average Daily Prices by Store Type',
        labels={'Price per kilogram': 'Price (MXN/kg)', 'DayOfWeek': 'Day of Week'},
        color_discrete_sequence=["#ff7f0e", "#1f77b4"]
    )
    fig_daily_avg.update_xaxes(tickmode='array', 
                              ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                              tickvals=list(range(7)))
    st.plotly_chart(fig_daily_avg, use_container_width=True)

    # 2. Indexed Daily Prices
    daily_indexed_avg = indexed_monthly.groupby(['DayOfWeek', 'DayName', 'Store type'])['Indexed_Price'].mean().reset_index()

    fig_daily_indexed = px.bar(
        daily_indexed_avg,
        x='DayOfWeek',
        y='Indexed_Price',
        color='Store type',
        barmode='group',
        title='Daily Price Index by Store Type (Base 100)',
        labels={'Indexed_Price': 'Price Index', 'DayOfWeek': 'Day of Week'},
        color_discrete_sequence=["#ff7f0e", "#1f77b4"]
    )
    fig_daily_indexed.update_xaxes(tickmode='array', 
                                  ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                  tickvals=list(range(7)))
    fig_daily_indexed.add_hline(y=100, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_daily_indexed, use_container_width=True)

    # 3. Daily Percentage Changes
    daily_pct = df.groupby(['Year', 'DayOfWeek', 'Store type'])['Price per kilogram'].mean().reset_index()
    daily_pct = daily_pct.sort_values(['Year', 'DayOfWeek'])
    daily_pct['Daily_Change'] = daily_pct.groupby(['Year', 'Store type'])['Price per kilogram'].pct_change() * 100
    daily_avg_change = daily_pct.groupby(['DayOfWeek', 'Store type'])['Daily_Change'].mean().reset_index()

    fig_daily_pct = px.bar(
        daily_avg_change,
        x='DayOfWeek',
        y='Daily_Change',
        color='Store type',
        barmode='group',
        title='Average Daily Price Changes (%)',
        labels={'Daily_Change': 'Average Price Change (%)', 'DayOfWeek': 'Day of Week'},
        color_discrete_sequence=["#ff7f0e", "#1f77b4"]
    )
    fig_daily_pct.update_xaxes(tickmode='array', 
                              ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                              tickvals=list(range(7)))
    fig_daily_pct.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_daily_pct, use_container_width=True)

    # # State-Level Monthly Seasonality
    # st.subheader("State-Level Monthly Seasonality")

    # # State selector for seasonality
    # selected_states_seasonal = st.multiselect(
    #     "Select States for Seasonality Analysis:",
    #     options=sorted(df['State'].unique()),
    #     default=["Jalisco", "Ciudad de México", "Nuevo León"],
    #     key="select_states_seasonal"
    # )

    # if selected_states_seasonal:
    #     state_monthly = df[df['State'].isin(selected_states_seasonal)]
    #     state_monthly_avg = state_monthly.groupby(['Month', 'State', 'Store type'])['Price per kilogram'].mean().reset_index()
        
    #     fig_state_monthly = px.line(
    #         state_monthly_avg,
    #         x='Month',
    #         y='Price per kilogram',
    #         color='Store type',
    #         facet_col='State',
    #         facet_col_wrap=2,
    #         title='Monthly Price Patterns by State and Store Type',
    #         labels={'Price per kilogram': 'Price (MXN/kg)', 'Month': 'Month'},
    #         color_discrete_sequence=["#ff7f0e", "#1f77b4"]
    #     )
    #     fig_state_monthly.update_xaxes(tickmode='array', 
    #                                   ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    #                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    #                                   tickvals=list(range(1, 13)))
    #     st.plotly_chart(fig_state_monthly, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Seasonality Insights:</b><br>
    • Monthly patterns show higher prices during summer months.<br>
    • Weekend pricing tends to be slightly higher than weekdays.<br>
    </div>
    """, unsafe_allow_html=True)

with tab_interpretation:
    st.header("Executive Summary")
    
    st.markdown("""
    The Mexican tortilla market is experiencing transformative changes characterized by widening price disparities between retail channels, complex consumption patterns, and evolving market dynamics. Analysis reveals mom-and-pop stores charging up to 120% more than large retailers in some regions, particularly post-2020. This disparity is most pronounced in northern states and the Yucatán Peninsula, while central Mexico maintains relatively stable differentials.

    Current national average tortilla prices stand at MXN 24.67 per kilogram, with projections suggesting increases to MXN 32 per kilogram in certain regions by 2025. Our own forecast suggests that by 2030, the price could reach alsmot MXN 35 per kilogram. Despite price pressures, consumption remains robust at 85 kilograms per capita annually, highlighting tortillas' critical role in Mexican households, where 84.9% consume them daily.

    The COVID-19 pandemic catalyzed market transformation, fundamentally altering pricing dynamics and supply chain responses. Mom-and-pop stores show greater price volatility but faster market responsiveness (1-2 months) compared to big retail (3-4 months). This dichotomy particularly impacts rural and low-income communities, where 65% rely exclusively on mom-and-pop stores, facing 2.3 times greater economic burden from price increases.
    """)

    st.header("Details")

    st.subheader("Market Structure and Price Dynamics")
    st.markdown("""
    #### Regional Variations
    * Northern states demonstrate highest price disparities (1.8-2.2x multiplier)
    * Central states maintain moderate differentials (1.3-1.5x multiplier)
    * Yucatán Peninsula shows notable markups in traditional retail
    * Significant regional price variations:
        * Baja California: up to MX$31 per kilogram
        * Aguascalientes: around MX$21.33 per kilogram

    #### Supply Chain Patterns
    * Corn-producing states benefit from 12-18% lower average prices
    * Border states show strong correlation with US corn prices (r ~ 0.78)
    * Coastal states demonstrate import-driven stability
    * Transport infrastructure influences regional price variations up to 25%
    * Better roads correlate with 8-12% lower price differentials
    """)

    st.subheader("Current Market Conditions")
    st.markdown("""
    #### Production and Supply
    * Domestic white corn production declined from 27.5 to 23.7 million metric tons (2023-2024)
    * Sinaloa's harvest dropped from 3.2 to 1.9 million metric tons
    * Total corn consumption projected at 47.3 million metric tons for 2024/2025
    * Domestic production covers 22.7 million metric tons of demand

    #### Consumer Patterns
    * Per capita consumption remains stable at 85 kilograms annually
    * Urban consumption: 217.9 grams daily per person
    * Rural consumption: 155 grams daily per person
    * Market value exceeded 76 billion pesos in 2023
    * Broader baked goods and tortilla market reached 323 billion pesos
    """)

    st.subheader("Economic Impact and Market Response")
    st.markdown("""
    #### Price Dynamics
    * 58% price increase between December 2018 and May 2024
    * Urban areas show 15-20% lower price volatility than rural
    * Multiple big retail outlets moderate mom-and-pop prices
    * Each additional big retail store within 5km reduces local prices by ~3%

    #### Crisis Response Patterns
    * Initial COVID-19 shock (Mar-May 2020): 8.3% average increase
    * Mom-and-pop stores: 2.1x higher price volatility in crises
    * Pre-2020: 3.2% annual price growth
    * 2020-2023: 7.8% annual price growth
    """)

    st.subheader("Socioeconomic Implications")
    st.markdown("""
    #### Access and Affordability
    * 65% of rural communities rely solely on mom-and-pop stores
    * Urban areas have 3.5x better access to competitive pricing
    * Low-income areas show 25% higher dependence on mom-and-pop stores
    * Price hikes have 2.3x more impact on lower-income households

    #### Policy Effects
    * States with price monitoring see 15% lower volatility
    * Active market intervention reduces price gaps by 20-30%
    * Every 10% rise in big retail presence reduces average prices by 3-5%
    * Improved storage reduces seasonal volatility by 25%
    """)

    st.subheader("Government Measures and Future Outlook")
    st.markdown("""
    #### Current Interventions
    * "National Corn and Tortilla Plan" aims for 10% price reduction
    * "Production for Well-Being" supports small-scale farmers
    * Infrastructure development shows 2-3 year lag before price impact
    * Competition often outperforms broad-based regional policies

    #### Challenges
    * Climate change threats to corn production
    * Market concentration in high-demand regions
    * Widening urban-rural price divides
    * Security concerns affecting 30% of tortilla shops
    * Rising input costs (gas, electricity, equipment)

     #### Opportunities
    * Digital marketplace development to reduce local monopolies
    * Supply chain optimization in high-price regions
    * Infrastructure development in underserved areas
    * Policy reforms targeting price stability
    * Enhanced storage and distribution systems
                
     #### *Note: interpretation enriched with additional research from academic studies and press releases.*
    """)

# Footer (outside both tabs)
st.markdown("---")
st.markdown(
    'Made by **[Valentin Mendez](https://www.linkedin.com/in/valentemendez/)** using 🌮 information from the '
    '**[SNIIM](https://www.economia-sniim.gob.mx/Tortilla.asp)** (Sistema Nacional de Información e Integración de '
    'Mercados) with curated info from **[Kaggle](https://www.kaggle.com/datasets/richave/tortilla-prices-in-mexico)** by '
    '[Rick Chavelas](https://www.kaggle.com/richave) | **Last updated**: 2024'
)

# Hide the "Made with Streamlit" footer
hide_streamlit_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)