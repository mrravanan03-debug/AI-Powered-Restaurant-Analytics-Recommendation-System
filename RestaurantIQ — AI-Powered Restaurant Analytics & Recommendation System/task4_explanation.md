# Task 4: Location-Based Geographical Analysis

## 1. Project Overview & Objective
The objective of **Task 4** is to perform a spatial and geographical data analysis of the restaurants in the dataset. The analysis explores spatial coordinates (latitude and longitude), evaluates restaurant density across cities and localities, and analyzes key performance indicators (ratings, price points, vote volume) by geographical region.

## 2. Geospatial Preprocessing & Filtering
1. **Coordinate Validation**: Records with invalid GPS entries (`Latitude = 0.0` or `Longitude = 0.0`) are excluded from geographical map projections while retaining full administrative city/locality records for aggregated statistics.
2. **Geographical Aggregation**: Restaurants are grouped by `City` and `Locality` to evaluate spatial density and regional trends.

## 3. Analytical Findings & Spatial Insights
1. **City Concentration**: Metropolitan hubs contain the highest density of dining establishments, showing strong clustering effects.
2. **Rating Variations Across Regions**: Certain premium commercial localities exhibit consistently higher mean aggregate ratings and price ranges compared to residential outer suburbs.
3. **Geographical Scatter & Mapping**: Spatial plots demonstrate how dining hubs are clustered around major urban corridors and transport hubs.

## 4. Interactive Mapping & Visualizations
- **Static Distribution Charts**: Bar charts highlighting top cities by total restaurant counts and scatter plots mapping coordinates against rating spectrums.
- **Folium Interactive Map**: Generates an interactive web map (`task4_interactive_map.html`) displaying color-coded markers for restaurants (Red = High Rating $\ge 4.0$, Blue = Standard Rating $< 4.0$) with popup metadata.

## 5. Deliverables & Output Files
- **Python Execution Script**: [task4_location_based_analysis.py](file:///C:/Users/Admin/.gemini/antigravity/scratch/cognifyz_ml_internship/task4_location_based_analysis.py)
- **Top Cities Summary CSV**: `task4_city_analysis_summary.csv`
- **Locality Statistics CSV**: `task4_locality_analysis_summary.csv`
- **Geographical Distribution Plot**: `task4_top_cities_distribution.png`
- **Coordinate Scatter Plot**: `task4_geo_distribution.png`
- **Interactive HTML Map**: `task4_interactive_map.html`
