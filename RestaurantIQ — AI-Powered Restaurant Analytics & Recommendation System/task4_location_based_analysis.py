import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

def run_task4():
    df = pd.read_csv("Dataset.csv")
    
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    df['City'] = df['City'].fillna('Unknown')
    df['Locality'] = df['Locality'].fillna('Unknown')
    
    valid_coords = df[(df['Latitude'] != 0.0) & (df['Longitude'] != 0.0)].copy()
    
    print("=" * 60)
    print("TASK 4: LOCATION-BASED GEOGRAPHICAL ANALYSIS")
    print("=" * 60)
    print(f"\nTotal Restaurants in Dataset : {len(df)}")
    print(f"Restaurants with Valid GPS  : {len(valid_coords)}")
    
    city_stats = df.groupby('City').agg(
        Restaurant_Count=('Restaurant ID', 'count'),
        Average_Rating=('Aggregate rating', 'mean'),
        Average_Votes=('Votes', 'mean'),
        Average_Price_Range=('Price range', 'mean')
    ).reset_index().sort_values(by='Restaurant_Count', ascending=False)
    
    print("\nTop 10 Cities by Restaurant Concentration:")
    print(city_stats.head(10).to_string(index=False))
    
    locality_stats = df.groupby(['City', 'Locality']).agg(
        Restaurant_Count=('Restaurant ID', 'count'),
        Average_Rating=('Aggregate rating', 'mean'),
        Average_Price_Range=('Price range', 'mean')
    ).reset_index().sort_values(by='Restaurant_Count', ascending=False)
    
    print("\nTop 10 Localities by Restaurant Concentration:")
    print(locality_stats.head(10).to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    top_cities = city_stats.head(10)
    sns.barplot(x='Restaurant_Count', y='City', data=top_cities, palette='magma')
    plt.title('Top 10 Cities by Restaurant Count')
    plt.xlabel('Number of Restaurants')
    plt.ylabel('City')
    plt.tight_layout()
    plt.savefig("task4_top_cities_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Longitude', y='Latitude', 
        hue='Aggregate rating', size='Price range',
        data=valid_coords, palette='coolwarm', alpha=0.7
    )
    plt.title('Geographical Distribution of Restaurants (Latitude vs Longitude)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig("task4_geo_distribution.png", dpi=300)
    plt.close()
    
    if not valid_coords.empty:
        center_lat = valid_coords['Latitude'].mean()
        center_lon = valid_coords['Longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        sample_coords = valid_coords.sample(n=min(500, len(valid_coords)), random_state=42)
        for _, row in sample_coords.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=4,
                popup=f"{row['Restaurant Name']} | Rating: {row['Aggregate rating']}",
                color='red' if row['Aggregate rating'] >= 4.0 else 'blue',
                fill=True,
                fill_opacity=0.6
            ).add_to(m)
            
        m.save("task4_interactive_map.html")
        print("\nInteractive Map saved to task4_interactive_map.html")

    city_stats.to_csv("task4_city_analysis_summary.csv", index=False)
    locality_stats.to_csv("task4_locality_analysis_summary.csv", index=False)
    print("Saved summary statistics to task4_city_analysis_summary.csv & task4_locality_analysis_summary.csv")

if __name__ == "__main__":
    run_task4()
