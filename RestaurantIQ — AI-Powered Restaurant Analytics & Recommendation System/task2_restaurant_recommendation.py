import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_recommendation_engine():
    df = pd.read_csv("Dataset.csv")
    
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    df['City'] = df['City'].fillna('Unknown')
    df['Locality'] = df['Locality'].fillna('Unknown')
    
    df['search_text'] = (
        df['Cuisines'].astype(str) + " " + 
        df['City'].astype(str) + " " + 
        df['Locality'].astype(str)
    )
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['search_text'])
    
    return df, tfidf, tfidf_matrix

def recommend_restaurants(df, tfidf, tfidf_matrix, preferred_cuisine=None, preferred_city=None, max_price_range=None, min_rating=0.0, top_n=5):
    filtered_df = df[df['Aggregate rating'] >= min_rating].copy()
    
    if max_price_range is not None:
        filtered_df = filtered_df[filtered_df['Price range'] <= max_price_range]
        
    if preferred_city is not None:
        city_filtered = filtered_df[filtered_df['City'].str.contains(preferred_city, case=False, na=False)]
        if not city_filtered.empty:
            filtered_df = city_filtered

    if filtered_df.empty:
        return pd.DataFrame()

    query_str = ""
    if preferred_cuisine:
        query_str += preferred_cuisine + " "
    if preferred_city:
        query_str += preferred_city

    if query_str.strip():
        query_vec = tfidf.transform([query_str])
        sub_indices = filtered_df.index
        sub_matrix = tfidf_matrix[sub_indices]
        
        sim_scores = cosine_similarity(query_vec, sub_matrix).flatten()
        filtered_df['Similarity'] = sim_scores
        
        filtered_df['Score'] = filtered_df['Similarity'] * 0.6 + (filtered_df['Aggregate rating'] / 5.0) * 0.4
        recommended = filtered_df.sort_values(by='Score', ascending=False).head(top_n)
    else:
        recommended = filtered_df.sort_values(by='Aggregate rating', ascending=False).head(top_n)
        
    cols = ['Restaurant Name', 'City', 'Locality', 'Cuisines', 'Price range', 'Aggregate rating', 'Votes']
    return recommended[cols]

def run_task2():
    df, tfidf, tfidf_matrix = create_recommendation_engine()
    
    print("=" * 60)
    print("TASK 2: RESTAURANT RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    test_cases = [
        {"preferred_cuisine": "Italian", "preferred_city": "New Delhi", "max_price_range": 3, "min_rating": 3.5},
        {"preferred_cuisine": "Chinese", "preferred_city": "Gurgaon", "max_price_range": 2, "min_rating": 4.0},
        {"preferred_cuisine": "North Indian", "preferred_city": None, "max_price_range": 4, "min_rating": 4.2}
    ]
    
    all_recommendations = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Sample Test Case {i} ---")
        print(f"Preferences: Cuisine={test['preferred_cuisine']}, City={test['preferred_city']}, Max Price={test['max_price_range']}, Min Rating={test['min_rating']}")
        
        recs = recommend_restaurants(
            df, tfidf, tfidf_matrix,
            preferred_cuisine=test['preferred_cuisine'],
            preferred_city=test['preferred_city'],
            max_price_range=test['max_price_range'],
            min_rating=test['min_rating'],
            top_n=5
        )
        
        if not recs.empty:
            safe_recs = recs.copy()
            safe_recs['Restaurant Name'] = safe_recs['Restaurant Name'].astype(str).str.encode('ascii', 'ignore').str.decode('ascii')
            print(safe_recs.to_string(index=False))
            recs['Test_Case'] = f"Case_{i}"
            all_recommendations.append(recs)
        else:
            print("No restaurants matched the criteria.")

    if all_recommendations:
        final_df = pd.concat(all_recommendations, ignore_index=True)
        final_df.to_csv("task2_sample_recommendations.csv", index=False, encoding='utf-8-sig')
        print("\nSaved sample recommendations to task2_sample_recommendations.csv")

if __name__ == "__main__":
    run_task2()
