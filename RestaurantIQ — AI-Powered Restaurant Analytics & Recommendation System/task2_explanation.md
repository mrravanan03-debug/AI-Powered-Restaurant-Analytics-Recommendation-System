# Task 2: Restaurant Recommendation System

## 1. Project Overview & Objective
The goal of **Task 2** is to build a personalized content-based recommendation system that recommends top restaurants based on specific user criteria including preferred cuisines, target city, maximum price range, and minimum acceptable aggregate rating.

## 2. Recommendation Strategy & Criteria
The recommendation framework combines **Content-Based Filtering** with multi-attribute constraint filtering:
1. **Hard Constraints Filtering**:
   - **`City Filter`**: Restricts candidate restaurants to the user's requested city (if provided).
   - **`Price Range Constraint`**: Filters out restaurants exceeding the user's budget threshold (`Price range <= max_price_range`).
   - **`Quality Threshold`**: Ensures recommended restaurants meet or exceed the user's minimum rating requirement (`Aggregate rating >= min_rating`).
2. **Content-Based Similarity Search**:
   - **`TF-IDF (Term Frequency-Inverse Document Frequency)`**: Converts textual features (`Cuisines`, `City`, `Locality`) into high-dimensional vector representations.
   - **`Cosine Similarity Scoring`**: Computes the cosine angle between user preference query vectors and restaurant document vectors.

## 3. Hybrid Ranking Score Computation
To balance feature relevance with overall restaurant reputation, a composite ranking score is calculated:
\[
\text{Composite Score} = (0.6 \times \text{Cosine Similarity}) + \left(0.4 \times \frac{\text{Aggregate Rating}}{5.0}\right)
\]
This ensures recommendations are both highly relevant to the query terms and top-rated in overall quality.

## 4. Evaluation & Sample Test Scenarios
The system was tested against multiple user preference profiles:
- **Scenario 1**: Italian cuisine enthusiast seeking mid-tier dining in New Delhi with rating $\ge 3.5$.
- **Scenario 2**: Budget-conscious Chinese food seeker in Gurgaon with rating $\ge 4.0$.
- **Scenario 3**: Fine-dining North Indian restaurant seeker across all cities with rating $\ge 4.2$.

The output produces an ordered table of top 5 matched restaurants containing name, location, cuisine breakdown, price range, aggregate rating, and vote counts.

## 5. Deliverables & Output Files
- **Python Execution Script**: [task2_restaurant_recommendation.py](file:///C:/Users/Admin/.gemini/antigravity/scratch/cognifyz_ml_internship/task2_restaurant_recommendation.py)
- **Sample Recommendation Results CSV**: `task2_sample_recommendations.csv`
