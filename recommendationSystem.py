import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample data
data = {
    'User': ['User1', 'User1', 'User2', 'User2', 'User3', 'User3'],
    'Movie': ['Inception', 'Titanic', 'Inception', 'Avatar', 'Titanic', 'Avatar'],
    'Rating': [5, 4, 4, 5, 3, 4]
}

df = pd.DataFrame(data)

# Create a pivot table
pivot_table = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

# Calculate cosine similarity
similarity = cosine_similarity(pivot_table)

# Recommend movies for User1
user_index = 0  
similar_users = similarity[user_index].argsort()[::-1][1:]  
recommended_movies = pivot_table.iloc[similar_users].mean(axis=0).sort_values(ascending=False)

print("Recommended Movies for User1:")
print(recommended_movies[recommended_movies > 0])

# Sample data
movies = {
    'Movie': ['Inception', 'Titanic', 'Avatar', 'Interstellar'],
    'Genre': ['Sci-Fi Thriller', 'Romance Drama', 'Sci-Fi Adventure', 'Sci-Fi Drama']
}

movies_df = pd.DataFrame(movies)

# Vectorize genres
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies_df['Genre'])

# Calculate cosine similarity
similarity = cosine_similarity(genre_matrix)

# Recommend movies similar to 'Inception'
movie_index = movies_df[movies_df['Movie'] == 'Inception'].index[0]
similar_movies = similarity[movie_index].argsort()[::-1][1:] 

print("Movies similar to 'Inception':")
for idx in similar_movies:
    print(movies_df.iloc[idx]['Movie'])
