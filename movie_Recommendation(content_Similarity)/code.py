# make recommendation system based on similarity b/w content
# find theta to find similarity use cosine similarity
# type of recommender-content based

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import time
image = Image.open('img.jpg')

# 3#################################################################################33


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")
# print df.columns

# Step 2: Select Features

features = ['keywords', 'cast', 'genres', 'director']
# Step 3: Create a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('')


def combine_features(row):
    try:
        return row['keywords'] + " "+row['cast']+" "+row["genres"]+" "+row["director"]
    except:
        print("Error:", row)


df["combined_features"] = df.apply(combine_features, axis=1)

# print "Combined Features:", df["combined_features"].head()

# Step 4: Create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

# Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

# Step 6: Get index of this movie from its title
# movie_index = get_index_from_title(movie_user_likes)

# similar_movies =  list(enumerate(cosine_sim[movie_index]))

# ## Step 7: Get a list of similar movies in descending order of similarity score
# sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

# ## Step 8: Print titles of first 50 movies
# i=0
# for element in sorted_similar_movies:
# 		print( get_title_from_index(element[0]))
# 		i=i+1
# 		if i>50:
# 			break
#################################################


def main():
    st.title("Movie recommendation system (content based recommendation)")
    st.image(image)
    st.markdown(
        """
    <style>
    .reportview-container {
        background-color: #a7c4bc
    }
    </style>
    """, unsafe_allow_html=True)

    movie_input = st.text_area("Enter name of movie")
    num = st.number_input("Enter no. of recommendations you want",min_value=1,max_value=70)

    def pr(output):
        i = 0
        for element in output:
            st.success(get_title_from_index(element[0]))
            i = i+1
            if i > num:
                break

    if st.button('Recommend'):
        if len(movie_input) == 0:
            st.error("Enter the name of the movie")
        else:
            movie_index = get_index_from_title(movie_input)
            similar_movies = list(enumerate(cosine_sim[movie_index]))
            sorted_similar_movies = sorted(
                similar_movies, key=lambda x: x[1], reverse=True)
            pr(sorted_similar_movies)


if __name__ == '__main__':
    main()
