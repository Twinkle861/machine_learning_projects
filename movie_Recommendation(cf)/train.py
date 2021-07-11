#using collabrative filtering ie views given by others
# 1see other user similar to one we need to predict(user to user cf)
#2 recommend movie on based of other movies ie item to item cf->work better as choices change with tme
#take angular dis than eucledian dis or use pearson dis ie modified version of angukar dis(cos theta)-mean
#if nan then do 0 then standarise the ratings bring mean =0 and range of 1
# Trnspose matrix as item to item and recommend on basis of movie if user to user dont
#in similar_score -2.5 as if user given bad rating then want it to not recommend therfore subtract the  mean ie rating<3 go more down

###python code
import pandas as pd
import pickle
from scipy import sparse
import streamlit as st
from PIL import Image
import time 
image = Image.open('img.jpg')

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.merge(movies,ratings).drop(['genres','timestamp'],axis=1)
# print(ratings.shape)
# ratings.head()

userRatings = ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
# userRatings.head()
# print("Before: ",userRatings.shape)
#drop movies reviwed by less than 10 users
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0,axis=1)
#userRatings.fillna(0, inplace=True)
# print("After: ",userRatings.shape)

def standardize(row):
    new_row = (row - row.mean())/(row.max()-row.min())
    return new_row
userRatings = userRatings.apply(standardize)

corrMatrix = userRatings.corr(method='pearson')
# corrMatrix.head(100)

def get_similar(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_ratings

# action_lover = [("Amazing Spider-Man, The (2012)",5),("Mission: Impossible III (2006)",4),("Toy Story 3 (2010)",2),("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)",4)]

# similar_movies = pd.DataFrame()
# for movie,rating in action_lover:
    # similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)

# #similar_movies.head(10)
# output = similar_movies.sum().sort_values(ascending=False).head(20)
# output = output.to_dict()
# print(output)



def main():
    st.title("Movie recommendation system using collaborative filtering")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #a7c4bc
    }
    </style>
    """, unsafe_allow_html=True)
    
    input1=[]
    movie_input = st.text_area("Enter name of movies")
    rate_input = st.text_area("Enter their respective rating that you will give to movie on scale of 1-5",0)

    def pr(output):
        for a,b in output.items():
            st.success("Movie name: " + a )
            st.success(" Rating: " + str(b))
            st.write(' ')
    
    if st.button('Recommend'):
        i1=movie_input.split('\n')
        i2=rate_input.split('\n')
        while(len(i1)>len(i2)):
            i2.append(0)
        for x in range(0,len(i1)):
            a=i1[x]
            b=int(i2[x])
            input1.append((a,b))
        # print(input1)
        if len(input1)==0:
            st.error("Enter atleast 1 movie")
        else:
            similar_movies = pd.DataFrame()
            for movie,rating in input1:
                similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)
            output = similar_movies.sum().sort_values(ascending=False).head(20)
            output = output.to_dict()
            (pr(output))
            # print(output)



if __name__=='__main__':
    main()