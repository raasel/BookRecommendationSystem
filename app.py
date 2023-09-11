import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import warnings
from fuzzywuzzy import process
from fuzzywuzzy import fuzz




###############--------code for Dataset import and Data Preprocessing-------------###################
books = pd.read_csv('./data/Books.csv',low_memory=False)
users = pd.read_csv('./data/Users.csv',low_memory=False)
ratings = pd.read_csv('./data/Ratings.csv',low_memory=False)

combine_book_rating = pd.merge(ratings, books, on='ISBN')
# combine_book_rating.drop_duplicates(subset=['userID', 'ISBN'], inplace=True)
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)

combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

book_ratingCount = (combine_book_rating.
     groupby(by = ['bookTitle'])['bookRating'].
     count().
     reset_index().
     rename(columns = {'bookRating': 'totalRatingCount'})
     [['bookTitle', 'totalRatingCount']]
    )

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')


popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount[rating_with_totalRatingCount['totalRatingCount'] >= popularity_threshold]

combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')

# us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")] #Filter USA Canada
us_canada_user_rating = combined
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)

us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

us_canada_user_rating_pivot2 = us_canada_user_rating.pivot(index = 'userID', columns = 'bookTitle', values = 'bookRating').fillna(0)

X = us_canada_user_rating_pivot2.values.T

SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)

warnings.filterwarnings("ignore",category =RuntimeWarning)
corr = np.corrcoef(matrix)

us_canada_book_title = us_canada_user_rating_pivot2.columns
us_canada_book_list = list(us_canada_book_title)

# coffey_hands = us_canada_book_list.index('A Bend in the Road')
# corr_coffey_hands  = corr[coffey_hands]
# listre= list(us_canada_book_title[(corr_coffey_hands > 0.9)])

# print(listre)
###############--------End for Dataset import and Data Preprocessing-------------###################


def get_recommendation(book_name):
    coffey_hands = us_canada_book_list.index(book_name)
    corr_coffey_hands  = corr[coffey_hands]
    list(us_canada_book_title[(corr_coffey_hands > 0.9)])
    return list(us_canada_book_title[(corr_coffey_hands > 0.9)])

def get_image_url(book_title):
    book_entry = books[books['bookTitle'] == book_title]
    if not book_entry.empty:
        return book_entry.iloc[0]['imageUrlL']
    return None

def search_books(query,book_list=us_canada_book_list,threshold=80):


    # Check for the presence of all keywords in each book title
    matches = process.extract(query, book_list, limit=len(book_list), scorer=fuzz.partial_ratio)
    return [match[0] for match in matches if match[1] >= threshold]

def main():
    st.title("Book Recommendation System")

    book_name = st.text_input("Enter Book Title")
    if st.button("Search") or book_name:
        try:
            recommended_books = get_recommendation(book_name)
            st.write("Recommended Books:")
            # Iterating over the books two at a time
            for i in range(0, len(recommended_books), 2):
                book1 = recommended_books[i]
                image_url1 = get_image_url(book1)
                
                # Check if there's a next book
                book2 = recommended_books[i + 1] if i + 1 < len(recommended_books) else None
                image_url2 = get_image_url(book2) if book2 else None

                col1, col2 = st.columns(2)

                if image_url1:
                    col1.success(book1)
                    col1.image(image_url1)

                if book2 and image_url2:
                    col2.success(book2)
                    col2.image(image_url2)
        except:
            st.warning("There is no Book with This Title. Write the Book Title Correctly")
            matched_books = search_books(book_name)
            st.success("Possible Title for a Book. Attempt with One of Them")
            for book in matched_books:
                st.write(book)
            




if __name__ == "__main__":
    main()