import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import warnings
import numpy as np



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

print("File is called")
# coffey_hands = us_canada_book_list.index('A Bend in the Road')
# corr_coffey_hands  = corr[coffey_hands]
# listre= list(us_canada_book_title[(corr_coffey_hands > 0.9)])

# print(listre)