import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
try:
    df = pd.read_csv('E:\pro\cc\copp.csv', engine='python', error_bad_lines=False)
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    # Handle the error or skip problematic lines
    df = pd.DataFrame()

# Check if the DataFrame is empty
if not df.empty:
    # Check for non-numeric values in 'average_rating' column
    non_numeric_indices = df['average_rating'].apply(lambda x: not pd.to_numeric(x, errors='coerce')).index
    non_numeric_values = df.loc[non_numeric_indices, 'average_rating']
    print(f"Non-numeric values found in 'average_rating' column:\n{non_numeric_values}")

    # Convert 'average_rating' column to numeric
    df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')

    # Filter books with ratings_count greater than 1,000,000
    top_ten = df[df['ratings_count'] > 1000000]

    # Bar plot of average_rating for the top books
    sns.barplot(x="average_rating", y="title", data=top_ten, palette='inferno')
    plt.show()

    # Histogram of average_rating for all books
    sns.histplot(df['average_rating'].dropna().astype(float), kde=True, color='red')
    plt.show()

    # Bar plot of most-read books
    most_books = df.sort_values(by='ratings_count', ascending=False).head(10)
    ax = sns.barplot(x=most_books['ratings_count'], y=most_books['title'], palette='inferno')
    ax.set(xlabel='Ratings Count', ylabel='Book Title')
    plt.show()

    df.average_rating = df.average_rating.astype(float)
    fig, ax = plt.subplots(figsize=[15, 10])
    sns.distplot(df['average_rating'], ax=ax)
    ax.set_title('Average rating distribution for all books', fontsize=20)
    ax.set_xlabel('Average rating', fontsize=13)

    ax = sns.relplot(data=df, x="average_rating", y="ratings_count", color='red', sizes=(100, 200), height=7, marker='o')
    plt.title("Relation between Rating counts and Average Ratings", fontsize=15)
    ax.set_axis_labels("Average Rating", "Ratings Count")
    plt.figure(figsize=(15, 10))

    ax = sns.relplot(x="average_rating", y="num_pages", data=df, color='red', sizes=(100, 200), height=7, marker='o')
    ax.set_axis_labels("Average Rating", "Number of Pages")

    df2 = df.copy()
    df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
    df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
    df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
    df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
    df2.loc[(df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"
    
    rating_df = pd.get_dummies(df2['rating_between'])
    language_df = pd.get_dummies(df2['language_code'])
    
    features = pd.concat([rating_df, language_df, df2['average_rating'], df2['ratings_count']], axis=1)

    # Fitting the k-nearest neighbors model
    min_max_scaler = MinMaxScaler()
    features_scaled = min_max_scaler.fit_transform(features)

    model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features_scaled)

    # Compute idlist using the fitted model
    dist, idlist = model.kneighbors(features_scaled)

    def BookRecommender(book_name):
        book_list_name = []
        book_id = df2[df2['title'] == book_name].index
        book_id = book_id[0]
        
        for newid in idlist[book_id]:
            book_list_name.append(df2.loc[newid].title)
        
        return book_list_name

    # Example of using the BookRecommender function
    BookNames = BookRecommender('Harry Potter and the Half-Blood Prince (Harry Potter  #6)')
    print("Book recommendations:")
    print(BookNames)

else:
    print("DataFrame is empty. Please check the CSV file and loading process.")
