from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import pandas as pd

def recommend_func(data, min_price, max_price, **args):
    """
    Recommends cars based on given filters.

    Parameters:
    - data: DataFrame, the dataset of cars
    - min_price: float, minimum price of the car
    - max_price: float, maximum price of the car
    - kwargs: dict, optional filtering parameters

    Returns:
    - DataFrame: recommended cars
    """
    # Apply filters based on provided parameters
    for key, value in args.items():
        if value:  # Check if value is provided
            data = data[data[key] == value]

    # Filter data based on price range
    data = data[(data['price'] >= min_price) & (data['price'] <= max_price)]

    # Reset index after filtering
    data.reset_index(drop=True, inplace=True)

    # Tokenize car details for TF-IDF vectorization
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=3, stop_words='english', token_pattern=r'\b\w+\b|\d+')
    tfidf_matrix = tf.fit_transform(data['car_details'])

    # Calculate similarity matrix using sigmoid kernel
    sg = sigmoid_kernel(tfidf_matrix, tfidf_matrix)

    # Get the important feature for recommendation
    imp_feature = list(args.keys())[0] if args else None

    if imp_feature:
        # Get indices corresponding to the important feature
        indices = pd.Series(data.index, index=data[imp_feature])
        idx = indices[args[imp_feature]]

        # Get pairwise similarity scores
        sig = list(enumerate(sg[idx]))

        # Sort cars based on similarity scores
        sig = sorted(sig, key=lambda x: x[1], reverse=True)

        # Get indices of top 6 similar cars
        cars_indexes = [i[0] for i in sig[:6]]

        # Recommendation for top 6 similar cars
        rec = data.iloc[cars_indexes]
        return rec
    else:
        # Sort cars based on optimal value (e.g., manufacturing year, odometer)
        sorted_cars = data.sort_values(by=['year', 'odometer'], ascending=[False, True])

        # Return the top 5 cars with the best optimal value
        best_cars = sorted_cars.head(5)
        return best_cars


