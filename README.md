# simpleSpark

> Movie Recommending System Based on Spark MLlib

### DataSets

MovieLens DataSets: https://grouplens.org/datasets/movielens/

### Process

There are three files in the DataSets:

- **Movies.dat**
- **Ratings.dat**
- **Users.dat**

The main file is **Ratings.dat**.It's format is like this:

> UserID::MovieID::RatingID::TimeStamp

We follow these steps to get the Model:

1. Preprocess the raw dataSet.
2. Training Model using ALS Algorithm.
3. Use ALS Model to get recommed movies.

### ALS Model



