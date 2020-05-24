# Tweet Classification Notebook
![Tweet Classification](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/05df8cc2-4413-4a7c-93c7-dbf7991b18a7/ddxyese-15d65e3f-fea8-4630-947c-78eb058c6821.png/v1/fill/w_1280,h_600,q_80,strp/tweet_classification_header_for_github_by_markdownimgmn_ddxyese-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOiIsImlzcyI6InVybjphcHA6Iiwib2JqIjpbW3siaGVpZ2h0IjoiPD02MDAiLCJwYXRoIjoiXC9mXC8wNWRmOGNjMi00NDEzLTRhN2MtOTNjNy1kYmY3OTkxYjE4YTdcL2RkeHllc2UtMTVkNjVlM2YtZmVhOC00NjMwLTk0N2MtNzhlYjA1OGM2ODIxLnBuZyIsIndpZHRoIjoiPD0xMjgwIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmltYWdlLm9wZXJhdGlvbnMiXX0.dWCTQ92V3NGWDm9LJubkFVubUwUg4-y6OExC_J8j8ys)

This project is one of my first projects in Natural Language Processing, specifically, document classification. Briefly, I've tokenized and vectorized the tweets and applied different traditional ML methods to classify the tweets. I've used both Count Vectorizer and TF-IDF vectorizer for vectorization, and logistic regression and SVC for classification to observe their performance.  It receives a tweet and predicts whether the sender was Justin Trudeau or Donald Trump. I will later add neural networks (probably LSTM or GRU layers), so tune in for updates.
## Models and Data Used

-   Data: Tweets of Trumps and Trudeau, given in tweets.csv
-   Classification Methods: Logistic Regression and SVM
-   Vectorizers: TF-IDF and Count Vectorizer

# Files

- *tweets.csv* including data
- *Who's_Tweeting?_Trump_v_Trudeau.ipynb* Interactive Python Notebook that includes the code itself

## Libraries Used

    sklearn
    pandas
    NLTK
## Author

-   **Merve Noyan**  - [merveenoyan](https://github.com/merveenoyan)
