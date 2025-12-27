# 1. [Drug-related-tweets:](https://www.kaggle.com/datasets/technocare/drug-related-tweets-dataset)

This dataset contains `53,766` drug-related text entries structured to resemble tweets. It was generated from the drugsComTest_raw.csv dataset, which originally included patient reviews of medications. Source: Extracted from patient-submitted reviews on Drugs.com. Format: CSV file with two columns: drugName – the name of the drug mentioned. tweet – the review text reformatted to simulate a tweet-like message.

### Features:
- `drugName`
- `tweet`


# 2. [Unique_medical_tweets_dataset:](https://www.kaggle.com/datasets/siddhantrajhans/10000-unique-medical-tweets)
This dataset contains `10,000` tweets related to medical conditions, including diabetes, cancer, and mental health. The tweets were collected from Twitter using a combination of hashtags and keywords, and were then labeled as either relevant or not relevant to a medical condition. The dataset is intended for use in natural language processing (NLP) research.

### Features:
- `id`: A unique identifier for each tweet
- `keyword`: A particular keyword from the tweet (e.g. disease, symptom, medication)
- `location`: The location the tweet was sent from (may be blank)
- `text`: The text of the tweet
- `target`: Denotes whether a tweet is about a real medical condition (1) or not (0)


# 3. [1.6-million-tweets-dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
This is the sentiment140 dataset. It contains `1,600,000` tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

### Features:
- `target`: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- `ids`: The id of the tweet ( 2087)
- `date`: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- `flag`: The query (lyx). If there is no query, then this value is NO_QUERY.
- `user`: the user that tweeted (robotickilldozr)
- `text`: the text of the tweet (Lyx is cool)

# 4. Final-dataset-cleaned:
The dataset was constructed by combining drug-related textual data with
medical and general social media tweets. Drug-related texts were labeled
as positive samples, while medical and general tweets without substance-
related content were labeled as negative samples. Standard NLP
preprocessing was applied to clean the text.

### Features:
- `label`: (1 : drug abuse positive, 0 : not positive)
- `text`: cleaned tweets