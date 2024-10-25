from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import googleapiclient.discovery
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import os

app = Flask(__name__)

# Initialize YouTube API
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyCZ9Q_lGbaO9IG2mXoMEVQkHu4zwoP_KEI"  # Replace with your actual YouTube API key

youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Load sentiment analysis model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Download the stopwords list
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def sentiment_engine(video_id):
    comments = []

    # Fetch comments
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    
    try:
        response = request.execute()

        # Process comments
        while response:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment.get('updatedAt', comment['publishedAt']),
                    comment['likeCount'],
                    comment['textDisplay'],
                    item['id']
                ])

            # Check for next page of comments
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=response['nextPageToken']
                )
                response = request.execute()
            else:
                break

        # Create DataFrame
        df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text', 'comment_id'])

        # Sentiment analysis function
        def polarity_scores_roberta(text):
            encoded_text = tokenizer(text, return_tensors='pt')
            output = model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            scores_dict = {
                'roberta_neg': scores[0],
                'roberta_neu': scores[1],
                'roberta_pos': scores[2]
            }
            return scores_dict

        # Add sentiment scores to each comment
        res = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                text = row['text']
                comment_id = row['comment_id']
                roberta_result = polarity_scores_roberta(text)
                res[comment_id] = roberta_result
            except Exception as e:
                print(f"Error for comment ID {comment_id}: {e}")

        # Create results DataFrame
        results_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'comment_id'})
        results_df = results_df.merge(df, how='left', on='comment_id')

        # Classify sentiment
        def classify_sentiment(row):
            if row['roberta_neg'] > row['roberta_neu'] and row['roberta_neg'] > row['roberta_pos']:
                return 'negative'
            elif row['roberta_neu'] > row['roberta_neg'] and row['roberta_neu'] > row['roberta_pos']:
                return 'neutral'
            else:
                return 'positive'

        results_df['roberta_sentiment'] = results_df.apply(classify_sentiment, axis=1)

        return results_df  # Return only the DataFrame

    except Exception as e:
        print(f"Error in sentiment_engine: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if an error occurs

def analyze_word_frequency(df, top_n=10):
    """Analyze and return the top N most common words excluding stop words."""
    df = df.dropna(subset=['text'])
    all_text = ' '.join(df['text'])
    
    # Split the text into words, convert to lower case, and filter out stop words
    words = [word.lower() for word in all_text.split() if word.lower() not in stop_words]
    word_counts = Counter(words)
    
    # Get the top N most common words
    top_words = word_counts.most_common(top_n)

    return top_words

def create_pie_chart(sentiment_distribution):
    """Create a pie chart for sentiment distribution and save it as an image."""
    labels = sentiment_distribution.keys()
    sizes = sentiment_distribution.values()
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
    
    # Save the pie chart as an image file
    chart_path = 'static/sentiment_distribution.png'  # Save to a static folder
    plt.savefig(chart_path)
    plt.close()  # Close the plot to avoid displaying it inline
    return chart_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_id = request.form.get('video_id')
    print(f"Received Video ID: {video_id}")  # Check the received Video ID

    if video_id:
        # Call sentiment analysis function
        results_df = sentiment_engine(video_id)
        
        # Perform word frequency analysis
        top_words = analyze_word_frequency(results_df, top_n=10)
        
        # Sentiment distribution data
        sentiment_distribution = results_df['roberta_sentiment'].value_counts().to_dict()

        # Create and save the pie chart
        pie_chart_path = create_pie_chart(sentiment_distribution)

        # Prepare the response
        response_data = {
            'video_id': video_id,
            'top_words': top_words,
            'sentiment_distribution': sentiment_distribution,
            'comments_count': results_df.shape[0],
            'pie_chart_path': pie_chart_path
        }

        return render_template('results.html', response_data=response_data)
    else:
        return "Error: No Video ID provided!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode
