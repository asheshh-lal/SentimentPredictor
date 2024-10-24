from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import googleapiclient.discovery
import pandas as pd
from tqdm import tqdm

app = Flask(__name__)

# Initialize YouTube API
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyCZ9Q_lGbaO9IG2mXoMEVQkHu4zwoP_KEI"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Load sentiment analysis model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def sentiment_engine(video_id):
    comments = []
    
    # Fetch comments
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()

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
    
    res = {}
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row['text']
            comment_id = row['comment_id']
            roberta_result = polarity_scores_roberta(text)
            both = {**roberta_result}
            res[comment_id] = both
            
        except Exception as e:
            print(f"Error for comment ID {comment_id}: {e}")
    
    results_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'comment_id'})
    results_df = results_df.merge(df, how='left', on='comment_id')

    def classify_sentiment(row):
        if row['roberta_neg'] > row['roberta_neu'] and row['roberta_neg'] > row['roberta_pos']:
            return 'negative'
        elif row['roberta_neu'] > row['roberta_neg'] and row['roberta_neu'] > row['roberta_pos']:
            return 'neutral'
        else:
            return 'positive'

    results_df['roberta_sentiment'] = results_df.apply(classify_sentiment, axis=1)

    return results_df

# Home route with form
@app.route('/')
def home():
    return render_template('index.html')

# Analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    video_id = request.form.get('video_id')  # Use get() for safe retrieval
    print(f"Video ID received: {video_id}")  # Print video ID for debugging

    if video_id:
        # Call sentiment analysis function
        results_df = sentiment_engine(video_id)
        print("Sentiment Analysis Results:")
        print(results_df.head())  # Print DataFrame to console
        return f"Analysis complete for Video ID: {video_id}!"
    else:
        return "Error: No Video ID provided!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
