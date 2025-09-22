# EmotionLens Usage Examples 💡

This document provides practical examples of how to use EmotionLens for various emotion classification tasks.

## 🚀 Quick Start Examples

### Example 1: Basic Emotion Detection
```python
# Sample texts for emotion detection
test_texts = [
    "I'm so excited about this new opportunity!",
    "This is really frustrating and annoying.",
    "What a beautiful day it is today!",
    "I can't believe this happened to me.",
    "I'm feeling a bit nervous about the presentation.",
    "Thank you so much for your help!",
    "I'm really disappointed with the results.",
    "I'm worried about the future."
]

# Expected output format (from the training notebooks):
"""
Text 1: I'm so excited about this new opportunity!
------------------------------------------------------------
GoEmotions FINAL EMOTION: excitement (confidence: 0.8234)
GoEmotions (Multi-label) - Top 3:
  excitement: 0.8234
  joy: 0.4521
  optimism: 0.3987

TweetEval FINAL EMOTION: joy (confidence: 0.7643)
All TweetEval probabilities:
  joy: 0.7643
  optimism: 0.1234
  anger: 0.0567
  sadness: 0.0556
"""
```

### Example 2: Batch Processing for Social Media Analysis
```python
# Example: Analyzing customer feedback
customer_feedback = [
    "Love this product! Amazing quality and fast shipping.",
    "The customer service was terrible. Very disappointed.",
    "Good value for money, but could be better.",
    "Excellent experience! Will definitely buy again.",
    "Product arrived damaged. Waiting for replacement.",
    "Not what I expected. Returning this item.",
    "Outstanding quality! Exceeded my expectations.",
    "Decent product but shipping took too long."
]

# Process in batches for efficiency
def analyze_customer_sentiment(feedback_list, model, tokenizer, config):
    """
    Analyze customer feedback for emotion patterns
    """
    results = predict_batch(feedback_list, model, tokenizer, config, device='cuda')
    
    # Aggregate results
    emotion_summary = {
        'positive_emotions': ['joy', 'excitement', 'gratitude', 'love', 'admiration'],
        'negative_emotions': ['anger', 'disappointment', 'sadness', 'disgust', 'annoyance'],
        'neutral_emotions': ['neutral', 'realization']
    }
    
    sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
    detailed_emotions = {}
    
    for result in results:
        # Analyze TweetEval results for basic sentiment
        emotion = result['tweeteval_prediction']
        if emotion in emotion_summary['positive_emotions']:
            sentiment_distribution['positive'] += 1
        elif emotion in emotion_summary['negative_emotions']:
            sentiment_distribution['negative'] += 1
        else:
            sentiment_distribution['neutral'] += 1
        
        # Count detailed GoEmotions
        for emotion_name, score in result['goemotions_top3']:
            if emotion_name not in detailed_emotions:
                detailed_emotions[emotion_name] = 0
            detailed_emotions[emotion_name] += score
    
    return {
        'sentiment_distribution': sentiment_distribution,
        'detailed_emotions': detailed_emotions,
        'individual_results': results
    }
```

### Example 3: Real-Time Chat Emotion Monitoring
```python
class EmotionMonitor:
    """
    Real-time emotion monitoring for chat applications
    """
    def __init__(self, model_path):
        self.model, self.tokenizer = self.load_model(model_path)
        self.config = MultiTaskConfig()
        self.emotion_history = []
    
    def analyze_message(self, message, user_id=None, timestamp=None):
        """
        Analyze a single chat message for emotions
        """
        result = predict_emotions(message, self.model, self.tokenizer, self.config, self.config.device)
        
        # Store in history
        analysis = {
            'message': message,
            'user_id': user_id,
            'timestamp': timestamp or datetime.now(),
            'emotions': result
        }
        
        self.emotion_history.append(analysis)
        
        # Check for concerning emotions
        concerning_emotions = ['anger', 'sadness', 'fear', 'grief', 'remorse']
        alerts = []
        
        for emotion, score in result['goemotions']['top_emotions']:
            if emotion in concerning_emotions and score > 0.7:
                alerts.append({
                    'type': 'high_negative_emotion',
                    'emotion': emotion,
                    'confidence': score,
                    'message': message
                })
        
        return {
            'analysis': analysis,
            'alerts': alerts
        }
    
    def get_user_emotion_trends(self, user_id, days=7):
        """
        Analyze emotion trends for a specific user
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        user_messages = [
            entry for entry in self.emotion_history 
            if entry['user_id'] == user_id and entry['timestamp'] >= cutoff_date
        ]
        
        if not user_messages:
            return None
        
        # Calculate emotion trends
        emotion_totals = {}
        for entry in user_messages:
            for emotion, score in entry['emotions']['goemotions']['top_emotions']:
                if emotion not in emotion_totals:
                    emotion_totals[emotion] = []
                emotion_totals[emotion].append(score)
        
        # Calculate averages
        emotion_averages = {
            emotion: sum(scores) / len(scores) 
            for emotion, scores in emotion_totals.items()
        }
        
        return {
            'user_id': user_id,
            'period_days': days,
            'message_count': len(user_messages),
            'dominant_emotions': sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)[:5],
            'emotion_timeline': user_messages
        }

# Usage example
monitor = EmotionMonitor('./models/multitask/best_model.pt')

# Analyze individual messages
result = monitor.analyze_message("I'm feeling really stressed about work lately", user_id="user123")
print(f"Alerts: {result['alerts']}")

# Get user trends
trends = monitor.get_user_emotion_trends("user123", days=30)
print(f"User's dominant emotions: {trends['dominant_emotions']}")
```

### Example 4: Content Recommendation Based on Emotions
```python
class EmotionBasedRecommender:
    """
    Recommend content based on detected emotions
    """
    def __init__(self, model_path):
        self.model, self.tokenizer = self.load_model(model_path)
        self.config = MultiTaskConfig()
        
        # Define content recommendations for different emotions
        self.emotion_recommendations = {
            'sadness': ['uplifting_music', 'comedy_videos', 'motivational_content'],
            'anger': ['meditation_guides', 'breathing_exercises', 'calming_music'],
            'fear': ['reassuring_content', 'educational_resources', 'community_support'],
            'excitement': ['adventure_content', 'celebration_ideas', 'sharing_platforms'],
            'love': ['romantic_content', 'gift_ideas', 'relationship_advice'],
            'gratitude': ['thankfulness_exercises', 'giving_opportunities', 'positive_stories'],
            'curiosity': ['educational_content', 'documentaries', 'how_to_guides'],
            'joy': ['celebration_content', 'sharing_platforms', 'happy_memories']
        }
    
    def recommend_content(self, user_input):
        """
        Recommend content based on detected emotions
        """
        emotions = predict_emotions(user_input, self.model, self.tokenizer, self.config, self.config.device)
        
        recommendations = []
        
        # Get top emotions from GoEmotions
        for emotion, confidence in emotions['goemotions']['top_emotions'][:3]:
            if emotion in self.emotion_recommendations and confidence > 0.3:
                for content_type in self.emotion_recommendations[emotion]:
                    recommendations.append({
                        'content_type': content_type,
                        'reason': f"Based on detected {emotion} (confidence: {confidence:.2f})",
                        'priority': confidence
                    })
        
        # Sort by priority and remove duplicates
        recommendations = sorted(recommendations, key=lambda x: x['priority'], reverse=True)
        unique_recommendations = []
        seen_content = set()
        
        for rec in recommendations:
            if rec['content_type'] not in seen_content:
                unique_recommendations.append(rec)
                seen_content.add(rec['content_type'])
        
        return {
            'user_input': user_input,
            'detected_emotions': emotions,
            'recommendations': unique_recommendations[:5]  # Top 5 recommendations
        }

# Usage example
recommender = EmotionBasedRecommender('./models/multitask/best_model.pt')

user_text = "I'm feeling really down after receiving some bad news today"
recommendations = recommender.recommend_content(user_text)

print("Detected emotions:", recommendations['detected_emotions']['goemotions']['top_emotions'])
print("Recommendations:")
for rec in recommendations['recommendations']:
    print(f"- {rec['content_type']}: {rec['reason']}")
```

### Example 5: Mental Health Monitoring Dashboard
```python
class MentalHealthMonitor:
    """
    Monitor mental health indicators through text analysis
    """
    def __init__(self, model_path):
        self.model, self.tokenizer = self.load_model(model_path)
        self.config = MultiTaskConfig()
        
        # Define emotional health indicators
        self.positive_indicators = ['joy', 'excitement', 'gratitude', 'love', 'pride', 'relief']
        self.negative_indicators = ['sadness', 'fear', 'anger', 'disgust', 'grief', 'remorse']
        self.stress_indicators = ['nervousness', 'fear', 'disappointment', 'annoyance']
        self.social_indicators = ['love', 'caring', 'gratitude', 'admiration']
    
    def analyze_mental_state(self, texts, time_period='week'):
        """
        Analyze mental state from a collection of texts
        """
        all_emotions = []
        individual_analyses = []
        
        for text in texts:
            result = predict_emotions(text, self.model, self.tokenizer, self.config, self.config.device)
            individual_analyses.append({
                'text': text,
                'emotions': result['goemotions']['top_emotions']
            })
            
            # Collect all emotions with their scores
            for emotion, score in result['goemotions']['top_emotions']:
                all_emotions.append((emotion, score))
        
        # Calculate indicators
        positive_score = sum(score for emotion, score in all_emotions if emotion in self.positive_indicators)
        negative_score = sum(score for emotion, score in all_emotions if emotion in self.negative_indicators)
        stress_score = sum(score for emotion, score in all_emotions if emotion in self.stress_indicators)
        social_score = sum(score for emotion, score in all_emotions if emotion in self.social_indicators)
        
        total_score = positive_score + negative_score + stress_score + social_score
        
        # Normalize scores
        if total_score > 0:
            positive_ratio = positive_score / total_score
            negative_ratio = negative_score / total_score
            stress_ratio = stress_score / total_score
            social_ratio = social_score / total_score
        else:
            positive_ratio = negative_ratio = stress_ratio = social_ratio = 0
        
        # Generate mental health insights
        insights = []
        
        if positive_ratio > 0.6:
            insights.append("Strong positive emotional state detected")
        elif positive_ratio < 0.2:
            insights.append("Low positive emotions - consider engaging in enjoyable activities")
        
        if stress_ratio > 0.4:
            insights.append("High stress indicators - consider stress management techniques")
        
        if social_ratio < 0.1:
            insights.append("Low social connection indicators - consider reaching out to friends/family")
        
        if negative_ratio > 0.5:
            insights.append("High negative emotions detected - consider professional support if persistent")
        
        return {
            'analysis_period': time_period,
            'text_count': len(texts),
            'emotional_balance': {
                'positive': positive_ratio,
                'negative': negative_ratio,
                'stress': stress_ratio,
                'social': social_ratio
            },
            'insights': insights,
            'individual_analyses': individual_analyses,
            'overall_wellbeing_score': max(0, min(100, (positive_ratio - negative_ratio + social_ratio - stress_ratio) * 50 + 50))
        }

# Usage example
monitor = MentalHealthMonitor('./models/multitask/best_model.pt')

# Sample diary entries or journal texts
diary_entries = [
    "Had a great day at work today, feeling accomplished",
    "Worried about the upcoming presentation tomorrow",
    "Spent time with family, felt really grateful",
    "Feeling overwhelmed with all the deadlines",
    "Enjoyed a peaceful walk in the park this evening"
]

mental_state = monitor.analyze_mental_state(diary_entries, 'week')

print(f"Overall Wellbeing Score: {mental_state['overall_wellbeing_score']:.1f}/100")
print(f"Emotional Balance:")
print(f"  Positive: {mental_state['emotional_balance']['positive']:.2%}")
print(f"  Negative: {mental_state['emotional_balance']['negative']:.2%}")
print(f"  Stress: {mental_state['emotional_balance']['stress']:.2%}")
print(f"  Social: {mental_state['emotional_balance']['social']:.2%}")

print("Insights:")
for insight in mental_state['insights']:
    print(f"  - {insight}")
```

### Example 6: Educational Content Analysis
```python
class EducationalEmotionAnalyzer:
    """
    Analyze emotions in educational content and student feedback
    """
    def __init__(self, model_path):
        self.model, self.tokenizer = self.load_model(model_path)
        self.config = MultiTaskConfig()
        
        # Define learning-related emotions
        self.learning_positive = ['curiosity', 'excitement', 'pride', 'joy', 'gratitude']
        self.learning_negative = ['confusion', 'frustration', 'disappointment', 'fear']
        self.engagement_indicators = ['curiosity', 'excitement', 'admiration', 'surprise']
    
    def analyze_student_feedback(self, feedback_texts, course_id=None):
        """
        Analyze student feedback for course improvement
        """
        analyses = []
        emotion_summary = {}
        
        for feedback in feedback_texts:
            result = predict_emotions(feedback, self.model, self.tokenizer, self.config, self.config.device)
            
            analysis = {
                'feedback': feedback,
                'emotions': result['goemotions']['top_emotions'],
                'engagement_score': 0,
                'learning_satisfaction': 0
            }
            
            # Calculate engagement and satisfaction scores
            for emotion, score in result['goemotions']['top_emotions']:
                if emotion in self.engagement_indicators:
                    analysis['engagement_score'] += score
                
                if emotion in self.learning_positive:
                    analysis['learning_satisfaction'] += score
                elif emotion in self.learning_negative:
                    analysis['learning_satisfaction'] -= score
                
                # Update emotion summary
                if emotion not in emotion_summary:
                    emotion_summary[emotion] = []
                emotion_summary[emotion].append(score)
            
            analyses.append(analysis)
        
        # Calculate course-level metrics
        avg_engagement = sum(a['engagement_score'] for a in analyses) / len(analyses)
        avg_satisfaction = sum(a['learning_satisfaction'] for a in analyses) / len(analyses)
        
        # Identify top concerns and highlights
        emotion_averages = {
            emotion: sum(scores) / len(scores)
            for emotion, scores in emotion_summary.items()
        }
        
        top_positive = [(e, s) for e, s in emotion_averages.items() if e in self.learning_positive]
        top_negative = [(e, s) for e, s in emotion_averages.items() if e in self.learning_negative]
        
        top_positive.sort(key=lambda x: x[1], reverse=True)
        top_negative.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        
        if avg_engagement < 0.3:
            recommendations.append("Consider adding more interactive elements to increase engagement")
        
        if any(emotion == 'confusion' and score > 0.4 for emotion, score in top_negative[:3]):
            recommendations.append("Students show confusion - consider clarifying complex concepts")
        
        if avg_satisfaction < 0:
            recommendations.append("Overall satisfaction is low - review course content and delivery")
        
        return {
            'course_id': course_id,
            'feedback_count': len(feedback_texts),
            'average_engagement': avg_engagement,
            'average_satisfaction': avg_satisfaction,
            'top_positive_emotions': top_positive[:5],
            'top_concerns': top_negative[:5],
            'recommendations': recommendations,
            'individual_analyses': analyses
        }

# Usage example
analyzer = EducationalEmotionAnalyzer('./models/multitask/best_model.pt')

student_feedback = [
    "This course was amazing! I learned so much and the instructor was very clear.",
    "I found the material quite confusing, especially the advanced topics.",
    "Great examples and practical applications. Really enjoyed the assignments.",
    "The pace was too fast for me. I struggled to keep up with the content.",
    "Excellent course! I feel much more confident in this subject now."
]

course_analysis = analyzer.analyze_student_feedback(student_feedback, course_id="CS101")

print(f"Course Analysis for {course_analysis['course_id']}:")
print(f"Average Engagement: {course_analysis['average_engagement']:.2f}")
print(f"Average Satisfaction: {course_analysis['average_satisfaction']:.2f}")

print("\nTop Positive Emotions:")
for emotion, score in course_analysis['top_positive_emotions']:
    print(f"  {emotion}: {score:.3f}")

print("\nRecommendations:")
for rec in course_analysis['recommendations']:
    print(f"  - {rec}")
```

## 🔧 Advanced Usage Patterns

### Custom Emotion Threshold Tuning
```python
def tune_emotion_thresholds(model, validation_data, initial_threshold=0.35):
    """
    Tune emotion detection thresholds for optimal performance
    """
    thresholds = np.arange(0.1, 0.8, 0.05)
    best_threshold = initial_threshold
    best_f1 = 0
    
    for threshold in thresholds:
        predictions = []
        true_labels = []
        
        for text, labels in validation_data:
            result = predict_emotions(text, model, tokenizer, config, device)
            pred = (np.array(result['goemotions']['probabilities']) > threshold).astype(int)
            predictions.append(pred)
            true_labels.append(labels)
        
        f1 = f1_score(true_labels, predictions, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

# Usage
# best_thresh, best_score = tune_emotion_thresholds(model, validation_set)
# config.goemotions_threshold = best_thresh
```

### Emotion Trend Analysis
```python
def analyze_emotion_trends(texts_with_timestamps):
    """
    Analyze how emotions change over time
    """
    results = []
    
    for timestamp, text in texts_with_timestamps:
        emotion_result = predict_emotions(text, model, tokenizer, config, device)
        results.append({
            'timestamp': timestamp,
            'text': text,
            'emotions': emotion_result['goemotions']['top_emotions']
        })
    
    # Group by time periods and analyze trends
    import pandas as pd
    
    df = pd.DataFrame(results)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Calculate daily emotion averages
    daily_emotions = {}
    for date in df['date'].unique():
        daily_data = df[df['date'] == date]
        emotion_totals = {}
        
        for _, row in daily_data.iterrows():
            for emotion, score in row['emotions']:
                if emotion not in emotion_totals:
                    emotion_totals[emotion] = []
                emotion_totals[emotion].append(score)
        
        daily_emotions[date] = {
            emotion: sum(scores) / len(scores)
            for emotion, scores in emotion_totals.items()
        }
    
    return daily_emotions

# Usage for tracking mood over time
# emotion_trends = analyze_emotion_trends(journal_entries_with_dates)
```

These examples demonstrate the versatility of EmotionLens for various applications, from simple emotion detection to complex analytical systems. The models can be adapted and extended for specific use cases while maintaining the core emotion classification capabilities.