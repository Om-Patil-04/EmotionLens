# EmotionLens 🎭

**EmotionLens** is a sophisticated emotion classification system that leverages transformer-based neural networks to detect and analyze emotions in text. This project implements both single-task and multi-task learning approaches for emotion recognition using state-of-the-art natural language processing techniques.

## 🚀 What Can EmotionLens Do?

### Core Capabilities

#### 1. **Multi-Dataset Emotion Classification**
- **GoEmotions Dataset**: Classifies text into 28 fine-grained emotions (admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral)
- **TweetEval Dataset**: Classifies text into 4 basic emotions (joy, optimism, anger, sadness)

#### 2. **Advanced Model Architectures**
- **Backbone Model**: Microsoft DeBERTa-v3-base for robust text understanding
- **Multi-Head Attention Pooling**: Enhanced feature extraction beyond standard CLS token
- **Multi-Task Learning**: Simultaneous training on multiple emotion classification tasks
- **Focal Loss Implementation**: Addresses class imbalance in emotion datasets
- **Label Smoothing**: Improves model generalization

#### 3. **Flexible Training Approaches**
- **Single-Task Training**: Focus on one emotion classification task at a time
- **Multi-Task Training**: Joint training on both GoEmotions and TweetEval datasets
- **Transfer Learning**: Fine-tune pre-trained models for specific emotion detection tasks
- **Mixed Precision Training**: Efficient GPU memory utilization and faster training

### 🔧 Technical Features

#### Model Architecture
- **Transformer Base**: DeBERTa-v3-base (140M parameters)
- **Attention Mechanism**: Multi-head attention pooling for better sequence representation
- **Classification Heads**: Separate heads for different emotion classification tasks
- **Dropout Regularization**: Configurable hidden and attention dropout for preventing overfitting

#### Training Optimizations
- **AdamW Optimizer**: Weight decay regularization
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Based on validation performance
- **Model Checkpointing**: Automatic saving of best performing models

#### Data Processing
- **Advanced Tokenization**: Using DeBERTa tokenizer with configurable max length
- **Text Preprocessing**: URL normalization, emoji handling, hashtag processing
- **Data Augmentation**: (Available in EDA notebook)
- **Balanced Sampling**: Handles class imbalance in datasets

## 📊 Performance Metrics

### GoEmotions (28 Emotions)
- **Accuracy**: 28.6%
- **F1-Score (Macro)**: 49.9%
- **F1-Score (Micro)**: 57.2%
- **Precision (Macro)**: 42.5%
- **Recall (Macro)**: 65.3%

### TweetEval (4 Emotions)
- **Accuracy**: 83.4%
- **F1-Score (Macro)**: 77.2%
- **F1-Score (Weighted)**: 83.1%
- **Precision (Macro)**: 80.4%
- **Recall (Macro)**: 75.7%

### Multi-Task Learning Results
- **Combined Training**: Achieves competitive performance on both tasks simultaneously
- **Knowledge Transfer**: Benefits from shared representations between emotion tasks
- **Efficiency**: Single model handles multiple emotion classification scenarios

## 🛠️ What You Can Build With EmotionLens

### 1. **Emotion-Aware Applications**
- **Social Media Monitoring**: Analyze sentiment and emotions in social media posts
- **Customer Service**: Detect frustration, satisfaction, or confusion in customer interactions
- **Content Moderation**: Identify potentially harmful emotional content
- **Mental Health Applications**: Monitor emotional patterns in text communications

### 2. **Research Applications**
- **Emotion Analysis Studies**: Comprehensive emotion detection for research
- **Comparative Studies**: Compare performance across different emotion taxonomies
- **Model Development**: Base framework for developing custom emotion classifiers
- **Dataset Analysis**: EDA tools for understanding emotion distribution patterns

### 3. **Real-Time Inference**
- **API Integration**: Deploy models for real-time emotion prediction
- **Batch Processing**: Analyze large text corpora for emotion patterns
- **Multi-Language Support**: Extend to other languages (with appropriate datasets)
- **Custom Emotion Sets**: Train on domain-specific emotion taxonomies

## 📁 Project Structure

```
EmotionLens/
├── Notebooks/Training/
│   ├── 1_GoEmotions_Baseline_Training.ipynb    # Single-task GoEmotions training
│   ├── 2_TweetEval_EDA.ipynb                   # Exploratory data analysis
│   ├── 3_TweetEval_FineTuning.ipynb            # Single-task TweetEval training
│   └── 4_multi_task_trainer.ipynb              # Multi-task training implementation
├── Results/
│   ├── GoEmotions/                             # GoEmotions training results
│   ├── Tweet-Eval/                             # TweetEval training results
│   └── Multi-Task/                             # Multi-task training results
├── Datasets/                                   # Dataset storage and processing
├── Tokens/                                     # Tokenizer configurations
└── README.md                                   # This documentation
```

## 🔬 Key Components

### Models
1. **GoEmotionsModel**: Single-task model for 28-emotion classification
2. **MultiTaskModel**: Joint model for both emotion classification tasks
3. **FocalLoss**: Custom loss function for handling class imbalance
4. **LabelSmoothingCrossEntropy**: Regularized loss for better generalization

### Trainers
1. **TweetEvalTrainer**: Specialized trainer for 4-emotion classification
2. **MultiTaskTrainer**: Advanced trainer for simultaneous multi-task learning
3. **Evaluation Methods**: Comprehensive metrics calculation and validation

### Data Handling
1. **TweetEvalDataset**: Custom dataset class for TweetEval data
2. **MultiTaskDataset**: Combined dataset for multi-task learning
3. **Data Preprocessing**: Text cleaning and normalization utilities

## 🎯 Use Cases

### Business Applications
- **Brand Monitoring**: Track emotional responses to products/services
- **Market Research**: Understand customer emotional reactions
- **HR Analytics**: Analyze employee feedback and satisfaction
- **Content Strategy**: Optimize content for desired emotional impact

### Academic Research
- **Computational Linguistics**: Study emotion expression patterns
- **Psychology Research**: Analyze emotional language in different contexts
- **Cross-Cultural Studies**: Compare emotion expression across communities
- **Temporal Analysis**: Track emotion changes over time

### Technical Integration
- **Chatbots**: Emotion-aware conversational systems
- **Recommendation Systems**: Emotion-based content recommendations
- **Search Enhancement**: Emotion-aware search and filtering
- **Content Classification**: Automatic emotional tagging of content

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Transformers library
- CUDA-capable GPU (recommended)

### Quick Start
1. **Clone the repository**
2. **Install dependencies** (requirements not provided, but based on notebooks)
3. **Run the notebooks** in the Training folder
4. **Start with single-task training** (notebooks 1 or 3)
5. **Advance to multi-task learning** (notebook 4)

### Training Your Own Model
1. **Single-Task**: Use notebooks 1 or 3 for focused emotion classification
2. **Multi-Task**: Use notebook 4 for joint training on multiple datasets
3. **Custom Data**: Adapt the dataset classes for your own emotion data
4. **Fine-Tuning**: Start from pre-trained checkpoints for faster convergence

## 🔮 Future Capabilities

EmotionLens provides a robust foundation that can be extended for:
- **Multilingual Emotion Detection**: Support for multiple languages
- **Real-Time Streaming**: Process live text streams
- **Ensemble Methods**: Combine multiple models for better performance
- **Domain Adaptation**: Specialize for specific domains (medical, legal, etc.)
- **Explainable AI**: Understand which words/phrases drive emotion predictions
- **Continuous Learning**: Update models with new emotion data over time

## 📈 Model Performance Comparison

| Model Type | Dataset | Accuracy | F1-Macro | F1-Micro | Training Time |
|------------|---------|----------|----------|----------|---------------|
| Single-Task | GoEmotions | 28.6% | 49.9% | 57.2% | ~2-3 hours |
| Single-Task | TweetEval | 83.4% | 77.2% | 83.1% | ~30 minutes |
| Multi-Task | Both | Competitive | Balanced | Efficient | ~3-4 hours |

*Performance metrics may vary based on hardware and training configuration*

---

**EmotionLens** represents a comprehensive approach to emotion classification, combining state-of-the-art transformer models with advanced training techniques to deliver robust, accurate emotion detection capabilities for a wide range of applications.