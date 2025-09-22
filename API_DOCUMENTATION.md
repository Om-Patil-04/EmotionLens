# EmotionLens API Documentation 🔧

This document provides detailed information about using EmotionLens models for emotion classification tasks.

## 📋 Quick Reference

### Supported Emotion Sets

#### GoEmotions (28 Emotions)
```python
GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
```

#### TweetEval (4 Emotions)
```python
TWEETEVAL_LABELS = ['joy', 'optimism', 'anger', 'sadness']
```

## 🏗️ Model Architecture

### Configuration Classes

#### MultiTaskConfig
```python
class MultiTaskConfig:
    def __init__(self):
        self.base_model = "microsoft/deberta-v3-base"
        self.goemotions_labels = 28
        self.tweeteval_labels = 4
        self.max_length = 284
        self.hidden_dropout = 0.3
        self.attention_dropout = 0.15
        self.batch_size = 12
        self.learning_rate = 8e-6
        self.num_epochs = 8
        self.warmup_ratio = 0.15
        self.weight_decay = 0.02
        self.gradient_clip = 1.0
        self.goemotions_threshold = 0.35
        self.focal_alpha = 0.3
        self.focal_gamma = 2.5
        self.mixed_precision = True
        self.persistent_workers = True
        self.model_save_path = "./models/multitask"
        self.results_path = "./results/multitask"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 8 if torch.cuda.is_available() else 0
```

#### Single-Task Config (TweetEval)
```python
class Config:
    def __init__(self):
        self.base_model = "microsoft/deberta-v3-base"
        self.num_emotions = 4
        self.max_length = 256
        self.hidden_dropout = 0.3
        self.attention_dropout = 0.15
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 5
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.gradient_clip = 1.0
        self.model_save_path = "./models/tweeteval"
        self.results_path = "./results/tweeteval"
```

## 🧠 Model Classes

### MultiTaskModel
Advanced model supporting both GoEmotions and TweetEval classification:

```python
class MultiTaskModel(nn.Module):
    def __init__(self, config):
        super(MultiTaskModel, self).__init__()
        self.config = config
        
        # Backbone transformer
        self.backbone = AutoModel.from_pretrained(
            config.base_model,
            hidden_dropout_prob=config.hidden_dropout,
            attention_probs_dropout_prob=config.attention_dropout
        )
        
        # Attention pooling layer
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.backbone.config.hidden_size,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        self.goemotions_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.goemotions_labels)
        )
        
        self.tweeteval_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config.tweeteval_labels)
        )

    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # CLS token representation
        cls_output = hidden_states[:, 0, :]
        
        # Attention pooling
        attended_output, _ = self.attention_pooling(hidden_states, hidden_states, hidden_states)
        mean_attended = attended_output.mean(dim=1)
        
        # Combine representations
        combined = torch.cat([cls_output, mean_attended], dim=1)
        shared_repr = self.shared_layer(combined)
        
        # Task-specific predictions
        goemotions_logits = self.goemotions_head(shared_repr)
        tweeteval_logits = self.tweeteval_head(shared_repr)
        
        return goemotions_logits, tweeteval_logits
```

### Single-Task Model (GoEmotions/TweetEval)
```python
class GoEmotionsModel(nn.Module):
    def __init__(self, config):
        super(GoEmotionsModel, self).__init__()
        self.config = config
        
        self.backbone = AutoModel.from_pretrained(
            config.base_model,
            hidden_dropout_prob=config.hidden_dropout,
            attention_probs_dropout_prob=config.attention_dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout),
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.num_emotions)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
```

## 🎯 Usage Examples

### Loading a Trained Model

#### Multi-Task Model
```python
import torch
from transformers import AutoTokenizer

# Load configuration
config = MultiTaskConfig()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_save_path)

# Load model
model = MultiTaskModel(config)
checkpoint = torch.load('./models/multitask/best_model.pt', map_location=config.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

#### Single-Task Model
```python
# For TweetEval
config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.base_model)
model = GoEmotionsModel(config)
checkpoint = torch.load('./models/tweeteval/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Making Predictions

#### Single Text Prediction
```python
def predict_emotions(text, model, tokenizer, config, device):
    """
    Predict emotions for a single text input
    """
    model.eval()
    
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=config.max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        if isinstance(model, MultiTaskModel):
            goemotions_logits, tweeteval_logits = model(input_ids, attention_mask)
            
            # GoEmotions predictions (multi-label)
            goemotions_probs = torch.sigmoid(goemotions_logits).cpu().numpy()[0]
            goemotions_predictions = (goemotions_probs > config.goemotions_threshold).astype(int)
            
            # TweetEval predictions (single-label)
            tweeteval_probs = torch.softmax(tweeteval_logits, dim=1).cpu().numpy()[0]
            tweeteval_prediction = tweeteval_probs.argmax()
            
            return {
                'goemotions': {
                    'probabilities': goemotions_probs,
                    'predictions': goemotions_predictions,
                    'top_emotions': [(GOEMOTIONS_LABELS[i], goemotions_probs[i]) 
                                   for i in goemotions_probs.argsort()[-3:][::-1]]
                },
                'tweeteval': {
                    'probabilities': tweeteval_probs,
                    'prediction': tweeteval_prediction,
                    'predicted_emotion': TWEETEVAL_LABELS[tweeteval_prediction],
                    'confidence': tweeteval_probs[tweeteval_prediction]
                }
            }
        else:
            # Single-task model
            logits = model(input_ids, attention_mask)
            if config.num_emotions == 28:  # GoEmotions
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                predictions = (probs > config.threshold).astype(int)
                return {
                    'probabilities': probs,
                    'predictions': predictions,
                    'top_emotions': [(GOEMOTIONS_LABELS[i], probs[i]) 
                                   for i in probs.argsort()[-3:][::-1]]
                }
            else:  # TweetEval
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prediction = probs.argmax()
                return {
                    'probabilities': probs,
                    'prediction': prediction,
                    'predicted_emotion': TWEETEVAL_LABELS[prediction],
                    'confidence': probs[prediction]
                }

# Example usage
text = "I'm so excited about this new project!"
results = predict_emotions(text, model, tokenizer, config, device)
print(results)
```

#### Batch Prediction
```python
def predict_batch(texts, model, tokenizer, config, device, batch_size=16):
    """
    Predict emotions for multiple texts efficiently
    """
    model.eval()
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            if isinstance(model, MultiTaskModel):
                goemotions_logits, tweeteval_logits = model(input_ids, attention_mask)
                
                goemotions_probs = torch.sigmoid(goemotions_logits).cpu().numpy()
                tweeteval_probs = torch.softmax(tweeteval_logits, dim=1).cpu().numpy()
                
                for j in range(len(batch_texts)):
                    all_results.append({
                        'text': batch_texts[j],
                        'goemotions_top3': [(GOEMOTIONS_LABELS[k], goemotions_probs[j][k]) 
                                          for k in goemotions_probs[j].argsort()[-3:][::-1]],
                        'tweeteval_prediction': TWEETEVAL_LABELS[tweeteval_probs[j].argmax()],
                        'tweeteval_confidence': tweeteval_probs[j].max()
                    })
            else:
                logits = model(input_ids, attention_mask)
                if config.num_emotions == 28:
                    probs = torch.sigmoid(logits).cpu().numpy()
                    for j in range(len(batch_texts)):
                        all_results.append({
                            'text': batch_texts[j],
                            'top_emotions': [(GOEMOTIONS_LABELS[k], probs[j][k]) 
                                           for k in probs[j].argsort()[-3:][::-1]]
                        })
                else:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    for j in range(len(batch_texts)):
                        pred_idx = probs[j].argmax()
                        all_results.append({
                            'text': batch_texts[j],
                            'predicted_emotion': TWEETEVAL_LABELS[pred_idx],
                            'confidence': probs[j][pred_idx]
                        })
    
    return all_results
```

## 🔧 Custom Training

### Training a New Model
```python
# Example: Training on custom data
def train_custom_model(train_texts, train_labels, val_texts, val_labels):
    config = MultiTaskConfig()
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    train_dataset = MultiTaskDataset(train_texts, train_labels, None, tokenizer, config.max_length)
    val_dataset = MultiTaskDataset(val_texts, val_labels, None, tokenizer, config.max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize trainer
    trainer = MultiTaskTrainer(config)
    trainer.setup_model()
    
    # Train
    trainer.train((train_loader, val_loader, None), (train_loader, val_loader, None))
    
    return trainer.model
```

### Fine-tuning Existing Models
```python
def fine_tune_model(pretrained_path, new_data):
    config = MultiTaskConfig()
    trainer = MultiTaskTrainer(config)
    
    # Load pretrained model
    trainer.load_model(pretrained_path)
    
    # Prepare new data
    train_loader, val_loader = prepare_data(new_data)
    
    # Fine-tune with lower learning rate
    config.learning_rate = 1e-6
    config.num_epochs = 3
    
    trainer.train((train_loader, val_loader, None), (train_loader, val_loader, None))
    
    return trainer.model
```

## 📊 Evaluation Utilities

### Performance Metrics
```python
def evaluate_model_performance(model, test_loader, config):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            if isinstance(model, MultiTaskModel):
                goemotions_logits, tweeteval_logits = model(input_ids, attention_mask)
                # Process predictions based on task
            else:
                logits = model(input_ids, attention_mask)
                # Process single-task predictions
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'f1_macro': f1_score(all_labels, all_predictions, average='macro'),
        'f1_micro': f1_score(all_labels, all_predictions, average='micro'),
        'precision': precision_score(all_labels, all_predictions, average='macro'),
        'recall': recall_score(all_labels, all_predictions, average='macro')
    }
    
    return metrics
```

## 🎯 Advanced Features

### Attention Visualization
```python
def visualize_attention(text, model, tokenizer, config):
    """
    Extract and visualize attention weights for interpretability
    """
    model.eval()
    
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)
    
    with torch.no_grad():
        outputs = model.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_attentions=True
        )
        
        # Extract attention weights from last layer
        attention_weights = outputs.attentions[-1]
        
        # Process and visualize attention patterns
        # Implementation depends on visualization library choice
        
    return attention_weights
```

### Custom Loss Functions
```python
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy for better generalization
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
```

## 🚀 Deployment

### Model Export for Production
```python
def export_model_for_inference(model, tokenizer, save_path):
    """
    Export model and tokenizer for production deployment
    """
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config.__dict__ if hasattr(model, 'config') else {},
    }, f"{save_path}/model.pt")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    
    print(f"Model exported to {save_path}")

def load_model_for_inference(model_path, device='cpu'):
    """
    Load model for inference in production
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    checkpoint = torch.load(f"{model_path}/model.pt", map_location=device)
    
    # Reconstruct model (you'll need to determine model type)
    # This example assumes you save model type information
    if 'model_type' in checkpoint:
        if checkpoint['model_type'] == 'multitask':
            config = MultiTaskConfig()
            model = MultiTaskModel(config)
        else:
            config = Config()
            model = GoEmotionsModel(config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer
```

This API documentation provides comprehensive information for using EmotionLens models effectively. For additional examples and advanced usage, refer to the training notebooks in the repository.