"""
Toxic Comment Prediction Script
================================
Use the trained model to classify new comments as toxic or non-toxic.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse


class ToxicCommentClassifier:
    """Classifier for predicting toxicity of comments."""
    
    def __init__(self, model_path='toxic_classifier_model'):
        """
        Initialize the classifier with a trained model.
        
        Args:
            model_path: Path to the saved model directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully (Device: {self.device})")
        
        self.labels = ['Non-Toxic', 'Toxic']
    
    def predict(self, text, return_proba=True):
        """
        Predict toxicity of a single comment.
        
        Args:
            text: The comment text to classify
            return_proba: Whether to return probability scores
            
        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
        
        result = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': self.labels[prediction],
            'is_toxic': prediction == 1
        }
        
        if return_proba:
            result['confidence'] = probabilities[0][prediction].item()
            result['toxic_probability'] = probabilities[0][1].item()
            result['non_toxic_probability'] = probabilities[0][0].item()
        
        return result
    
    def predict_batch(self, texts, return_proba=True):
        """
        Predict toxicity for a batch of comments.
        
        Args:
            texts: List of comment texts
            return_proba: Whether to return probability scores
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text, return_proba) for text in texts]
    
    def classify(self, text):
        """
        Simple classification returning just the label.
        
        Args:
            text: The comment text to classify
            
        Returns:
            'Toxic' or 'Non-Toxic'
        """
        result = self.predict(text, return_proba=False)
        return result['prediction']


def print_prediction(result):
    """Pretty print a prediction result."""
    print("\n" + "─" * 50)
    print(f"Text: {result['text']}")
    print(f"─" * 50)
    
    if result['is_toxic']:
        print(f"🔴 Prediction: {result['prediction']}")
    else:
        print(f"🟢 Prediction: {result['prediction']}")
    
    if 'confidence' in result:
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        print(f"   Toxic Probability: {result['toxic_probability']*100:.1f}%")


def interactive_mode(classifier):
    """Run the classifier in interactive mode."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE")
    print("  Type a comment to classify (or 'quit' to exit)")
    print("=" * 60)
    
    while True:
        print()
        text = input("Enter comment: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter a comment.")
            continue
        
        result = classifier.predict(text)
        print_prediction(result)


def demo_mode(classifier):
    """Run demo with sample comments."""
    print("\n" + "=" * 60)
    print("  DEMO MODE - Sample Classifications")
    print("=" * 60)
    
    sample_comments = [
        "This is a great article, thanks for sharing!",
        "You're an idiot and should shut up.",
        "I respectfully disagree with your opinion on this matter.",
        "What a bunch of losers, nobody cares about your stupid ideas.",
        "Could you please provide more details about this topic?",
        "Go to hell you moron!",
        "I appreciate the effort you put into this research.",
        "This is the dumbest thing I've ever read.",
    ]
    
    for comment in sample_comments:
        result = classifier.predict(comment)
        print_prediction(result)
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Toxic Comment Classifier - Make predictions with trained model'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='toxic_classifier_model',
        help='Path to the trained model directory'
    )
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Single comment to classify'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demo with sample comments'
    )
    
    args = parser.parse_args()
    
    # Load classifier
    classifier = ToxicCommentClassifier(args.model)
    
    if args.text:
        # Single prediction
        result = classifier.predict(args.text)
        print_prediction(result)
    elif args.interactive:
        # Interactive mode
        interactive_mode(classifier)
    elif args.demo:
        # Demo mode
        demo_mode(classifier)
    else:
        # Default to demo mode
        demo_mode(classifier)
        print("\nTip: Use --interactive for interactive mode or --text 'your comment' for single predictions")


if __name__ == "__main__":
    main()
