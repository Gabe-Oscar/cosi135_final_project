from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from time import time

class StopWatch():
    def __init__(self):
        self.start_time = time()
        
    def get_elapsed_time(self):
        return time() - self.start_time
    def print_elapsed_time(self):
        elapsed_time = self.get_elapsed_time()
        mins = elapsed_time // 60
        secs = int(elapsed_time - (mins * 60))
        print(f"Time elapsed: {mins}:{secs:02d} :D")
     

def generate_metrics(epoch, avg_train_loss, val_loss, labels, predictions, do_print = True):
    metrics = calculate_metrics(labels, predictions)
    epoch_results = {
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'metrics': metrics
    }
    if do_print:
        print_epoch_results(epoch_results)
    return epoch_results

    
def calculate_metrics(true_labels, predictions):
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Calculate precision, recall, and f1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        zero_division=0
    )
    
    # Calculate macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        average='macro', 
        zero_division=0
    )
    
    # Calculate weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        average='weighted', 
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    }

def print_epoch_results(results):
    """
    Print formatted training results for an epoch
    """
    print("\nEpoch Results:")
    print(f"Training Loss: {results['train_loss']:.4f}")
    print(f"Validation Loss: {results['val_loss']:.4f}")
    
    metrics = results['metrics']
    print(f"\nValidation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    print("\nPer-class metrics:")
    per_class = metrics['per_class']
    for i in range(len(per_class['precision'])):
        print(f"\nLabel {i}:")
        print(f"  Precision: {per_class['precision'][i]:.4f}")
        print(f"  Recall: {per_class['recall'][i]:.4f}")
        print(f"  F1-score: {per_class['f1'][i]:.4f}")
        print(f"  Support: {per_class['support'][i]}")
    
    print("\n" + "="*50)


