import csv
from pathlib import Path

def output_to_spreadsheet(elapsed_time,model_dimension, hidden_layer_dimension, num_heads, key_dimension, value_dimension, num_encoder_layers, batch_size, output_filename, results):

    row = [
        str(elapsed_time),
        str(model_dimension),
        str(hidden_layer_dimension),
        str(num_heads),
        str(key_dimension),
        str(value_dimension),
        str(num_encoder_layers),
        str(batch_size),
        results['train_loss'],
        results['val_loss'],
        results['metrics']['accuracy'],
        results['metrics']['macro_precision'],
        results['metrics']['macro_recall'],
        results['metrics']['macro_f1'],
        results['metrics']['weighted_f1'],
        results['metrics']['weighted_precision'],
        results['metrics']['weighted_recall']
    ]

    row.extend(results['metrics']['per_class']['precision'])
    row.extend(results['metrics']['per_class']['recall'])
    row.extend(results['metrics']['per_class']['f1'])
    row.extend(results['metrics']['per_class']['support'])

    header = [
        'elapsed time',
        'model',
        'hidden_layer',
        'heads',
        'key',
        'value',
        'encoder',
        'batch',
        'train_loss',
        'val_loss',
        'accuracy',
        'macro_precision',
        'macro_recall',
        'macro_f1',
        'weighted_f1',
        'weighted_precision',
        'weighted_recall'
    ]

    num_classes = len(results['metrics']['per_class']['precision'])
    for i in range(num_classes):
        header.append(f'precision_class_{i}')
    for i in range(num_classes):
        header.append(f'recall_class_{i}')
    for i in range(num_classes):
        header.append(f'f1_class_{i}')
    for i in range(num_classes):
        header.append(f'support_class_{i}')

    write_header = not Path(output_filename).is_file()
    write_mode = 'a'
    with open(output_filename, write_mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header: writer.writerow(header)
        writer.writerow(row)