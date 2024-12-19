Welcome to my final project for COSI 135!

A model can be trained by calling training.py and specifying the following parameters (some of which are optional):
    parser.add_argument('--model_dimension', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--key_dimension', type=int, default=256) 
    parser.add_argument('--value_dimension', type=int, default=256)
    parser.add_argument('--hidden_dimension', type=int, default=512)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument("--num_labels", type=int, default=9)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_count', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    parser.add_argument('--dataset_name', type=str, default='tner/conll2003')
    parser.add_argument('--max_seq_len', type=int, default=None)

    # Model
    parser.add_argument('--model_type', type=str, choices=['transformer', 'word2vec'], default='transformer')
    
    # Output
    parser.add_argument("--model_results_path", default="")
    parser.add_argument("--model_path", default="")

Additionally, the run_experiments file can be modified to train whichever configurations you would like and run as well. 

If you would like to look at the y-hat generated by a given model for a given set of sentences, look_at_predictions.py can be called. The main method of preditions.py can be modified based on the model which you would like to generate predictions and the text that you would like to feed into the model. A sample is below:

       look_at_predictions(data_path="tner/conll2003", test_sentences_path="texts/test_sentences.txt", model_path="model_outputs/transformer256_8_1epoch.pt", output_path="test_256_8_predictions_test.csv", model_dimension=256, heads=8)
