from training import main as train
import sys

def run_with_args(m, h, h_d, k, v, e, b, ep, l_r, d_n, m_t, results_file, model_file, max_seq_len):
    sys.argv = [
        'training.py',
        '--model_dimension', str(m),
        '--num_heads', str(h),
        '--hidden_dimension', str(h_d),
        '--key_dimension', str(k),
        '--value_dimension', str(v),
        '--num_encoder_layers', str(e),
        '--batch_size', str(b),
        '--epoch_count', str(ep),
        '--learning_rate', str(l_r),
        '--dataset_name', str(d_n),
        '--model_type', str(m_t),
        "--model_results_path", f"model_results/{results_file}.csv", 
        '--model_path', f"model_outputs/{model_file}.pt",
        "--max_seq_len", str(max_seq_len)
    ]
    train()

def main():
   # run_with_args(100,0,0,0,0,0,32,5, 0.0005, 'tner/conll2003', 'word2vec', 'results', 'word2vec_4096')
   #heads = [2,4,8]
   #dims = [64,128,256,512]
   #for h in heads:
   #    for d in dims:
   #         run_with_args(d,h,d*2,d,d,6,32,5, 0.0005, 'tner/conll2003', 'transformer', 'transformer_results', f'transformer_{d}_{h}', 113)
    run_with_args(256,8,512,256,256,6,32,1,0.0003, 'tner/conll2003', "transformer",  "metrics_results", f"transformer{512}_{8}", 113)



if __name__ == "__main__":
    main()
