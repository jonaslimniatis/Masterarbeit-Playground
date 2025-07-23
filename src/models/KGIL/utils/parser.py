import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KGIL")
    # invariant learning 
    parser.add_argument('--K',   type=int, default=3, help='augmentation view number')
    parser.add_argument('--lamda',   type=float, default=0.01, help='augmentation view number')
    parser.add_argument('--tau', type=float, default=0.1, help='contrastive')
    parser.add_argument('--epup',   type=int, default=20, help='update epoch')
    parser.add_argument('--itup',   type=int, default=50, help='update iteration')
    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="amazon-All_Beauty", help="Choose a dataset:[Last-FM,amazon-All_Beauty,amazon-Video_Games]") # Updated by Jonas Limniatis
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    ) # Updated by Jonas Limniatis
    parser.add_argument("--model_technique", nargs="?", default="all-MiniLM-L6-v2", help="Choose a model:[all-MiniLM-L6-v2,intfloat/multilingual-e5-large-instruct,Alibaba-NLP/gte-Qwen2-1.5B-instruct,llama3.2, ner]")
    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=120, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size') # Updated by Jonas Limniatis
    parser.add_argument('--test_batch_size', type=int, default=512, help='batch size') # Updated by Jonas Limniatis
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') # Updated by Jonas Limniatis
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 1000]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    # Use parse_known_args() to ignore unknown arguments like --f from Jupyter
    args, unknown = parser.parse_known_args()

    # Optional: Print unknown arguments if you want to see what was ignored
    if unknown:
        print(f"Ignoring unrecognized arguments: {unknown}")


    return args