import argparse
import torch.cuda as cuda


def object_parse():
    parser = argparse.ArgumentParser(description="Run General Framework!")

    # General Configuration
    parser.add_argument("--show_mode",     type=str,   default="write",   choices=["write", "print"])
    parser.add_argument("--select_model",  type=str,   default="FedNCF",  choices=["FedNCF", "FedMLP", "FedGNN", "FedSoG"])
    parser.add_argument("--data",          type=str,   default="ML_1M", choices=["ML_1M", 'Steam', "filmtrust", "lastfm"])
    parser.add_argument("--device",        type=str,   default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--layers",        type=str,   default="[64, 32, 16, 8]")
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--embedding_dim", type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=0.005)
    parser.add_argument("--epoch",         type=int,   default=5000)
    parser.add_argument("--frac",          type=float, default=0.1)
    parser.add_argument("--num_neg",       type=int,   default=1)
    parser.add_argument("--seed",          type=int,   default=608)
    parser.add_argument("--top_k",         type=str,   default="[10,20]")
    parser.add_argument("--agg",           type=str,   default="avg", choices=["avg"])
    parser.add_argument('--valid_step',     type=int,   default=1)

    # For our attack
    parser.add_argument("--mali_ratio",    type=float, default=0.0)
    parser.add_argument("--attack_user",   type=str,   default="NoAttack", choices=["IndSH",  "ColSH", "Noisy_Col",  "NoAttack"])
    parser.add_argument("--Noisy_pat",     type=str,   default="ColDP", choices=["IndDP", "ColDP"])
    parser.add_argument("--attack_item",   type=str,   default="RatingOfChange") 
    parser.add_argument("--sample_size",   type=int,   default=50)
    parser.add_argument("--alpha",         type=float, default=0.1)
    parser.add_argument("--sigma",         type=float, default=0.1)
    parser.add_argument("--window_size",   type=int,   default=2) 

    # For our defense
    parser.add_argument("--is_detect",     type=int,     default=1)
    parser.add_argument("--start_detect",  type=int,     default=1)
    parser.add_argument("--wind_sz",       type=int,     default=50)

    # For FedSoG or FedGNN
    parser.add_argument('--clip',           type=float, default=0.1)
    parser.add_argument('--laplace_lambda', type=float, default=0.1)
    parser.add_argument("--loss",           type=str,   default="mae")
    parser.add_argument('--weight_decay',   type=float, default=0.001)
    parser.add_argument('--head_num',       type=int,   default=1)

    return parser.parse_args()

args = object_parse()
