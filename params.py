import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4", "RN50x64", "RN50x16", "RN50_flat", "RN50_t1", "RN50_t2", "RN50_t3",
                      "RN50_t4", "RN50_t5", "RN50_t6",
                      "RN50_flat_ft", "RN50_t1_pos_ft", "RN50_t2_pos_ft", "RN50_t1_pos", "RN50_t2_pos",
                      "RN50_flat_large", "RN50_t1_large", "RN50_t2_large",
                      "RN50_a2", "RN50_a2s", "ViT-H-14"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B/32", "ViT-L/14", "ViT-B/16"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0
    )
    parser.add_argument(
        "--ckpt",
        help='path to checkpoint file'
    )
    parser.add_argument(
        "--model",
        choices=["RN50", "RN101", "RN50x4", "RN50x64", "RN50x16", "ViT-B/16", "ViT-B/32", "ViT-L/14", "ViT-H-14",
                 "RN50_flat", "RN50_t1", "RN50_t2", "RN50_t3", "RN50_t4", "RN50_t5", "RN50_t6",
                 "RN50_flat_ft", "RN50_t1_pos_ft", "RN50_t2_pos_ft", "RN50_t1_pos", "RN50_t2_pos",
                 "RN50_flat_large", "RN50_t1_large", "RN50_t2_large",
                 "RN50_a2", "RN50_a2s"],
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--dataset", "--ds",
        type=str,
        default=None,
        help="Path to txt file of retrieval data",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision."
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", "--bs", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature of infoNCE loss."
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--patience", type=int, default=1
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")

    args = parser.parse_args()
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
