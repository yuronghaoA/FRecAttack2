from parse import args
import os


def filename_gen():
    dir_path = "./results/{}/{}/".format(args.select_model, args.data)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if args.select_model in ["FedNCF", "FedMLP"]:
        if args.attack_user in ["IndSH",  "ColSH"]:
            if args.attack_item == "RatingOfChange":
                if args.is_detect == 0:
                    file_name = "{}-mr{}-{}-seed{}".format(args.attack_user, args.mali_ratio, args.agg, args.seed)
                elif args.is_detect == 1:
                    file_name = "{}-mr{}-GuardCQ_{}-seed{}".format(args.attack_user, args.mali_ratio, args.agg, args.seed)
        else:
            if args.is_detect == 0:
                file_name = "{}-mr{}-{}-seed{}".format(args.attack_user, args.mali_ratio, args.agg, args.seed)
            elif args.is_detect == 1:
                file_name = "{}-mr{}-GuardCQ_{}-seed{}".format(args.attack_user, args.mali_ratio, args.agg, args.seed)

    elif args.select_model in ["FedSoG", "FedGNN"]:
        if args.attack_user in ["Noisy_Col"]:
            if args.attack_item == "RatingOfChange":
                if args.is_detect == 0:
                    file_name = "{}-mr{}-{}-seed{}".format(args.Noisy_pat, args.mali_ratio, args.agg, args.seed)
                elif args.is_detect == 1:
                    file_name = "{}-mr{}-GuardCQ{}-seed{}".format(args.Noisy_pat, args.mali_ratio, args.agg, args.seed)
        else:
            if args.is_detect == 0:
                file_name = "{}-mr{}-{}-seed{}".format(args.attack_user, args.mali_ratio, args.agg, args.seed)
            elif args.is_detect == 1:
                file_name = "{}-mr{}-GuardCQ_{}-seed{}".format(args.attack_user, args.mali_ratio, args.agg, args.seed)

    args.save_file = os.path.join(dir_path, file_name + '.txt')


if args.show_mode == "write":
    filename_gen()