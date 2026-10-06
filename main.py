import os
import torch
import random
import numpy as np
import pickle as pkl
from parse import args
import filename_gen


def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])

    if args.show_mode == "print":
        print("Arguments: %s " % args_str)
    elif args.show_mode == "write":
        with open(args.save_file, "w", encoding="utf-8") as f:
            f.writelines("Arguments: %s " % args_str + '\n')

    if args.select_model in ["FedSoG", "FedGNN"]:
        from server import FedGraphServer
        from client import FedGraphClient
        from dataloader import processing_data, processing_valid_data
        from attack import FedGraphAttack
        from col_server import ColServer

        # -------------Data Process-----------#
        file_name_SoG = "./Data/{}.pkl".format(args.data) 
        data_file = open(file_name_SoG, 'rb')
        train_data, valid_data, test_data, uid_list, iid_list, social = pkl.load(data_file)

        valid_data = processing_valid_data(valid_data)
        train_item, train_ratings, rating_max, rating_min = processing_data(uid_list, train_data)

        true_uid_list = [uid for uid in uid_list if (len(train_item[uid]) > 0)]

        # -------------Federated Model------------#
        if args.attack_user == "NoAttack":
            mali_client = []
        else:
            mali_client = random.sample(true_uid_list, k=int(len(uid_list) * args.mali_ratio)) 
        # client
        clients = []
        for uid in uid_list:
            if uid in mali_client:
                clients.append(FedGraphAttack(uid, train_item[uid], train_ratings[uid], list(social[uid]), iid_list).to(args.device))
            else:
                clients.append(FedGraphClient(uid, train_item[uid], train_ratings[uid], list(social[uid])).to(args.device))
        # server
        server = FedGraphServer(clients, true_uid_list, uid_list, iid_list, rating_max, rating_min).to(args.device)

        col_server = ColServer(clients, mali_client).to(args.device)

    elif args.select_model in ["FedNCF", "FedMLP"]:
        from server import FedRecServer
        from client import FedRecClient
        from attack import FedRecAttack
        from dataloader import data_loader
        from col_server import ColServer

        # --------------Data Process-------------#
        m_item, m_user, train_data, valid_data, test_data = data_loader()

        # --------------Federated Model----------#
        if args.attack_user == "NoAttack":
            mali_client = []
        else:
            mali_client = random.sample(range(m_user), k=int(m_user * args.mali_ratio))
        # client
        clients = []
        for uid, (train_data_, valid_data_, test_data_) in enumerate(zip(train_data, valid_data, test_data)):
            if uid in mali_client:
                clients.append(FedRecAttack(train_data_, valid_data_, test_data_, m_item))
            else:
                clients.append(FedRecClient(train_data_, valid_data_, test_data_, m_item))
        # server
        server = FedRecServer(m_item, clients).to(args.device)

        col_server = ColServer(clients, mali_client)

    else:
        print(args.selected_model + " is not exist!!!")

    # ------------------------------------------------train and test---------------------------------------------- #
    for epoch in range(args.epoch):
        # --------------------------train----------------------#
        # NoAttack
        if args.mali_ratio == 0.0 or args.attack_user == "NoAttack":
            epoch_loss = server.train_([], epoch)
        # IndSH
        elif args.attack_user in ["IndSH"]:
            epoch_loss = server.train_(mali_client, epoch)
        # ColSH or ColDP
        elif args.attack_user in ["ColSH", "Noisy_Col"]:
            epoch_loss = server.train_col(mali_client, col_server, epoch)
            if args.select_model in ["FedNCF", "FedMLP"]:
                col_server.train_(server)
        else:
            exit(args.attack_user + " is not exist!!!")

        # --------------------------test------------------------#
        with torch.no_grad():
            if epoch % args.valid_step == 0:
                if args.select_model in ['FedSoG', 'FedGNN']:
                    test_result = server.predict(valid_data)
                    test_loss = test_result[0]
                    hr10, _ = test_result[1].tolist()
                    ndcg10, _ = test_result[3].tolist()

                    if args.show_mode == "write":
                        with open(args.save_file, 'a') as f:
                            f.writelines("Iteration %d, " % (epoch) +
                                         ', loss = {:.5f}, HR@10 = {:.5f}, nDCG@10 = {:.5f}'
                                         .format(test_loss, hr10, ndcg10) + '\n')
                    else:
                        print("Iteration %d, " % (epoch) +
                              ', loss = {:.5f}, HR@10 = {:.5f}, nDCG@10 = {:.5f}'
                              .format(test_loss, hr10, ndcg10))

                else:
                    test_result = server.test_()
                    test_loss = test_result[0]
                    hr10, _= test_result[1].tolist()
                    ndcg10, _= test_result[3].tolist()

                    if args.show_mode == "write":
                        with open(args.save_file, 'a') as f:
                            f.writelines("Iteration %d" % (epoch) +
                                         ', loss = {:.5f}, HR@10 = {:.5f}, nDCG@10 = {:.5f} '
                                         .format(test_loss, hr10, ndcg10) + '\n')
                    else:
                        print("Iteration %d" % (epoch) +
                              ', loss = {:.5f}, HR@10 = {:.5f}, nDCG@10 = {:.5f} '
                              .format(test_loss, hr10, ndcg10))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_seed(args.seed)

    if args.show_mode == "write":
        try:
            main()
        except:
            if not os.path.exists(args.save_file):
                with open(args.save_file, 'w', encoding="utf-8") as f:
                    f.writelines("Error" + "\n")
            else:
                with open(args.save_file, 'a', encoding="utf-8") as f:  
                    f.writelines("Error" + "\n")
    else:
        main()
