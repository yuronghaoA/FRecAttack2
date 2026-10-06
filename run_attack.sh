

Attack Example:

# FedNCF [NoAttack + Ind-SH + Col-SH]  ML-1M
nohup python main.py --select_model FedNCF --data ML_1M --lr 0.005 --epoch 360 --mali_ratio 0.0 --agg avg --is_detect 0 &
nohup python main.py --select_model FedNCF --data ML_1M --lr 0.005 --epoch 355 --mali_ratio 0.1 --attack_user IndSH --attack_item RatingOfChange --sample_size 100 --alpha 0.1 --agg avg --is_detect 0 &
nohup python main.py --select_model FedNCF --data ML_1M --lr 0.005 --epoch 355 --mali_ratio 0.1 --attack_user ColSH --attack_item RatingOfChange --sample_size 100 --alpha 0.1 --agg avg --is_detect 0 &


# FedSoG [NoAttack + Ind-DP + Col-DP] filmtrust
nohup python main.py --select_model FedSoG --data filmtrust --lr 0.1 --epoch 1630 --mali_ratio 0.0 --alpha 0.1 --attack_user NoAttack --agg avg --is_detect 0  &
nohup python main.py --select_model FedSoG --data filmtrust --lr 0.1 --epoch 1455 --mali_ratio 0.1 --alpha 0.1 --attack_user Noisy_Col --attack_item RatingOfChange --Noisy_pat IndDP --window_size 2 --sample_size 50 --agg avg --is_detect 0  &
nohup python main.py --select_model FedSoG --data filmtrust --lr 0.1 --epoch 1455 --mali_ratio 0.1 --alpha 0.1 --attack_user Noisy_Col --attack_item RatingOfChange --Noisy_pat ColDP --window_size 2 --sample_size 50 --agg avg --is_detect 0  &

