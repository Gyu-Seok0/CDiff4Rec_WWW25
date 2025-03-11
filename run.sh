# #!/bin/bash

dataset_name=$1

case $dataset_name in
    "yelp")
        cmd="python -u main.py --cuda --dataset yelp --alpha 0.1 --beta 0.1 --gamma 0.8 --ur --r_agg sim --uf --no-tf --f_agg sim --lr 0.0001 --topk 50 --topk_pseudo_user 1000"
        ;;
    "amazon_game")
        cmd="python -u main.py --cuda --dataset amazon_game --alpha 0.8 --beta 0.15 --gamma 0.05 --ur --r_agg avg --uf --no-tf --f_agg sim --lr 0.00005 --topk 20 --topk_pseudo_user 1000"
        ;;
    "citeulike_t")
        cmd="python -u main.py --cuda --dataset citeulike_t --alpha 0.9 --beta 0.05 --gamma 0.05 --ur --r_agg avg --uf --no-tf --f_agg avg --lr 0.00005 --topk 20 --topk_pseudo_user 10000"
        ;;
    *)
        echo "Invalid dataset name! Please provide one of the following dataset names: yelp, amazon_game, citeulike_t."
        exit 1
        ;;
esac

echo $cmd
eval "$cmd"