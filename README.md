cd src

train: python main.py --model DRCAN --scale 2 --save drcan_x2 --res_scale 0.1 --reset 

test：python main.py --data_test Set5 --scale 2  --test_only
