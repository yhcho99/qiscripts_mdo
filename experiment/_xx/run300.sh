#!/bin/bash

mkdir -p /home/sronly/sr-storage/xx_qrft_macro_exp1_stock70_num300/infer
cp -r /home/sronly/sr-storage/xx_qrft_macro_exp1_stock350/infer/* /home/sronly/sr-storage/xx_qrft_macro_exp1_stock70_num300/infer/
python experiment/xx_qrft_weight.py xx_qrft_macro_exp1_stock70_num300

mkdir -p /home/sronly/sr-storage/xx_qrft_macro_exp2_stock70_num300/infer
cp -r /home/sronly/sr-storage/xx_qrft_macro_exp2_stock350/infer/* /home/sronly/sr-storage/xx_qrft_macro_exp2_stock70_num300/infer/
python experiment/xx_qrft_weight.py xx_qrft_macro_exp2_stock70_num300

mkdir -p /home/sronly/sr-storage/xx_qrft_macro_exp3_stock70_num300/infer
cp -r /home/sronly/sr-storage/xx_qrft_macro_exp3_stock350/infer/* /home/sronly/sr-storage/xx_qrft_macro_exp3_stock70_num300/infer/
python experiment/xx_qrft_weight.py xx_qrft_macro_exp3_stock70_num300