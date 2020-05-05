# export CUDA_VISIBLE_DEVICES=1
# python grad-cam-basic.py \
# --image-path './examples/test' \
# --save-path './examples/test_result_basic' \
# --use-cuda \


#export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0,1,2
python grad-cam.py \
--lr 0.03 \
--image-path './examples/test' \
--save-path './examples/test_result_bala' \
--use-cuda \
--low-dim 64 \
--test-only \
--resume batch200_temp0.07_epoch3_model_best.pth.tar \

 