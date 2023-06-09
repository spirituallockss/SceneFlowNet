CUDA_VISIBLE_DEVICES=1,0 python main.py --exp_name=dcp_v2  \
--emb_nn=dgcnn --pointer=transformer --head=svd \
--batch_size=5 --dataset=kitti2015flow  --test_batch_size=1 --model=unsupervised_dcflow \
--epochs=500 --num_points=1024
