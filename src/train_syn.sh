CUDA_VISIBLE_DEVICES=0 python main.py --save_models --save SGM-PyTorch-Syn-Train \
	                                            --data_range 1-1200/1-100 \
                                                    --dir_hr /data1/yangwenhan/datasets/Normal/ \
                                                    --dir_lr /data1/yangwenhan/datasets/Low/
