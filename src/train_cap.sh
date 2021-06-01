CUDA_VISIBLE_DEVICES=0 python main.py --save_models --save SGM-PyTorch-Cap-Train \
	                                            --data_range 1-1200/1-100 \
                                                    --dir_hr /data1/yangwenhan/datasets/Our_normal/ \
                                                    --dir_lr /data1/yangwenhan/datasets/Our_low/
