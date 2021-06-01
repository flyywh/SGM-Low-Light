CUDA_VISIBLE_DEVICES=1 python main_test.py --test_only --save_results \
	                   --save SGM-PyTorch-Cap-Test \
                 	   --data_range 1-800/1-800 \
			   --i_path ./pretrained/model_I_real_noise_free.pt \
		           --r_path ./pretrained/model_R_real_noise_free.pt \
			   --s_path ./pretrained/model_S_real_noise_free.pt \
			   --dir_hr /data1/yangwenhan/datasets/Low_real_test_2_rs/ \
			   --dir_lr /data1/yangwenhan/datasets/Low_real_test_2_rs/
