CUDA_VISIBLE_DEVICES=1 python main_test.py --test_only --save_results \
	                   --save SGM-PyTorch-Cap-Test \
                 	   --data_range 1-800/1-800 \
			   --i_path ./pretrained/model_I_cap.pt \
		           --r_path ./pretrained/model_R_cap.pt \
			   --s_path ./pretrained/model_S_cap.pt \
			   --dir_hr /data1/yangwenhan/datasets/Our_normal_test/ \
			   --dir_lr /data1/yangwenhan/datasets/Our_low_test/
