CUDA_VISIBLE_DEVICES=1 python main_test.py --test_only --save_results \
	                   --save SGM-PyTorch-Syn-Test \
                 	   --data_range 1-800/1-800 \
			   --i_path ./pretrained/model_I_syn.pt \
		           --r_path ./pretrained/model_R_syn.pt \
			   --s_path ./pretrained/model_S_syn.pt \
			   --dir_hr /data1/yangwenhan/datasets/Normal_test/ \
			   --dir_lr /data1/yangwenhan/datasets/Low_test/
