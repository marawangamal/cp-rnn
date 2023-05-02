import os
import os.path as osp

for filename in os.listdir("./runs/ptb"):

	filename_old = filename
	# filename_new = filename.split('_i200')[0] + '_i0' + filename.split('_i200')[1]
	# filename_new = filename.split('_em0_t0')[0] + '_t0'
	# filename_new = filename.split('_char_tchar_tr0')[0] + '_tchar_tr0'

	print("Moving {} to {}".format(filename_old, filename_new))
	os.rename(osp.join("./runs/ptb", filename_old), osp.join("./runs/ptb", filename_new))