import torch
from torch.utils.cpp_extension import load

AFCapture = load(
	name="afcapture",
	sources = [
		"datasets/load.cu",
		"datasets/wavelet.cu",
		"datasets/pack12.cu",
		"datasets/shuffle.cu",
		"datasets/bayer.cu",
		],
	# TODO remove this, as the includes should be globally available after install of nvcomp
	extra_include_paths = [
	],
	extra_cflags = [
		"-DWITH_CUDA",
	],
	extra_ldflags = [
		"-lnvcomp",
	]);

