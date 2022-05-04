import torch
from torch.utils.cpp_extension import load

AFCapture = load(
	name="afcapture",
	sources = [
		"dataTools/load.cu",
		"dataTools/wavelet.cu",
		"dataTools/pack12.cu",
		"dataTools/shuffle.cu",
		"dataTools/bayer.cu",
		],
	# TODO remove this, as the includes should be globally available after install of nvcomp
	extra_include_paths = [
	],
	extra_cflags = [
		"-DWITH_CUDA"
	],
	extra_ldflags = [
		"-lnvcomp",
	]);

