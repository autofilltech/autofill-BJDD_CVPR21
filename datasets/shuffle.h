#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "operators.h"

torch::Tensor shuffle(torch::Tensor src);
torch::Tensor shuffle4(torch::Tensor src);
torch::Tensor unshuffle(torch::Tensor src);
torch::Tensor unshuffle4(torch::Tensor src);

