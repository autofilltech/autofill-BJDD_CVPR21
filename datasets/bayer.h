#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "operators.h"

torch::Tensor bayer(torch::Tensor src);

