# collective_matmul

This unit test composes two back-to-back GEMM layers (FC1 and FC2 of LLM MLP). FC1 does AG+GEMM, and FC2 does GEMM+RS.

## Running examples

### 175B config

`python collective_matmul.py --dp 2 --tp 4`

You can change dp (Data Parallel) and tp (Tensor Model Parallel) by simply giving differen numbre to above commandline. 

To run baseline (i.e., no overlapping), add `--no_tp_overlap` in the commandline.

### 5B config

`python collective_matmul.py --batch_size 4 --hidden_size 4096`

DP, TP, and overlapping arguments are configured in the same way as 175B.
