<h1 align="center"> Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed </h1>


The PyTorch Implementation of *AAAI 2024 -- "Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed"*.

<p align="center"><img src="./imgs/main.png" width=95%></p>

This paper introduce a novel and generic method for solving VRPs named GNARKD to transform AR models into NAR ones to improve the inference speed while preserving essential knowledge.


### How to Run


```shell
# 1. Training (for each teacher, e.g. POMO for TSP)
python -u GNARKD-POMO\TSP\Training.py

# Note that due to file size limitations, we removed the teacher's pre-training parameters, which you can download from the github link mentioned in the corresponding paper for successful training.


# 2. Testing (e.g., GNARKD-POMO for TSP)
python -u GNARKD-POMO\TSP\Test_file.py
```

The detail performance is as follows.
![image](https://github.com/xybFight/GNARKD/imgs/Performance.jpg)


### Acknowledgments

* We would like to thank the anonymous reviewers and (S)ACs of AAAI 2024 for their constructive comments and dedicated service to the community. The reviews are available [here](https://github.com/xybFight/GNARKD/AAAI24_Comments.pdf).

* We also would like to thank the following repositories, which are baselines of our code:

  * https://github.com/wouterkool/attention-learn-to-route

  * https://github.com/yd-kwon/POMO

  * https://github.com/xbresson/TSP_Transformer


### Citation

If you find our paper and code useful, please cite our paper:


to be continue...
