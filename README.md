<h1 align="center"> Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed </h1>


The PyTorch Implementation of *AAAI 2024 -- "Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed"*[pdf](https://arxiv.org/abs/2312.12469).

<p align="center"><img src="./imgs/main.jpg" width=95%></p>

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
<p align="center"><img src="./imgs/Performance.jpg" width=95%></p>


### Acknowledgments

* We would like to thank the anonymous reviewers and (S)ACs of AAAI 2024 for their constructive comments and dedicated service to the community. The reviews are available [here](https://github.com/xybFight/GNARKD/blob/master/AAAI24_Comments.pdf)
* We also would like to thank the following repositories, which are baselines of our code:

  * https://github.com/wouterkool/attention-learn-to-route

  * https://github.com/yd-kwon/POMO

  * https://github.com/xbresson/TSP_Transformer


### Citation

If you find our paper and code useful, please cite our paper:

```tex
@misc{Xiao2023,
      title={Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed}, 
      author={Yubin Xiao and Di Wang and Boyang Li and Mingzhao Wang and Xuan Wu and Changliang Zhou and You Zhou},
      year={2023},
      eprint={2312.12469},
      archivePrefix={arXiv},
}
```
Or after the publication of the AAAI24 paper:
```tex
@inproceedings{Ding2020,
  author={Yubin Xiao and Di Wang and Boyang Li and Mingzhao Wang and Xuan Wu and Changliang Zhou and You Zhou},
  year      = {2024},
  title     = {Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
}

