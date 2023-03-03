# ALN-DSAC
This code is for the paper: C. Zhou, B. Huang and P. Fränti, "Representation Learning and Reinforcement Learning for Dynamic Complex Motion Planning System," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2023.3247160.

## Abstract
Abstract—Indoor motion planning challenges researchers because of the high density and unpredictability of moving obstacles. Classical algorithms work well in the case of static obstacles but suffer from collisions in the case of dense and dynamic obstacles. Recent reinforcement learning algorithms provide safe solutions for multiagent robotic motion planning system. However, these algorithms face challenges in convergence: slow convergence speed and suboptimal converged result. Inspired by reinforcement learning and representation learning, we introduced the ALN-DSAC: a hybrid motion planning algorithm where attention-based LSTM and novel data replay combine with discrete soft actor critic. First, we implemented discrete soft actor critic algorithm which is the soft actor critic in the setting of discrete action space. Second, we optimized existing distance-based LSTM encoding by attention-based encoding to improve the data quality. Third, we introduced novel data replay method by combining the online learning and offline learning to improve the efficacy of data replay. The convergence of our ALN-DSAC outperforms that of trainable state-of-arts. Evaluations demonstrate that our algorithm achieves near 100% success with less time to reach the goal in motion planning tasks when comparing to the state-of-arts. Test code is available on https://github.com/CHUENGMINCHOU/ALN-DSAC.

![image](https://user-images.githubusercontent.com/22268151/222774256-98076d26-0b3c-44c7-b052-e4a59d3cf870.png)


## Dependencies
1. [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install the dependencies
```
pip3 install -e .
```

## Commands
1. Train a policy.
```
python3 train.py --policy sa_sac_rl

```
2. Test policies with 500 test cases.
```
python3 test.py --policy sa_sac_rl --output_dir ENVS/data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python3 test.py --policy sa_sac_rl --output_dir ENVS/data/output --phase test --visualize --test_case 0
```

## Training curves
![image](https://user-images.githubusercontent.com/22268151/222767985-98e9c99d-70b9-4206-94aa-2374504aa89d.png)

## Motion planning process
![image](https://user-images.githubusercontent.com/22268151/222768330-ad40e4eb-9142-4fd5-a5b6-9b945d18638b.png)

## Policy evolvement
![image](https://user-images.githubusercontent.com/22268151/222768603-991fe309-eb8a-4952-87e8-cf16ebf77e39.png)

## HOW TO CITE THIS PAPER
If you find this contribution useful, please cite it BY: 
```
C. Zhou, B. Huang and P. Fränti, "Representation Learning and Reinforcement Learning for Dynamic Complex Motion Planning System," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2023.3247160.
```

OR BY:
```
@ARTICLE{10058085,
  author={Zhou, Chengmin and Huang, Bingding and Fränti, Pasi},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Representation Learning and Reinforcement Learning for Dynamic Complex Motion Planning System}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2023.3247160}}
```

  
  
