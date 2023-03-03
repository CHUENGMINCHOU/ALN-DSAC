# ALN-DSAC
This code is for the paper: C. Zhou, B. Huang and P. Fränti, "Representation Learning and Reinforcement Learning for Dynamic Complex Motion Planning System," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2023.3247160.

## Dependencies
1. [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
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

## Train curves
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

  
  
