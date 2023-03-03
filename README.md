# ALN-DSAC
The code is for paper "Representation Learning and Reinforcement Learning for Dynamic Complex Motion Planning System".

## Dependencies
1. [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library

pip3 install -e .

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
