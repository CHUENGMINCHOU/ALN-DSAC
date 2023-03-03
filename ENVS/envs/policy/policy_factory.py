from ENVS.envs.policy.linear import Linear
from ENVS.envs.policy.orca import ORCA
#from ENVS.envs.policy.sil import SIL
#from ENVS.envs.policy.lstm_rl import LstmRL
#from ENVS.envs.policy.sarl import SARL
from ENVS.envs.policy.sa_sac_rl import SaSacRL


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
#policy_factory['lstm_rl'] = LstmRL
#policy_factory['sarl'] = SARL
policy_factory['sa_sac_rl'] = SaSacRL
policy_factory['none'] = none_policy
