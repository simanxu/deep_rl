import torch
import numpy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BernoulliBandit:
    """
        伯努利多臂老虎机，输入K表示拉杆个数
    """
    def __init__(self, k):
        self.K = k
        self.probs = torch.rand(k)
        self.best_idx = torch.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]

    def step(self, k):
        """
            当玩家选择了第k个老虎机，根据拉动该老虎机连杆获得奖励的概率返回1（获奖）或0（未获奖）
        """
        if torch.rand(1) < self.probs[k]:
            return 1
        else:
            return 0


setup_seed(1001)

# 均匀分布 [0, 1]
a = torch.rand(2, 2)
# 标准正态分布（均值为0，方差为1，即高斯白噪声）
b = torch.randn(2, 2)

K = 10
bandit_10_arm = BernoulliBandit(K)
print(f"随机生成了一个{K}臂伯努利老虎机")
print(f"获奖概率最大的拉杆为{bandit_10_arm.best_idx.numpy()}，概率为{bandit_10_arm.best_prob.numpy():.4f}")


