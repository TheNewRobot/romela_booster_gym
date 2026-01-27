import isaacgym
from runners.trainer import Runner

if __name__ == "__main__":
    runner = Runner(test=False)
    runner.train()
