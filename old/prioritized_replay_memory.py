from sortedcontainers import SortedDict
import numpy.random as random
import numpy as np


class PrioritizedExperienceReplay_Ranked:
    def __init__(self):
        self.sd = SortedDict()
        self.alpha = 0.7
        self.epsilon = 0.001
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.0001

    def store(self, experiences, td_errors):
        for experience, td_error in zip(experiences, td_errors):
            value = td_error.cpu().numpy()
            state, action, reward, next_state, done = experience
            state = tuple(state)
            next_state = tuple(next_state)
            experience = (state, action, reward, next_state, done)
            self.sd[experience] = (value + self.epsilon) ** self.alpha

    def sample(self, number):
        length = len(self.sd) - 1
        chunks = int(length / number)
        elements = []
        values = []
        total = sum(self.sd.values())
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        # get 1 element in each of the bins
        for i in range(number):
            a = i * chunks
            b = (i + 1) * chunks
            key, value = self.sd.peekitem(np.random.random_integers(a, b))
            elements.append(key)
            values.append(value / total)
        is_weight = np.power(np.dot(len(self.sd), values), -self.beta)
        is_weight /= is_weight.max()
        return elements, is_weight


def main():
    exp = PrioritizedExperienceReplay_Ranked()


if __name__ == "__main__":
    main()
