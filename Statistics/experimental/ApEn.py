import random
import math

#https://en.wikipedia.org/wiki/Approximate_entropy 
def approx_entropy(time_series, run_length, filter_level) -> float:
    """
    Approximate entropy
 
    >>> import random
    >>> regularly = [85, 80, 89] * 17
    >>> print(f"{approx_entropy(regularly, 2, 3):e}")
    1.099654e-05
    >>> randomly = [random.choice([85, 80, 89]) for _ in range(17*3)]
    >>> 0.8 < approx_entropy(randomly, 2, 3) < 1
    True
    """
 
    def _maxdist(x_i, x_j):
        return max(abs(ua - va) for ua, va in zip(x_i, x_j))
 
    def _phi(m):
        n = time_series_length - m + 1
        x = [
            [time_series[j] for j in range(i, i + m - 1 + 1)]
            for i in range(time_series_length - m + 1)
        ]
        counts = [
            sum(1 for x_j in x if _maxdist(x_i, x_j) <= filter_level) / n for x_i in x
        ]
        return sum(math.log(c) for c in counts) / n
 
    time_series_length = len(time_series)
 
    return abs(_phi(run_length + 1) - _phi(run_length))
 
 
if __name__ == "__main__":
    import doctest
 
    doctest.testmod()
 
 
def generate_random_numbers(count, lower, upper):
    random_numbers = []
    while len(random_numbers) < count:
        random.seed()
        random_numbers.append(random.randint(lower, upper))
    return random_numbers
 
# Generate random numbers
numbers = generate_random_numbers(
    count=600,
    lower=1,
    upper=6
    
)
 
# Calculate approximate entropy
entropy = approx_entropy(numbers, run_length=2, filter_level=0.5)
print()
print(f"Approximate entropy of the generated sequence: {entropy}")
print()
print(numbers)
