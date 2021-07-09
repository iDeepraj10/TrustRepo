from collections import Counter

a = [1,2,3,1,2,1,2]

c = Counter(a).most_common()[0][0]

print(c)