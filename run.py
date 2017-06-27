# run regression.py with increasing number of trees
# run this with just "python run.py". It will fail if you run using az ml execute.

import os

for numtrees in range(5, 25, 5):
    os.system('az ml execute start -c local ./regression.py {}'.format(numtrees))
