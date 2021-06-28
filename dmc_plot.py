import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

sns.set_style("white")
df = pd.read_csv(sys.argv[1])
df["step"] += 1
df = df[df['tau']==0.01]
print(df.keys())

g=sns.PairGrid(df,x_vars=['step'],y_vars=['elocal','weight','eref','weightvar'],hue='popstep')
g.map(plt.scatter,s=1)
g.add_legend()
plt.tight_layout()
plt.show()
#plt.savefig("traces.pdf", bbox_inches='tight')
