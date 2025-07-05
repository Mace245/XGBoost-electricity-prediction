import pandas as pd
import matplotlib.pylab as plt

pjme = pd.read_csv('Data/output.csv', index_col=[0], parse_dates=[0])

# for x, y in pjme:
#     # Create the label text
#     label = f"({x}, {y})"

#     # Place the text on the plot
#     # We add a small offset to y so the text appears just above the point
#     pjme.text(x, y + 0.5, label, ha='center', fontsize=12)

plt.figure(figsize=(10,6))
pjme.plot()
plt.show()