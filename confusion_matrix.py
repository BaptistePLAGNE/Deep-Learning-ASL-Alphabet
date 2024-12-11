import matplotlib.pyplot as plt
import numpy as np

# Example confusion matrix data
confusion_matrix = np.array(
[[80,7,2,1,4,1,1,4],
[9,81,1,2,5,1,1,0],
[3,4,88,4,0,1,0,0],
[3,4,4,79,0,7,0,3],
[2,1,2,0,83,8,0,4],
[3,4,2,5,3,76,4,3],
[1,1,1,2,2,6,82,5],
[1,0,1,3,2,6,0,87]])
classes =["A","B","C","H","L","R","W","Y"]


# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(confusion_matrix, cmap=plt.cm.Oranges)
plt.colorbar(cax, shrink=0.8)
output_file = "./confusion_matrix.png"
# Adding labels to the axes
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
# plt.xlabel('Predicted')
# plt.ylabel('Labels')

# Adding text annotations
for (i, j), val in np.ndenumerate(confusion_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')

# plt.title('Confusion Matrix')
plt.savefig(output_file, format='png')
