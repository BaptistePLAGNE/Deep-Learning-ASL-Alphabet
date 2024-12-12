import torch
import random
import numpy as np

from tqdm             import tqdm
from torch.utils.data import DataLoader

from helper_logger  import DataLogger
from model_scn2     import ExtendedSimpleCNN2D
from helper_tester  import ModelTesterMetrics
from dataset        import SimpleTorchDataset
from torchvision    import transforms

SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.use_deterministic_algorithms(True)

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics = ModelTesterMetrics()

metrics.loss       = torch.nn.BCEWithLogitsLoss()
metrics.activation = torch.nn.Softmax(1)

model        = ExtendedSimpleCNN2D(3, 8).to(device)
optimizer    = torch.optim.Adam(model.parameters(), lr = 0.0001)

testing_dataset    = SimpleTorchDataset('./01_data/test')

testing_datasetloader    = DataLoader(testing_dataset,    batch_size = 1,          shuffle = True)

# Load Model State
model = ExtendedSimpleCNN2D(3,8)

state_dictonary = torch.load("./03_runs/class_langage-4/best_checkpoint.pth", map_location = device)
model.load_state_dict(state_dictonary['model_state'])
model = model.to(device)   # set the model to evaluation
metrics.reset() # reset the metrics

for (image, label) in tqdm(testing_datasetloader):
        
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        metrics.compute(output, label)

testing_mean_loss     = metrics.average_loss()
testing_mean_accuracy = metrics.average_accuracy()

print("")
print(f"# Final Testing Loss     : {testing_mean_loss}")
print(f"# Final Testing Accuracy : {testing_mean_accuracy}")
print(f"# Report :")
print(metrics.report())
print(f"# Confusion Matrix :")
print(metrics.confusion())
print("")