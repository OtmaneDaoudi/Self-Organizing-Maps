from SOM import *
import matplotlib.pyplot as plt

DIMENSIONS = 3

MAP_ROWS = 20
MAP_COLS = 20

EPOCHES = 2500

DATASET_ROWS = 40
DATASET_COLS = 40
DATASET_SIZE = DATASET_ROWS * DATASET_COLS

dataset = np.random.randint(0, 255, (DATASET_SIZE,  DIMENSIONS))
dataset = dataset / 255

fig, ax = plt.subplots(1, 2)


# plot dataset
ax[0].set_title(f"Generated dataset ({DATASET_ROWS} X {DATASET_COLS})")
ax[0].imshow(dataset.reshape(DATASET_ROWS, DATASET_COLS, DIMENSIONS))
# plt.show()
som = SelfOrganazingMap(MAP_ROWS, MAP_COLS, DIMENSIONS, iterations=EPOCHES)
som.train(dataset)

# plot resulting map
ax[1].set_title(f"Resulting map ({MAP_ROWS} X {MAP_COLS})")
ax[1].imshow(som.weights)
plt.show()
