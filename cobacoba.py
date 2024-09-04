import numpy as np
predictions = [[2, 6, 2], [4, 6, 1]]

final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)

print(final_predictions)

