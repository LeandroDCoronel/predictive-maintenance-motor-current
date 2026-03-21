import scipy.io
import numpy as np

path = "data/raw/motor_current_dataset/ITSC/dataset/Data_ITSC.mat"
data = scipy.io.loadmat(path)

itsc = data["itsc"]

print("TYPE itsc:", type(itsc))
print("SHAPE itsc:", itsc.shape)

itsc = itsc[0,0]

print("\nFIELDS:")
print(itsc.dtype.names)

for name in itsc.dtype.names:
    item = itsc[name]
    print("\n---", name, "---")
    print("type:", type(item))
    print("shape:", np.shape(item))
    print("sample:", item)
    break