from scipy.io import loadmat

path = "data/raw/motor_current_dataset/ITSC/dataset/Data_ITSC.mat"

print("Loading dataset...")
mat = loadmat(path)

print("\nTop level keys:")
print(mat.keys())

itsc = mat["itsc"]

print("\nType of itsc:")
print(type(itsc))

print("\nShape of itsc:")
print(itsc.shape)

print("\nContent preview:")
print(itsc)