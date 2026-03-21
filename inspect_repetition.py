import scipy.io

path = "data/raw/motor_current_dataset/ITSC/dataset/Data_ITSC.mat"

print("Loading dataset...")
data = scipy.io.loadmat(path)

itsc = data["itsc"]

# primera condición
condition = itsc[0,0]

print("Fields:")
print(condition.dtype.names)

# primera repetición
rep_name = condition.dtype.names[0]

repetition = condition[rep_name][0,0]

print("\nRepetition shape:")
print(repetition.shape)

print("\nFirst row:")
print(repetition[0])

print("\nType of row:")
print(type(repetition[0]))