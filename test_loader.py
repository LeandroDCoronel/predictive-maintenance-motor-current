from src.data_loader import load_motor_current_data

path = "data/raw/motor_current_dataset/ITSC/dataset/Data_ITSC.mat"

data = load_motor_current_data(path)

print("Dataset shape:", data.shape)
print("First rows:")
print(data[:5])