from single_point import generate_single_point

dataset, validation = generate_single_point(5, 5)
print("Dataset shape", dataset.shape)
print("Validation shape", validation.shape)
