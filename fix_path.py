
# open the file "SWIFT/data/dtd/T2T500.txt" and read the lines
# then for each line, remove the prefix "/scratch/group/real-fs/retrieved/dtd/"
# and overwrite the file with the modified lines

dataset = ['dtd', 'eurosat', 'fgvc-aircraft', 'stanford_cars', 'semi-aves']

for data in dataset:
    file_path = f"data/{data}/T2T500.txt"

    with open(file_path, "r") as file:
        lines = file.readlines()
    modified_lines = [line.replace(f"/scratch/group/real-fs/retrieved/{data}/", "") for line in lines]
    with open(file_path, "w") as file:
        file.writelines(modified_lines)
