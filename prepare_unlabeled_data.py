import random

datasets = [
    # 'dtd',
    'eurosat',
    # 'fgvc-aircraft',
    # 'stanford_cars',
    # 'food101'
]

for dataset in datasets:

    print(f"\nProcessing {dataset}")
    # read the train.txt and val.txt into a list
    with open(f'data/{dataset}/train.txt', 'r') as f:
        train = f.readlines()
    if train[-1] == '\n':
        print("Removing tailing empty line from train.txt")
        train.pop()
    print(f"train: {len(train)}")

    # add a new line to the last item of train
    train[-1] = train[-1].strip('\n') + '\n'

    with open(f'data/{dataset}/val.txt', 'r') as f:
        val = f.readlines()
    # remove tailing empty line
    if val[-1] == '\n':
        print("Removing tailing empty line from val.txt")
        val.pop()
    print(f"val: {len(val)}")

    train_val = train + val
    print(f"train_val: {len(train_val)}")

    # read the fewshot16_seed1.txt into a list
    with open(f'data/{dataset}/fewshot16_seed1.txt', 'r') as f:
        fewshot = f.readlines()
    if fewshot[-1] == '\n':
        print("Removing tailing empty line from fewshot16_seed1.txt")
        fewshot.pop()
    print(f"fewshot: {len(fewshot)}")

    # remove the fewshot lines from train_val
    for line in fewshot:
        train_val.remove(line)
    print(f"u_train_in: {len(train_val)}")

    # write the train_val to u_train_in.txt
    with open(f'data/{dataset}/u_train_in.txt', 'w') as f:
        f.writelines(train_val)

    # sample fewshot4 and fewshot8 for each class from the fewshot16_seed1.txt
    # collect line by class
    fewshot16 = dict()
    for line in fewshot:
        path, class_id, source = line.strip('\n').split(' ')
        if class_id in fewshot16:
            fewshot16[class_id].append(path)
        else:
            fewshot16[class_id] = [path]

    # set the random seed value
    # seed = 1
    # seed = 2
    seed = 3

    random.seed(seed)

    # randomly sample ct images from each class
    for ct in [4, 8]:
        fewshot_ct = dict()
        for class_id, paths in fewshot16.items():
            if len(paths) < ct:
                fewshot_ct[class_id] = paths
                print(f'class_id: {class_id}, len(paths): {len(paths)}')
            else:
                fewshot_ct[class_id] = random.sample(paths, ct)

        # write out to fewshot{ct}_seed{seed}.txt
        fn = f'data/{dataset}/fewshot{ct}_seed{seed}.txt'
        fewshot_lines = []
        with open(fn, 'w') as f:
            for class_id, paths in fewshot_ct.items():
                for path in paths:
                    line = f'{path} {class_id} 1\n'
                    fewshot_lines.append(line)
                    f.write(line)
        print(f'saved to {fn}')

print("\nDone")