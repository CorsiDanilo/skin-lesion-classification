import pandas as pd


def remove_duplicates(csv_path):
    metadata_train = pd.read_csv(csv_path)

    # Find duplicates in 'lesion_id'
    duplicates = metadata_train[metadata_train.duplicated(
        'lesion_id', keep=False)]

    print(f"Original metadata length: {len(metadata_train)}")
    # Print duplicates
    print(duplicates)

    # Add 'is_duplicated' column
    metadata_train['is_duplicated'] = metadata_train.duplicated(
        'lesion_id', keep=False)

    # Sort by 'lesion_id' and 'is_duplicated'
    metadata_train.sort_values(['lesion_id', 'is_duplicated'], inplace=True)

    # Drop duplicates, keeping the first occurrence
    matadata_train_no_duplicates = metadata_train.drop_duplicates(
        'lesion_id', keep='first')

    matadata_train_no_duplicates.drop('is_duplicated', axis=1, inplace=True)

    # metadata_train_sorted = matadata_train_no_duplicates.sort_values(
    # 'lesion_id')

    print(
        f"Metadata length without duplicates: {len(matadata_train_no_duplicates)}")
    # print(metadata_train_sorted)
    # metadata_train_sorted.to_csv(os.path.join(
    #     DATA_DIR, "HAM10000_metadata_train_no_duplicates.csv"), index=False)
    return matadata_train_no_duplicates
