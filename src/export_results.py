import json
import pandas as pd
import os


# Read the JSON file
def read_json(json_file):
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return None
    with open(json_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Invalid JSON format: {json_file}")
            return None
    return data


if __name__ == '__main__':
    SCENES = (
        "courtyard",
        "delivery_area",
        "electro",
        "facade",
        "kicker",
        "meadow",
        "office",
        "pipes",
        "playground",
        "relief",
        "relief_2",
        "terrace",
        "terrains",
    )

    # Initialize the DataFrame with appropriate columns
    df = pd.DataFrame(columns=['Scene', 'Method', 'Missing cameras',
                               'Auc@3', 'Auc@5', 'Auc@10', 'Auc@30',
                               'RRE@3', 'RRE@5', 'RRE@10', 'RRE@30',
                               'RTE@3', 'RTE@5', 'RTE@10', 'RTE@30'])

    methods = ['Glomap', 'VGGSfm', 'FlowMap', 'AceZero']

    for scene in SCENES:
        for method in methods:
            # Construct the file path
            if str.lower(method) == 'glomap':
                json_file = f'../data/results/{str.lower(method)}/ETH3D/{scene}/colmap/sparse/0/auc.json'
            elif str.lower(method) == 'flowmap':
                json_file = f'../data/results/{str.lower(method)}/ETH3D/{scene}/colmap/sparse/0/auc.json'
            elif str.lower(method) == 'vggsfm':
                json_file = f'../data/results/{str.lower(method)}/ETH3D/{scene}/colmap/sparse/0/auc.json'
            elif str.lower(method) == 'acezero':
                json_file = f'../data/results/{str.lower(method)}/ETH3D/{scene}/colmap/sparse/0/auc.json'
            else:
                print(f"Invalid method: {str.lower(method)}")
                continue

            # Read the JSON file
            data = read_json(json_file)

            if data is None:
                # Append a row indicating missing data
                df.loc[len(df)] = {
                    'Scene': scene,
                    'Method': method,
                    'Missing cameras': None,
                    'Auc@3': None, 'Auc@5': None, 'Auc@10': None, 'Auc@30': None,
                    'RRE@3': None, 'RRE@5': None, 'RRE@10': None, 'RRE@30': None,
                    'RTE@3': None, 'RTE@5': None, 'RTE@10': None, 'RTE@30': None,
                }
                continue

            # Extract values from JSON
            row = {
                'Scene': scene,
                'Method': method,
                'Missing cameras': data.get('Missing_cameras', None),
                'Auc@3': data.get('Auc_3', None),
                'Auc@5': data.get('Auc_5', None),
                'Auc@10': data.get('Auc_10', None),
                'Auc@30': data.get('Auc_30', None),
                'RRE@3': data.get('RRE_3', None),
                'RRE@5': data.get('RRE_5', None),
                'RRE@10': data.get('RRE_10', None),
                'RRE@30': data.get('RRE_30', None),
                'RTE@3': data.get('RTE_3', None),
                'RTE@5': data.get('RTE_5', None),
                'RTE@10': data.get('RTE_10', None),
                'RTE@30': data.get('RTE_30', None),
            }

            # Append the row to the DataFrame
            df.loc[len(df)] = row

    # Save the results to a CSV file
    output_csv = '../data/results/results.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")