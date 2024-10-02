import pandas as pd

def get_elf(target):
    metadata = pd.read_csv("../nft-analyst-starter-pack/metadata.csv")
    print("done")
    print(f"finding elf #{target} in metadata.csv...", end="")
    elf = metadata[metadata["asset_id"] == int(target)]
    print("done")
    print("creating embed...", end="")
    description_lines = ['```\n']
    description_length = 0
    for idx, row in elf.iterrows():
        max_k_length = 0
        for col in elf.columns:
            if col.endswith("_attribute") and len(col) - len("_attribute") + 1 > max_k_length:
                max_k_length = len(col) - len("_attribute") + 1
        for col in elf.columns:
            if col.endswith("_attribute"):
                attribute = row[col]
                description = row[col.replace("_attribute", "_description")]
                line = f"{col.replace('_attribute', ''):<{max_k_length}}"
                if attribute is not None:
                    line += f"{attribute}"
                else:
                    line += "None"
                if isinstance(description, str):
                    description = description.replace("â€™", "'")
                    line += f": {description}"
                rarity_score = row[col.replace("_attribute", "_rarity_score")]
                one_of = int(1/rarity_score*metadata.shape[0])
                line += " ("
                print(f"{attribute} {rarity_score=}{one_of=}")
                if one_of <= 50:
                    line += f"1 of {one_of}, "
                line += f"rarity={rarity_score:,.1f})"
                # line = f"{col:<{max_k_length}}: {row[col]}"
                if description_length + len(line) > 2000:
                    break
                description_length += len(line)
                description_lines.append(line)
    description_lines.append('```')
    return description_lines, elf