"""ImageNet-1K protocols; the union contains seven unique OOD datasets."""
GROUPS = {"Standard": ["iNaturalist", "SUN", "Places", "Textures"],
          "Near": ["SSB-Hard", "NINCO"],
          "Far": ["iNaturalist", "Textures", "OpenImage-O"]}
DATASETS = list(dict.fromkeys(d for group in GROUPS.values() for d in group))
FOLDERS = dict(zip(DATASETS, ["inaturalist", "SUN", "Places", "texture", "ssb_hard", "ninco", "openimage_o"]))
COUNTS = dict(zip(DATASETS, [10000,10000,10000,5640,49000,5878,17632]))
