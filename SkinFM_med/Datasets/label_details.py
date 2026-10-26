
# paper: https://www.nature.com/articles/s41746-024-01395-z

# MILK10K datasets
MILK10K_CLASS_COLS = [
    "AKIEC",  "BCC",  "BEN_OTH",  "BKL",  "DF",  "INF",  "MAL_OTH",  "MEL",  "NV",  "SCCKA",  "VASC"]

MILK10K_TREE = {
    "root": {
        "benign": [
            "NV",        # melanocytic nevus
            "BKL",       # benign keratosis
            "DF",        # dermatofibroma
            "VASC",      # vascular
            "BEN_OTH",   # benign other
            "INF"        # inflammatory
        ],
        "premalignant": [ "AKIEC" ],
        "malignant": [
            "MEL",       # melanoma
            "BCC",       # basal cell carcinoma
            "SCCKA",     # squamous / keratoacanthoma
            "MAL_OTH"    # malignant other
        ] }}
MILK10K_NODES = [ "AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC","benign","premalignant","malignant","root"]


# DERM12345 datasets

DERM12345_CLASS_COLS = [
    "acb", "acd", "ajb", "ajd", "ak", "alm", "angk", "anm",
    "bcc", "bd", "bdb", "cb", "ccb", "ccd", "cd", "ch",
    "cjb", "db", "df", "dfsp", "ha", "isl", "jb", "jd",
    "ks", "la", "lk", "lm", "lmm", "ls", "mcb", "mel",
    "mpd", "pg", "rd", "sa", "scc", "sk", "sl", "srjd"]


DERM12345_TREE = {
    "root": {

        # -------------------------------------
        # MELANOCYTIC BENIGN
        # -------------------------------------
        "melanocytic_benign": {

            "banal_compound": [
                "acb",   # acral compound banal nevus
                "ccb",   # congenital compound banal nevus
                "mcb",   # Miescher compound banal nevus
                "cb",    # compound banal nevus
            ],

            "banal_dermal": [
                "bdb",   # blue dermal banal nevus
                "db",    # dermal banal nevus
            ],

            "banal_junctional": [
                "ajb",   # acral junctional banal nevus
                "cjb",   # congenital junctional banal nevus
                "jb",    # junctional banal nevus
            ],

            "lentigo": [
                "isl",   # ink spot lentigo
                "ls",    # lentigo simplex
                "sl",    # solar lentigo
            ],

            "dysplastic_compound": [
                "acd",   # acral compound dysplastic nevus
                "ccd",   # congenital compound dysplastic nevus
                "cd",    # compound dysplastic nevus
            ],

            "dysplastic_junctional": [
                "ajd",   # acral junctional dysplastic nevus
                "srjd",  # Spitz/Reed junctional dysplastic nevus
                "jd",    # junctional dysplastic nevus
            ],

            "dysplastic_recurrent": [
                "rd",    # recurrent dysplastic nevus
            ],
        },

        # -------------------------------------
        # MELANOCYTIC MALIGNANT
        # -------------------------------------
        "melanocytic_malignant": {

            "melanoma": [
                "alm",   # acral lentiginous melanoma
                "anm",   # amelanotic melanoma
                "lm",    # lentigo maligna
                "lmm",   # lentigo maligna melanoma
                "mel",   # melanoma
            ],
        },

        # -------------------------------------
        # NONMELANOCYTIC BENIGN
        # -------------------------------------
        "nonmelanocytic_benign": {

            "keratinocytic_benign": [
                "sk",    # seborrheic keratosis
                "lk",    # lichen planus-like keratosis
            ],

            "fibrohistiocytic_benign": [
                "df",    # dermatofibroma
            ],

            "vascular_benign": [
                "angk",  # angiokeratoma
                "ha",    # hemangioma
                "la",    # lymphangioma
                "pg",    # pyogenic granuloma
                "sa",    # spider angioma
            ],
        },

        # -------------------------------------
        # NONMELANOCYTIC INDETERMINATE
        # -------------------------------------
        "nonmelanocytic_indeterminate": {

            "keratinocytic_indeterminate": [
                "ak",    # actinic keratosis
            ],
        },

        # -------------------------------------
        # NONMELANOCYTIC MALIGNANT
        # -------------------------------------
        "nonmelanocytic_malignant": {

            "keratinocytic_malignant": [
                "bcc",   # basal cell carcinoma
                "bd",    # Bowen disease
                "ch",    # cutaneous horn
                "mpd",   # mammary Paget disease
                "scc",   # squamous cell carcinoma
            ],

            "fibrohistiocytic_malignant": [
                "dfsp",  # dermatofibrosarcoma protuberans
            ],

            "vascular_malignant": [
                "ks",    # Kaposi sarcoma
            ],
        },
    }
}

DERM12345_NODES = [
    # 40 leaf classes
    "acb","acd","ajb","ajd","ak","alm","angk","anm", "bcc","bd","bdb","cb","ccb","ccd","cd","ch", "cjb","db","df","dfsp","ha","isl","jb","jd", "ks","la","lk","lm","lmm","ls","mcb","mel",    "mpd","pg","rd","sa","scc","sk","sl","srjd",
    # 15 main classes
    "banal_compound","banal_dermal","banal_junctional","lentigo","dysplastic_compound","dysplastic_junctional","dysplastic_recurrent","melanoma",
    "keratinocytic_benign","fibrohistiocytic_benign","vascular_benign",
    "keratinocytic_indeterminate",
    "keratinocytic_malignant",
    "fibrohistiocytic_malignant",
    "vascular_malignant",
    # 5 superclasses
    "melanocytic_benign",
    "melanocytic_malignant",
    "nonmelanocytic_benign",
    "nonmelanocytic_indeterminate",
    "nonmelanocytic_malignant",
    "root",
]


ISIC2019_CLASS_COLS = [
    "MEL",   # melanoma
    "NV",    # melanocytic nevus
    "BCC",   # basal cell carcinoma
    "AK",    # actinic keratosis
    "BKL",   # benign keratosis
    "DF",    # dermatofibroma
    "VASC",  # vascular lesion
    "SCC",   # squamous cell carcinoma
    "UNK",   # unknown / none of the others
]
# =========================================================
# ISIC 2019 Hierarchical Tree
# root
# ├── benign
# │   ├── melanocytic_benign → NV
# │   ├── keratinocytic_benign → BKL
# │   ├── fibrohistiocytic_benign → DF
# │   └── vascular_benign → VASC
# ├── premalignant
# │   └── keratinocytic_precancer → AK
# ├── malignant
# │   ├── melanocytic_malignant → MEL
# │   └── keratinocytic_malignant → BCC, SCC
# └── unknown
#     └── UNK
# =========================================================


ISIC2019_TREE = {
    "root": {
        "benign": {
            "melanocytic_benign": [
                "NV",
            ],
            "keratinocytic_benign": [
                "BKL",
            ],
            "fibrohistiocytic_benign": [
                "DF",
            ],
            "vascular_benign": [
                "VASC",
            ],
        },
        "premalignant": {
            "keratinocytic_precancer": [
                "AK",
            ],
        },
        "malignant": {
            "melanocytic_malignant": [
                "MEL",
            ],
            "keratinocytic_malignant": [
                "BCC",
                "SCC",
            ],
        },
        "unknown": [
            "UNK",
        ],
    }
}

ISIC2019_NODES = [
    # 9 leaf classes — must match CLASS_COLS
    "MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK",
    # disease-group nodes
    "melanocytic_benign",
    "keratinocytic_benign",
    "fibrohistiocytic_benign",
    "vascular_benign",
    "keratinocytic_precancer",
    "melanocytic_malignant",
    "keratinocytic_malignant",
    # first layer after root
    "benign",
    "premalignant",
    "malignant",
    "unknown",
    # root
    "root",
]

## cifar 10 dataset

CIFAR10_CLASS_COLS = [ "airplane", "automobile",    "bird",    "cat",     "deer",          "dog",    "frog",     "horse",   "ship",         "truck",   ]

CIFAR10_TREE = {
    "root": {
        "animal": [
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse"
        ],
        "vehicle": [
            "airplane",
            "automobile",
            "ship",
            "truck"
        ]
    }
}

CIFAR10_NODES = [
    "airplane",      # 0
    "automobile",    # 1
    "bird",          # 2
    "cat",           # 3
    "deer",          # 4
    "dog",           # 5
    "frog",          # 6
    "horse",         # 7
    "ship",          # 8
    "truck",         # 9
    "animal",        # 10
    "vehicle",       # 11
    "root"           # 12
]



DATASET_HIERARCHIES = {
    "milk10k": {
        "tree": MILK10K_TREE,
        "nodes": MILK10K_NODES,
        'class_col': MILK10K_CLASS_COLS,
    },

    "isic2019": {
        "tree": ISIC2019_TREE,
        "nodes": ISIC2019_NODES,
        'class_col': ISIC2019_CLASS_COLS
    },

    "derm12345": {
        "tree": DERM12345_TREE,
        "nodes": DERM12345_NODES,
        'class_col': DERM12345_CLASS_COLS,
    },
    
    "cifar10": {
        "tree": CIFAR10_TREE,
        "nodes": CIFAR10_NODES,
        'class_col': CIFAR10_CLASS_COLS,
    },

}


def get_hierarchy(dataset):
    dataset = dataset.lower()

    if dataset not in DATASET_HIERARCHIES:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            f"Available: {list(DATASET_HIERARCHIES.keys())}"
        )

    config = DATASET_HIERARCHIES[dataset]

    return config["tree"], config["nodes"], config['class_col']