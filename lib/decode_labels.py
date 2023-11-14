import pandas as pd


def decode_labels(label_file, attributes):
    """ from [0 1 0 0 ..] to 'wheel' """
    labels = pd.read_csv(label_file, delimiter=' ', index_col=0, header=None)
    labels_encoded = []
    fnames = []
    for f, row in labels.iterrows():
        fnames.append(f)
        labels_encoded.append(attributes[row.idxmax() - 1])
    return fnames, labels_encoded