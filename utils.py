import torch
import torch.utils.data as Data


def load_dataset(filePath):
    data = torch.load(filePath)
    data_load = []
    # contact_map=[]
    max_lingth = 0
    for value in data:
        if max_lingth < len(value["sequence"]) and len(value["sequence"]) < 100:
            max_lingth = len(value["sequence"])

    for value in data:
        if len(value["sequence"]) < 100:
            sequence = " ".join(value["sequence"])
            try:
                contact_map = value["dist_arr"].tolist()
            except Exception as e:
                # print(e)
                contact_map = value["dist_arr"]
            padList = [-1. for i in range(0, max_lingth)]
            for index in range(len(contact_map)):
                for i in range(max_lingth - len(value["sequence"])):
                    if not isinstance(contact_map[index], list):
                        contact_map[index] = contact_map[index].tolist()
                    contact_map[index].append(-1.)
            for index in range(max_lingth - len(value["sequence"])):
                sequence += " PAD"
                contact_map.append(padList)
            data_load.append([sequence, contact_map])

    print(len(data_load))
    return data_load




tran_code = {
    "PAD": 0, "A": 1, 'C': 2, 'D': 3, 'E': 4,
    'F': 5, 'G': 6, 'H': 7, 'K': 8,
    'I': 9, 'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16,
    'T': 17, 'V': 18, 'Y': 19, 'W': 20, "X": 21
}

sentences = [
    # 德语和英语的单词个数不要求相同
    # enc_input                dec_input           dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# 德语和英语的单词要分开建立词库
# Padding Should be Zero
src_vocab = {
    "PAD": 0, "A": 1, 'C': 2, 'D': 3, 'E': 4,
    'F': 5, 'G': 6, 'H': 7, 'K': 8,
    'I': 9, 'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16,
    'T': 17, 'V': 18, 'Y': 19, 'W': 20, "X": 21
}

src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)

def make_data(sequence):
    enc_inputs, contact_maps = [], []
    for i in range(len(sequence)):
        enc_input = [[src_vocab[n] for n in sequence[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        contact_map = [n for n in sequence[i][1]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        enc_inputs.extend(enc_input)
        contact_maps.append(contact_map)

    enc_inputs = torch.LongTensor(enc_inputs)
    contact_maps = torch.Tensor(contact_maps).float()
    return enc_inputs,contact_maps


class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, contact_map):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.contact_map = contact_map

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.contact_map[idx]
