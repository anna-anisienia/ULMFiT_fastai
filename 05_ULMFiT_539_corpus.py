from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
path = Path('ULMFiT/full_20_labels_same_data')
path.mkdir(exist_ok=True)
torch.cuda.set_device(1)
df = pd.read_csv("dataset/full_corpus.csv")
df["lm_label"] = [0]*len(df)
train, valid = train_test_split(df, test_size = 0.3)
test, valid = train_test_split(valid, test_size = 0.6)

# keep only cols relevant to Classification
train_clas = train.drop(columns = ["lm_label"], axis=1)
test_clas = test.drop(columns = ["lm_label"], axis=1)
valid_clas = valid.drop(columns = ["lm_label"], axis=1)
# save them to CSV in clas_path
clas_path = path/'clas'
clas_path.mkdir(exist_ok=True)
train_clas.to_csv(clas_path/"train.csv", header=False, index=False)
test_clas.to_csv(clas_path/"test.csv", header=False, index=False)
valid_clas.to_csv(clas_path/"valid.csv", header=False, index=False)

# keep only cols relevant to LM
train_lm = train[["lm_label", "Fulltext"]]
test_lm = test[["lm_label", "Fulltext"]]
valid_lm = valid[["lm_label", "Fulltext"]]
# save them to CSV in lm_path
lm_path = path/'lm'
lm_path.mkdir(exist_ok=True)
train_lm.to_csv(lm_path/"train.csv", header=False, index=False)
test_lm.to_csv(lm_path/"test.csv", header=False, index=False)
valid_lm.to_csv(lm_path/"valid.csv", header=False, index=False)

train_ds = TextDataset.from_csv(folder=clas_path, name="train", n_labels=20)
data_lm = TextLMDataBunch.from_csv(path=lm_path, train="train", valid="valid", n_labels=1)
data_clas = TextClasDataBunch.from_csv(path=clas_path, train="train", valid="valid", test="test",
                                      vocab = data_lm.train_ds.vocab, bs = 16, n_labels = 20)
                                      # Fine-tuning of Wikitext 103 LM based on my data

learn = RNNLearner.language_model(data_lm, pretrained_model=URLs.WT103)
learn.fit(2, 1e-2)
# Further Fine-tuning of LM to Target task data
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)
learn.save_encoder('lm_encoder')

# Using LM encoder on Target task: Multitask-Learning Classification Problem
learn = RNNLearner.classifier(data_clas)
#learn.load_encoder('lm_encoder')
# FileNotFoundError: [Errno 2] No such file or directory: 'ULMFiT/full_20_labels_same_data/clas/models/lm_encoder.pth'
learn.fit_one_cycle(1, 1e-3)
learn.save('full_classifier')
# ValueError: Target size (torch.Size([8, 20])) must be the same as input size (torch.Size([8, 2]))
