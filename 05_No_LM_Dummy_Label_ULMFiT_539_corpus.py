from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
import pandas as pd
from sklearn.model_selection import train_test_split

path = Path('ULMFiT/full_20_labels_same_data2')
path.mkdir(exist_ok=True)
torch.cuda.set_device(1)
df = pd.read_csv("dataset/full_corpus.csv")

# Train test split
train, valid = train_test_split(df, test_size = 0.3)
test, valid = train_test_split(valid, test_size = 0.6)

# save them to CSV in path
train.to_csv(path/"train.csv", header=False, index=False)
test.to_csv(path/"test.csv", header=False, index=False)
valid.to_csv(path/"valid.csv", header=False, index=False)

#_____________________________________
### TextDataset & DataBunch
train_ds = TextDataset.from_csv(folder=path, name="train", n_labels=20)
data_lm = TextLMDataBunch.from_csv(path=path, train="train", valid="valid")
data_clas = TextClasDataBunch.from_csv(path=path, train="train", valid="valid", test="test",
                                      vocab = data_lm.train_ds.vocab, bs = 16, n_labels = 20)
                                      
# Fine-tuning of Wikitext 103 LM based on my data
learn = RNNLearner.language_model(data_lm, pretrained_model=URLs.WT103)
learn.fit(2, 1e-2)
# Further Fine-tuning of LM to Target task data
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)
#learn.save_encoder('lm_encoder')

# Using LM encoder on Target task: Multitask-Learning Classification Problem
learn = RNNLearner.classifier(data_clas)
learn.fit_one_cycle(1, 1e-3)
#learn.save('full_classifier')
# ValueError: Target size (torch.Size([8, 20])) must be the same as input size (torch.Size([8, 2]))
