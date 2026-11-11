from agent import data
from agent.db import store

con = store.connect()
files = data.list_files()
train, test = data.split(files)
for f in files:
    s = data.load(f)
    split = "test" if f in test else "train"
    store.register_dataset(con, s, split)
    if split == "train":
        store.seed_ground_truth(con, s)
con.commit()
for t in ("datasets", "labels"):
    print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
print("train", len(train), "test", len(test))
