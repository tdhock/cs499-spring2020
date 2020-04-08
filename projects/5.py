import tensorflow as tf
import numpy as np
import plotnine as p9 #conda install -c conda-forge plotnine
import pandas as pd
from metrics import ExactAUC, testN, testConst

spam = np.genfromtxt("spam.data", delimiter=" ")
X_unscaled = spam[:,:-1]
X = (X_unscaled-X_unscaled.mean(axis=0))/X_unscaled.std(axis=0)
y = spam[:,-1]
n_folds = 5
test_fold_ids = np.arange(n_folds)
np.random.seed(0)
test_fold_vec = np.random.permutation(np.tile(test_fold_ids, len(y))[:len(y)])

test_fold = 0
set_dict = {
    "test": test_fold_vec == test_fold,
    "train": test_fold_vec != test_fold,
    }
X_dict = {}
y_dict = {}
for set_name, is_set in set_dict.items():
    X_dict[set_name] = X[is_set, :]
    y_dict[set_name] = y[is_set]

inputs = tf.keras.Input(shape=(X.shape[1],))
hidden = tf.keras.layers.Dense(
    100, activation="sigmoid", use_bias=False)(inputs)
outputs = tf.keras.layers.Dense(
    1, activation="sigmoid", use_bias=False)(hidden)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="spam_model")

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=["accuracy", ExactAUC, testN, testConst])

n_epochs = 100
history = model.fit(
    X_dict["train"], y_dict["train"],
    epochs=n_epochs,
    verbose=2,
    validation_split=0.5)
test_scores = dict(zip(
    model.metrics_names,
    model.evaluate(X_dict["test"], y_dict["test"])))
history.history

history_wide = pd.DataFrame(history.history)
history_wide["epoch"] = np.arange(n_epochs)+1
history_tall = pd.melt(history_wide, id_vars="epoch")
history_tall_sets = pd.concat([
    history_tall,
    history_tall["variable"].str.extract("(?P<prefix>val_|)(?P<metric>.*)"),
    ], axis=1)
history_tall_sets["set"] = history_tall_sets["prefix"].apply(
    lambda x: "validation" if x == "val_" else "train")
best_i = history_wide["val_loss"].idxmin()
best_epoch = history_wide.loc[ best_i, "epoch" ]
best_tall = history_tall_sets.loc[ history_tall_sets["epoch"]==best_epoch ]

gg = p9.ggplot(history_tall_sets, p9.aes(x="epoch", y="value", color="set"))+\
    p9.geom_line()+\
    p9.theme_bw()+\
    p9.facet_grid("metric ~ .", scales="free")+\
    p9.geom_point(data=best_tall)+\
    p9.theme(
        facet_spacing={'right': 0.75},
        panel_spacing=0)
gg.save("5-acc-loss.png", width=5, height=5)
