import numpy as np
import pandas as pd
import keras.backend as K

from keras.layers import Dense, Conv2D, Conv1D, Input, Flatten, Softmax
from keras.layers import MaxPool2D, AvgPool2D, MaxPool1D
from keras.models import Model, load_model

from sklearn.model_selection import GroupKFold


col_read = ['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'Dir',
            'NflId', 'Season', 'NflIdRusher', 'PlayDirection',
            'PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Yards']
train = pd.read_csv('./input/train.csv', usecols=col_read)

# ============================================================================== Preprocess
map_abbr = {
    'ARZ': 'ARI',
    'BLT': 'BAL',
    'CLV': 'CLE',
    'HST': 'HOU'
}
train['PossessionTeam'] = train['PossessionTeam'].replace(map_abbr)

# === identify offense team
train.loc[(train['Team'] == 'home') & (train['PossessionTeam'] == train['HomeTeamAbbr']), 'Side'] = 'offense'
train.loc[(train['Team'] == 'away') & (train['PossessionTeam'] == train['VisitorTeamAbbr']), 'Side'] = 'offense'
train['Side'] = train['Side'].fillna('defense')

# === speed
train['Dir'] = train['Dir'].fillna(0)
train['Sx'] = np.cos(train['Dir'] / 180 * np.pi) * train['S']
train['Sy'] = np.sin(train['Dir'] / 180 * np.pi) * train['S']

# === (rusher, defenser) pair
n_member = 11
col_feature = ['X', 'Y', 'Sx', 'Sy']
arr_rush = train.loc[train['NflId'] == train['NflIdRusher'], col_feature].values
arr_def = train.loc[train['Side'] == 'defense', col_feature].values
arr_delta = arr_def - np.repeat(arr_rush, n_member, axis=0)

# delta_X, delta_Y, delta_Sx, delta_Sy, Sx, Sy -> shape = (n_play, n_feature)
arr_feature = np.concatenate([arr_delta, arr_def[:,2:]], axis=1)
# use GameId in GroupKFold
arr_game_id = train.loc[train['Side'] == 'defense', 'GameId'].values


# ============================================================================== Model
def crps(y_true, y_pred):
    y_pred = K.clip(K.cumsum(y_pred, axis=1), min_value=0, max_value=1)
    return K.mean(K.square(y_true - y_pred))


def build_model():
    inp = Input(shape=(11,6))
    x = Conv1D(128, kernel_size=1, strides=1, activation='relu')(inp)
    x = Conv1D(160, kernel_size=1, strides=1, activation='relu')(x)
    x = Conv1D(128, kernel_size=1, strides=1, activation='relu')(x)
    x = MaxPool1D(pool_size=11)(x)
    x = Flatten()(x)
    out = Dense(199, activation='softmax')(x)

    model = Model(input=inp, output=out)
    model.compile(optimizer='adam', loss=crps)
    print(model.summary())
    return model


model = build_model()

x_train = np.reshape(arr_feature, [-1, 11, 6])
y_train = np.zeros([len(x_train), 199])
for i, yard in enumerate(train.loc[::22, 'Yards']):
    y_train[i, yard+99:] = 1

oof = np.zeros_like(y_train)
arr_group = arr_game_id[::11]
folds = GroupKFold(n_splits=3)
for f, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train, arr_group)):
    x_trn = x_train[trn_idx]
    y_trn = y_train[trn_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    model.fit(
        x_trn,
        y_trn,
        batch_size=64,
        epochs=1,
        verbose=1,
        validation_data=[x_val, y_val]
    )

    oof[val_idx] = model.predict(x_val)
    model.save(f'model_fold{f}.h5')

oof = np.clip(np.cumsum(oof, axis=1), a_min=0, a_max=1)
cv_score = np.mean((y_train - oof)**2)
print(cv_score)

models = []
for f in folds.n_splits:
    models.append(load_model(f'model_fold{f}.h5'), custom_object={'crps': crps})

y_pred = np.mean([m.predict(x_test) for m in models])

