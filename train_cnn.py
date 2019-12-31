import numpy as np
import pandas as pd
import keras.backend as K

from keras.layers import Dense, Input, Flatten, Lambda, BatchNormalization, Dropout, Add
from keras.layers import Conv2D, Conv1D, MaxPool2D, AvgPool2D, MaxPool1D, AvgPool1D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold


# ============================================================================== Load Data
col_read = ['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'Dir', 'Dis',
            'NflId', 'Season', 'NflIdRusher', 'PlayDirection',
            'PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Yards']
train = pd.read_csv('./input/train.csv', usecols=col_read)

# ============================================================================== Parameter
np.random.seed(2020)

# ============================================================================== Preprocess
map_abbr = {
    'ARZ': 'ARI',
    'BLT': 'BAL',
    'CLV': 'CLE',
    'HST': 'HOU'
}
train['PossessionTeam'] = train['PossessionTeam'].replace(map_abbr)

# === offense/defense
train.loc[(train['Team'] == 'home') & (train['PossessionTeam'] == train['HomeTeamAbbr']), 'Side'] = 'offense'
train.loc[(train['Team'] == 'away') & (train['PossessionTeam'] == train['VisitorTeamAbbr']), 'Side'] = 'offense'
train.loc[train['NflId'] == train['NflIdRusher'], 'Side'] = 'rush'
train['Side'] = train['Side'].fillna('defense')

# === standardize
FIELD_LENGTH = 120
FIELD_WIDTH = 160/3
ENDZONE_LENGTH = 10
idx_left = train['PlayDirection'] == 'left'
train.loc[idx_left, 'Y'] = FIELD_WIDTH - train.loc[idx_left, 'Y']
train.loc[idx_left, 'X'] = FIELD_LENGTH - train.loc[idx_left, 'X']
# train['X'] = train['X'] - ENDZONE_LENGTH
train.loc[idx_left, 'Dir'] = (train.loc[idx_left, 'Dir'] + 180) % 360

# === speed
# train['S'] = train['Dis'] * 10
train['Dir'] = train['Dir'].fillna(0)
train['S_X'] = np.sin(train['Dir'] / 180 * np.pi) * train['S']
train['S_Y'] = np.cos(train['Dir'] / 180 * np.pi) * train['S']

# === (defender, offender) pair
N_DEFENDER = 11
N_OFFENDER = 10     # exclude 'rusher'
col_delta = ['X', 'Y', 'S_X', 'S_Y']

arr_rush = train.loc[train['Side'] == 'rush', col_delta].values
arr_def = train.loc[train['Side'] == 'defense', col_delta].values
arr_off = train.loc[train['Side'] == 'offense', col_delta].values

arr_off_r = np.reshape(arr_off, [-1, N_OFFENDER, len(col_delta)])
arr_off_r = np.repeat(arr_off_r, N_DEFENDER, axis=0)
arr_off_r = np.reshape(arr_off_r, [-1, len(col_delta)])

delta_def_off = np.repeat(arr_def, N_OFFENDER, axis=0) - arr_off_r
delta_def_rush = arr_def - np.repeat(arr_rush, N_DEFENDER, axis=0)
delta_def_rush = np.repeat(delta_def_rush, N_OFFENDER, axis=0)

arr_feature = np.concatenate([
    delta_def_off,
    delta_def_rush,
    np.repeat(arr_def[:, 2:], N_OFFENDER, axis=0)
], axis=1)


# ============================================================================== Model
def crps(y_true, y_pred):
    y_pred = K.clip(K.cumsum(y_pred, axis=1), min_value=0, max_value=1)
    return K.mean(K.square(y_true - y_pred))


def build_model():
    inp = Input(shape=(11, 10, 10))

    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(inp)
    x = Conv2D(160, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)

    x_max = MaxPool2D(pool_size=(1, 10))(x)
    x_max = Lambda(lambda t: 0.3 * t)(x_max)
    x_avg = AvgPool2D(pool_size=(1, 10))(x)
    x_avg = Lambda(lambda t: 0.7 * t)(x_avg)
    x = Add()([x_max, x_avg])
    x = Lambda(lambda t: K.squeeze(t, 2))(x)
    x = BatchNormalization()(x)

    x = Conv1D(160, kernel_size=1, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(96, kernel_size=1, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(96, kernel_size=1, strides=1, activation='relu')(x)
    x = BatchNormalization()(x)

    x_max = MaxPool1D(pool_size=11)(x)
    x_max = Lambda(lambda t: 0.3 * t)(x_max)
    x_avg = AvgPool1D(pool_size=11)(x)
    x_avg = Lambda(lambda t: 0.7 * t)(x_avg)
    x = Add()([x_max, x_avg])
    x = Flatten()(x)

    x = Dense(96, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.3)(x)

    out = Dense(199, activation='softmax')(x)

    model = Model(input=inp, output=out)
    print(model.summary())
    return model


x_train = np.reshape(arr_feature, [-1, N_DEFENDER, N_OFFENDER, arr_feature.shape[1]])
y_train = np.zeros([len(x_train), 199])
for i, yard in enumerate(train.loc[::22, 'Yards']):
    y_train[i, yard+99:] = 1

oof = np.zeros_like(y_train)
arr_group = train.loc[::22, 'GameId'].values
folds = GroupKFold(n_splits=3)
n_bags = 3
bag_ratio = 0.8
for f, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train, arr_group)):
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    for b in range(n_bags):
        print(f'fold°{f} - bag°{b}')
        bag_idx = np.random.choice(trn_idx, int(bag_ratio * len(trn_idx)), replace=True)
        x_trn = x_train[bag_idx]
        y_trn = y_train[bag_idx]

        model = build_model()
        model.compile(optimizer='adam', loss=crps)
        es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)

        model.fit(
            x_trn,
            y_trn,
            batch_size=64,
            epochs=50,
            verbose=2,
            validation_data=[x_val, y_val],
            callbacks=[es]
        )

        oof[val_idx] += model.predict(x_val) / n_bags
        model.save(f'model_fold{f}_bag{b}.h5')

oof = np.clip(np.cumsum(oof, axis=1), a_min=0, a_max=1)
idx_cv = train.loc[::22, 'Season'] == 2018
cv_score = np.mean((y_train[idx_cv] - oof[idx_cv])**2)
print(cv_score)

# ============================================================================== Prediction
# models = []
# for f in folds.n_splits:
#     models.append(load_model(f'model_fold{f}.h5'), custom_object={'crps': crps})
#
# y_pred = np.mean([m.predict(x_test) for m in models])



