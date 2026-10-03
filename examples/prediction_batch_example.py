import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXAMPLES_DIR))
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device=None, max_context=512)

# 3. Prepare Data
# Uses data/XSHG_5min_600977.csv if present, otherwise the bundled sample in the same 5-min A-share format
DATA_FILE = os.path.join(os.path.dirname(EXAMPLES_DIR), "data", "XSHG_5min_600977.csv")
if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(os.path.dirname(EXAMPLES_DIR), "data", "sample_a_share_5min.csv")
df = pd.read_csv(DATA_FILE)
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

dfs = []
xtsp = []
ytsp = []
for i in range(5):
    idf = df.loc[(i*400):(i*400+lookback-1), ['open', 'high', 'low', 'close', 'volume', 'amount']]
    i_x_timestamp = df.loc[(i*400):(i*400+lookback-1), 'timestamps']
    i_y_timestamp = df.loc[(i*400+lookback):(i*400+lookback+pred_len-1), 'timestamps']

    dfs.append(idf)
    xtsp.append(i_x_timestamp)
    ytsp.append(i_y_timestamp)

pred_df = predictor.predict_batch(
    df_list=dfs,
    x_timestamp_list=xtsp,
    y_timestamp_list=ytsp,
    pred_len=pred_len,
)

for i, p in enumerate(pred_df):
    print(f"Series {i} forecast head:")
    print(p.head())
