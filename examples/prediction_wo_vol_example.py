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

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax.set_ylabel('Close Price', fontsize=14)
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(EXAMPLES_DIR), "outputs", os.path.basename(__file__).replace(".py", ".png"))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    print(f"Chart saved to {out_png}")
    plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

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

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback+pred_len-1]

# visualize
plot_prediction(kline_df, pred_df)

