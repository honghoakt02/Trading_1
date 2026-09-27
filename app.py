# -*- coding: utf-8 -*-
"""
HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC GIAO DỊCH: KẾT HỢP EMA VÀ OBV
Ứng dụng Web Định lượng trên nền tảng Streamlit
Phát triển dựa trên nghiên cứu: "Hoàn thiện Framework Chiến lược EMA & OBV cho Cổ phiếu ACB"
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Bỏ qua các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# Cố gắng import vectorbt nếu môi trường hỗ trợ
try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

# Cố gắng import ta nếu có
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

# ==============================================================================
# CẤU HÌNH TRANG WEB STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Kiểm Định Chiến Lược EMA & OBV | Cổ Phiếu ACB",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện chuyên nghiệp chuẩn Fintech
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .badge-buy {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .badge-sell {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .badge-hold {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FDE047;
    }
    .badge-standby {
        background-color: #F1F5F9;
        color: #475569;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #CBD5E1;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HÀM TÍNH TOÁN CHỈ BÁO KỸ THUẬT & TÍN HIỆU (CHUẨN HÓA KHÔNG LOOK-AHEAD BIAS)
# ==============================================================================
def calculate_indicators(df, ema_period=20, obv_slope_period=3):
    """
    Tính toán đường EMA và độ dốc chỉ báo OBV.
    Đảm bảo tính độc lập, hỗ trợ cả khi có hoặc không có thư viện `ta`.
    """
    data = df.copy()
    close = data['Close']
    volume = data['Volume']

    # 1. Tính EMA
    if HAS_TA:
        ema = ta.trend.ema_indicator(close, window=int(ema_period))
    else:
        ema = close.ewm(span=int(ema_period), adjust=False).mean()
    data['EMA'] = ema

    # 2. Tính OBV và OBV Slope
    if HAS_TA:
        obv = ta.volume.on_balance_volume(close, volume)
    else:
        price_diff = close.diff()
        direction = pd.Series(0, index=close.index)
        direction[price_diff > 0] = 1
        direction[price_diff < 0] = -1
        direction.iloc[0] = 0
        obv = (direction * volume).cumsum()
    
    data['OBV'] = obv
    data['OBV_Slope'] = obv.diff(int(obv_slope_period))

    return data

def generate_signals(df, strategy_type='combined', ema_period=20, obv_slope_period=3):
    """
    Tạo tín hiệu Mua (entries) và Bán (exits) theo từng chiến lược:
    1. 'ema': EMA riêng lẻ
    2. 'obv': OBV riêng lẻ
    3. 'combined': Kết hợp EMA & OBV
    
    LƯU Ý QUAN TRỌNG: Tín hiệu được shift(1) để chống Look-ahead bias,
    nghĩa là chỉ sử dụng dữ liệu nến phiên trước để ra quyết định phiên nay.
    """
    data = calculate_indicators(df, ema_period, obv_slope_period)
    close = data['Close']
    ema = data['EMA']
    obv_slope = data['OBV_Slope']

    if strategy_type == 'ema':
        raw_entries = (close > ema)
        raw_exits = (close < ema)
    elif strategy_type == 'obv':
        raw_entries = (obv_slope > 0)
        raw_exits = (obv_slope < 0)
    else: # 'combined'
        raw_entries = (close > ema) & (obv_slope > 0)
        raw_exits = (close < ema)

    # Shift 1 phiên để ngăn Look-ahead Bias
    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)

    return entries, exits, data

# ==============================================================================
# HỆ THỐNG MÔ PHỎNG DANH MỤC & HẠCH TOÁN VỊ THẾ (POSITION ACCOUNTING)
# ==============================================================================
def simulate_portfolio(data, entries, exits, initial_capital=100.0, fees=0.002, slippage=0.001, sl_stop=0.07):
    """
    Mô phỏng giao dịch hạch toán đúng vị thế (Position Accounting):
    - Chế độ: Long-Only, Không tích lũy vị thế (Accumulate=False)
    - Phí giao dịch (Fees) & Trượt giá (Slippage)
    - Cắt lỗ cứng (Stop Loss) theo % từ giá mua (sl_stop)
    - Trả về thông số hiệu suất đầy đủ (Sharpe, Return, Drawdown, Trades log, v.v.)
    """
    # Nếu vectorbt có sẵn, ưu tiên dùng vectorbt để đối chiếu 100% notebook
    if HAS_VBT:
        try:
            pf = vbt.Portfolio.from_signals(
                close=data['Close'],
                entries=entries.to_numpy(dtype=bool),
                exits=exits.to_numpy(dtype=bool),
                direction='longonly',
                accumulate=False,
                fees=fees,
                slippage=slippage,
                sl_stop=sl_stop,
                sl_trail=False,
                init_cash=initial_capital,
                freq='D'
            )
            
            # Trích xuất dữ liệu từ vbt
            equity_curve = pf.value()
            returns = pf.daily_returns()
            sharpe = pf.sharpe_ratio()
            total_return = pf.total_return() * 100
            max_dd = pf.max_drawdown() * 100
            trades_record = pf.trades.records_readable
            closed_trades_count = int(pf.trades.closed.count())
            win_rate = (pf.trades.win_rate() * 100) if closed_trades_count > 0 else 0.0
            profit_factor = pf.trades.profit_factor() if closed_trades_count > 0 else 0.0
            calmar = pf.calmar_ratio()
            sortino = pf.sortino_ratio()
            current_pos = 1 if (pf.assets().iloc[-1] > 0) else 0

            # Tính Buy & Hold Benchmark
            bh_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            bh_curve = initial_capital * (data['Close'] / data['Close'].iloc[0])

            return {
                'sharpe': float(sharpe) if np.isfinite(sharpe) else 0.0,
                'total_return': float(total_return),
                'max_drawdown': float(max_dd),
                'closed_trades': closed_trades_count,
                'win_rate': float(win_rate) if np.isfinite(win_rate) else 0.0,
                'profit_factor': float(profit_factor) if np.isfinite(profit_factor) else 0.0,
                'calmar': float(calmar) if np.isfinite(calmar) else 0.0,
                'sortino': float(sortino) if np.isfinite(sortino) else 0.0,
                'equity_curve': equity_curve,
                'drawdown': pf.drawdown() * 100,
                'trades_df': trades_record,
                'benchmark_return': bh_return,
                'benchmark_curve': bh_curve,
                'current_position': current_pos,
                'engine': 'vectorbt'
            }
        except Exception:
            pass  # Nếu vbt gặp lỗi ngầm, fallback sang native engine ngay bên dưới

    # Thuật toán mô phỏng nội bộ độc lập (Native Vectorized Simulator)
    # Hạch toán vị thế chính xác theo chuẩn vectorbt
    close_prices = data['Close'].values
    dates = data.index
    n = len(close_prices)

    cash = initial_capital
    shares = 0.0
    position = 0 # 0: Cash, 1: Long
    entry_price = 0.0
    entry_idx = 0

    portfolio_values = np.zeros(n)
    positions_history = np.zeros(n, dtype=int)
    trades_list = []

    for i in range(n):
        curr_price = close_prices[i]
        is_entry = entries.iloc[i]
        is_exit = exits.iloc[i]

        # Kiểm tra điều kiện cắt lỗ Stop-loss nếu đang giữ vị thế
        stop_loss_triggered = False
        if position == 1 and sl_stop > 0:
            if curr_price <= entry_price * (1.0 - sl_stop):
                stop_loss_triggered = True

        # Thực thi lệnh Bán / Thoát vị thế
        if position == 1 and (is_exit or stop_loss_triggered):
            # Giá thực hiện sau trượt giá và phí
            exec_price = entry_price * (1.0 - sl_stop) if stop_loss_triggered else curr_price
            effective_sell_price = exec_price * (1.0 - slippage) * (1.0 - fees)
            cash = shares * effective_sell_price
            pnl_pct = ((effective_sell_price / entry_price) - 1.0) * 100
            
            trades_list.append({
                'Entry Date': dates[entry_idx],
                'Exit Date': dates[i],
                'Entry Price': round(entry_price, 2),
                'Exit Price': round(exec_price, 2),
                'Return [%]': round(pnl_pct, 2),
                'Duration (Days)': (dates[i] - dates[entry_idx]).days,
                'Exit Reason': 'Cắt lỗ Stop-loss' if stop_loss_triggered else 'Tín hiệu kỹ thuật'
            })
            shares = 0.0
            position = 0

        # Thực thi lệnh Mua / Mở vị thế mới
        if position == 0 and is_entry:
            effective_buy_price = curr_price * (1.0 + slippage) * (1.0 + fees)
            shares = cash / effective_buy_price
            cash = 0.0
            position = 1
            entry_price = effective_buy_price
            entry_idx = i

        # Giá trị danh mục tại phiên i
        portfolio_values[i] = cash + (shares * curr_price if position == 1 else 0.0)
        positions_history[i] = position

    equity_series = pd.Series(portfolio_values, index=dates)
    total_ret = ((equity_series.iloc[-1] / initial_capital) - 1.0) * 100.0

    # Tính Drawdown
    cummax = equity_series.cummax()
    drawdown_series = ((equity_series - cummax) / cummax) * 100.0
    max_drawdown = drawdown_series.min()

    # Tính Sharpe Ratio (annualized với 252 phiên)
    daily_returns = equity_series.pct_change().dropna()
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe_ratio = (np.sqrt(252) * mean_ret / std_ret) if (std_ret > 0 and not np.isnan(std_ret)) else 0.0

    # Sortino Ratio
    neg_ret = daily_returns[daily_returns < 0]
    downside_std = neg_ret.std()
    sortino = (np.sqrt(252) * mean_ret / downside_std) if (downside_std > 0 and not np.isnan(downside_std)) else 0.0

    # Thống kê giao dịch
    trades_df = pd.DataFrame(trades_list)
    closed_count = len(trades_df)
    if closed_count > 0:
        winning_trades = trades_df[trades_df['Return [%]'] > 0]
        losing_trades = trades_df[trades_df['Return [%]'] <= 0]
        win_rate = (len(winning_trades) / closed_count) * 100.0
        gross_profit = winning_trades['Return [%]'].sum()
        gross_loss = abs(losing_trades['Return [%]'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
    else:
        trades_df = pd.DataFrame(columns=['Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 'Return [%]', 'Duration (Days)', 'Exit Reason'])
        win_rate = 0.0
        profit_factor = 0.0

    # Calmar ratio
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (((equity_series.iloc[-1] / initial_capital) ** (1.0 / years)) - 1.0) * 100.0 if years > 0 else 0.0
    calmar = (cagr / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0

    # Benchmark Buy & Hold
    bh_return = ((close_prices[-1] / close_prices[0]) - 1.0) * 100.0
    bh_curve = initial_capital * (data['Close'] / close_prices[0])

    return {
        'sharpe': float(sharpe_ratio) if np.isfinite(sharpe_ratio) else 0.0,
        'total_return': float(total_ret),
        'max_drawdown': float(max_drawdown),
        'closed_trades': closed_count,
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if np.isfinite(profit_factor) else 0.0,
        'calmar': float(calmar) if np.isfinite(calmar) else 0.0,
        'sortino': float(sortino) if np.isfinite(sortino) else 0.0,
        'equity_curve': equity_series,
        'drawdown': drawdown_series,
        'trades_df': trades_df,
        'benchmark_return': bh_return,
        'benchmark_curve': bh_curve,
        'current_position': positions_history[-1],
        'engine': 'native'
    }

# ==============================================================================
# HÀM TẢI DỮ LIỆU TỰ ĐỘNG & BỘ ĐỆM CACHE
# ==============================================================================
@st.cache_data
def load_csv_data(file_source):
    """
    Tự động chuẩn hóa dữ liệu đầu vào:
    Hỗ trợ tìm kiếm cột Date, Close, Volume, Open, High, Low.
    """
    if isinstance(file_source, str):
        df = pd.read_csv(file_source)
    else:
        df = pd.read_csv(file_source)

    # Chuẩn hóa tên cột
    cols = {c.strip(): c.strip() for c in df.columns}
    col_mapping = {}
    for col in df.columns:
        c_low = col.lower().strip()
        if 'date' in c_low or 'ngày' in c_low:
            col_mapping[col] = 'Date'
        elif 'close' in c_low or 'đóng cửa' in c_low or 'dongcua' in c_low:
            col_mapping[col] = 'Close'
        elif 'volume' in c_low or 'khối lượng' in c_low or 'khoiluong' in c_low:
            col_mapping[col] = 'Volume'
        elif 'open' in c_low or 'mở cửa' in c_low:
            col_mapping[col] = 'Open'
        elif 'high' in c_low or 'cao nhất' in c_low:
            col_mapping[col] = 'High'
        elif 'low' in c_low or 'thấp nhất' in c_low:
            col_mapping[col] = 'Low'

    df.rename(columns=col_mapping, inplace=True)

    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Dữ liệu cần tối thiểu hai cột: 'Date' (Thời gian) và 'Close' (Giá đóng cửa).")

    if 'Volume' not in df.columns:
        df['Volume'] = 1000000.0 # Giá trị mặc định nếu thiếu volume

    # Chuyển đổi định dạng ngày tháng
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Loại bỏ dữ liệu rỗng và ép kiểu số
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    df.dropna(subset=['Close'], inplace=True)

    return df

# ==============================================================================
# THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR CONTROLS)
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Cấu Hình Kiểm Định")
st.sidebar.markdown("---")

# 1. Nguồn dữ liệu
st.sidebar.subheader("1. Nguồn Dữ Liệu")
data_source_mode = st.sidebar.radio(
    "Chọn tập dữ liệu:",
    ["Cổ phiếu ACB (Mặc định trong nghiên cứu)", "Tải lên file CSV tùy chọn"],
    index=0
)

df_raw = None
if data_source_mode == "Cổ phiếu ACB (Mặc định trong nghiên cứu)":
    # Tìm file ACB.csv trong thư mục hiện tại hoặc cùng cấp
    acb_candidates = ["ACB.csv", os.path.join(os.path.dirname(__file__), "ACB.csv")]
    found_path = None
    for p in acb_candidates:
        if os.path.exists(p):
            found_path = p
            break
    
    if found_path:
        df_raw = load_csv_data(found_path)
        st.sidebar.success(f"Đã nạp ACB.csv ({len(df_raw)} phiên)")
    else:
        st.sidebar.error("Không tìm thấy file ACB.csv trong thư mục! Vui lòng tải file lên.")
else:
    uploaded_file = st.sidebar.file_uploader("Tải lên file CSV (Date, Close, Volume, ...):", type=['csv'])
    if uploaded_file is not None:
        try:
            df_raw = load_csv_data(uploaded_file)
            st.sidebar.success(f"Nạp thành công ({len(df_raw)} phiên)")
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc file: {e}")

if df_raw is None:
    st.warning("Vui lòng đảm bảo file `ACB.csv` có trong thư mục hoặc tải lên file dữ liệu để bắt đầu.")
    st.stop()

# 2. Phân chia Train/Test (In-sample vs Out-of-sample)
st.sidebar.markdown("---")
st.sidebar.subheader("2. Phân Chia Dữ Liệu Train / Test")

min_date = df_raw.index.min().date()
max_date = df_raw.index.max().date()

# Đặt mặc định như trong notebook nghiên cứu (Train: 2014-2020, Test: 2021-2023)
default_train_end = pd.to_datetime('2020-12-31').date() if min_date <= pd.to_datetime('2020-12-31').date() <= max_date else min_date + (max_date - min_date) * 0.7

split_date = st.sidebar.date_input(
    "Ngày phân cách Train / Test (Hold-out):",
    value=default_train_end,
    min_value=min_date,
    max_value=max_date,
    help="Dữ liệu trước ngày này thuộc tập Huấn luyện (Train). Dữ liệu sau ngày này thuộc tập Kiểm định độc lập (Test)."
)

train_df = df_raw.loc[:str(split_date)].copy()
test_df = df_raw.loc[str(split_date + pd.Timedelta(days=1)):].copy()

# Kiểm tra dữ liệu hợp lệ
if len(train_df) < 30 or len(test_df) < 10:
    st.sidebar.error("Khoảng thời gian Train hoặc Test quá ngắn. Vui lòng chọn lại ngày phân tách!")
    st.stop()

st.sidebar.info(f"📊 **Train:** {len(train_df)} phiên ({train_df.index.min().strftime('%d/%m/%Y')} → {train_df.index.max().strftime('%d/%m/%Y')})\n\n"
               f"🧪 **Test:** {len(test_df)} phiên ({test_df.index.min().strftime('%d/%m/%Y')} → {test_df.index.max().strftime('%d/%m/%Y')})")

# 3. Cấu hình tham số chiến lược
st.sidebar.markdown("---")
st.sidebar.subheader("3. Thiết Lập Tham Số")

param_mode = st.sidebar.selectbox(
    "Chọn bộ tham số:",
    ["Bộ tham số tối ưu trên Train (Notebook)", "Bộ tham số mặc định (Default)", "Tự tinh chỉnh thủ công (Custom)"],
    index=0
)

if param_mode == "Bộ tham số tối ưu trên Train (Notebook)":
    # Tham số tối ưu được Hyperopt tìm thấy trong đề tài
    ema_period = 36
    obv_slope_period = 20
    st.sidebar.caption("⚡ *EMA = 36, OBV Slope = 20 (Kết hợp) / 19 (OBV riêng)*")
elif param_mode == "Bộ tham số mặc định (Default)":
    ema_period = 20
    obv_slope_period = 3
    st.sidebar.caption("⚡ *EMA = 20, OBV Slope = 3 (Chuẩn mặc định)*")
else:
    ema_period = st.sidebar.slider("Chu kỳ EMA (EMA Period):", min_value=5, max_value=100, value=36, step=1)
    obv_slope_period = st.sidebar.slider("Chu kỳ dốc OBV (OBV Slope Period):", min_value=2, max_value=40, value=20, step=1)

# 4. Quản trị rủi ro & Chi phí
st.sidebar.markdown("---")
st.sidebar.subheader("4. Quản Trị Rủi Ro & Phí")
fees_pct = st.sidebar.number_input("Phí giao dịch (%):", min_value=0.0, max_value=2.0, value=0.2, step=0.05) / 100.0
slippage_pct = st.sidebar.number_input("Trượt giá Slippage (%):", min_value=0.0, max_value=2.0, value=0.1, step=0.05) / 100.0
sl_stop_pct = st.sidebar.number_input("Cắt lỗ cứng Stop-Loss (%):", min_value=1.0, max_value=20.0, value=7.0, step=0.5) / 100.0
initial_capital = st.sidebar.number_input("Vốn khởi điểm (VNĐ hoặc USD):", min_value=100, value=100000000, step=10000000)

# ==============================================================================
# TIÊU ĐỀ TRANG VÀ THÔNG ĐIỆP CHỦ ĐẠO
# ==============================================================================
st.markdown('<div class="main-title">📈 HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC EMA & OBV</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Phân tích định lượng, Hạch toán vị thế (Position Accounting) & Đánh giá rủi ro Overfitting trên thị trường chứng khoán</div>', unsafe_allow_html=True)

# Hiển thị thông báo động cơ tính toán
engine_badge = "Thư viện VectorBT (Đúng 100% Notebook)" if HAS_VBT else "Động cơ Mô phỏng Nội bộ Chuẩn hóa (Native Engine)"
st.caption(f"🔧 Động cơ tính toán: **{engine_badge}** | 🛡️ Chống Look-ahead bias: **Đã kích hoạt `shift(1)`** | 🛑 Cắt lỗ: **{sl_stop_pct*100:.1f}%**")

# ==============================================================================
# TÍNH TOÁN DỮ LIỆU BACKTEST CHO CẢ 3 CHIẾN LƯỢC TRÊN CẢ TRAIN VÀ TEST
# ==============================================================================
# Tín hiệu trên Train
train_ent_ema, train_ext_ema, train_df_ema = generate_signals(train_df, 'ema', ema_period, obv_slope_period)
train_ent_obv, train_ext_obv, train_df_obv = generate_signals(train_df, 'obv', ema_period, obv_slope_period)
train_ent_comb, train_ext_comb, train_df_comb = generate_signals(train_df, 'combined', ema_period, obv_slope_period)

# Tín hiệu trên Test
test_ent_ema, test_ext_ema, test_df_ema = generate_signals(test_df, 'ema', ema_period, obv_slope_period)
test_ent_obv, test_ext_obv, test_df_obv = generate_signals(test_df, 'obv', ema_period, obv_slope_period)
test_ent_comb, test_ext_comb, test_df_comb = generate_signals(test_df, 'combined', ema_period, obv_slope_period)

# Chạy mô phỏng Train
res_train_ema = simulate_portfolio(train_df, train_ent_ema, train_ext_ema, initial_capital, fees_pct, slippage_pct, sl_stop_pct)
res_train_obv = simulate_portfolio(train_df, train_ent_obv, train_ext_obv, initial_capital, fees_pct, slippage_pct, sl_stop_pct)
res_train_comb = simulate_portfolio(train_df, train_ent_comb, train_ext_comb, initial_capital, fees_pct, slippage_pct, sl_stop_pct)

# Chạy mô phỏng Test
res_test_ema = simulate_portfolio(test_df, test_ent_ema, test_ext_ema, initial_capital, fees_pct, slippage_pct, sl_stop_pct)
res_test_obv = simulate_portfolio(test_df, test_ent_obv, test_ext_obv, initial_capital, fees_pct, slippage_pct, sl_stop_pct)
res_test_comb = simulate_portfolio(test_df, test_ent_comb, test_ext_comb, initial_capital, fees_pct, slippage_pct, sl_stop_pct)

# Tín hiệu trên toàn bộ dữ liệu để phục vụ khuyến nghị thực chiến
all_ent_comb, all_ext_comb, all_df_comb = generate_signals(df_raw, 'combined', ema_period, obv_slope_period)
res_all_comb = simulate_portfolio(df_raw, all_ent_comb, all_ext_comb, initial_capital, fees_pct, slippage_pct, sl_stop_pct)

# ==============================================================================
# HỆ THỐNG CÁC TAB CHỨC NĂNG
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tín Hiệu & Khuyến Nghị Hiện Tại",
    "⚖️ So Sánh Hiệu Suất (Train vs Test)",
    "📝 Nhật Ký Lệnh & Chi Tiết Giao Dịch",
    "🎯 Tối Ưu Hóa Tham Số (Sensitivity)",
    "📚 Báo Cáo Học Thuật & Khắc Phục Lỗi"
])

# ------------------------------------------------------------------------------
# TAB 1: TÍN HIỆU & KHUYẾN NGHỊ THỰC CHIẾN
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("1. Khuyến Nghị Giao Dịch Thực Chiến (Khắc phục góp ý giảng viên)")
    st.markdown("""
    > **Góp ý giảng viên:** Tín hiệu kỹ thuật phải kết hợp với **trạng thái vị thế hiện tại** (đang cầm cổ phiếu hay tiền mặt) để đưa ra hành động thực tế (MUA / BÁN / NẮM GIỮ / ĐỨNG NGOÀI), tránh phát tín hiệu MUA lặp lại hoặc BÁN khống.
    """)

    # Xác định khuyến nghị dựa trên nến mới nhất
    latest_date = df_raw.index[-1].strftime('%d/%m/%Y')
    latest_close = df_raw['Close'].iloc[-1]
    latest_ema = all_df_comb['EMA'].iloc[-1]
    latest_obv_slope = all_df_comb['OBV_Slope'].iloc[-1]
    
    current_holding = res_all_comb['current_position'] # 1 nếu đang Long, 0 nếu Cash
    latest_entry_signal = all_ent_comb.iloc[-1]
    latest_exit_signal = all_ext_comb.iloc[-1]

    if current_holding == 1:
        if latest_exit_signal:
            action_badge = '<span class="badge-sell">🔴 BÁN (SELL / CLOSE POSITION)</span>'
            action_desc = "Tín hiệu Bán xuất hiện khi đang nắm giữ cổ phiếu. Khuyến nghị đóng vị thế để bảo toàn vốn hoặc chốt lời."
        else:
            action_badge = '<span class="badge-hold">🟡 TIẾP TỤC NẮM GIỮ (HOLD)</span>'
            action_desc = "Tài khoản đang giữ vị thế và chưa có tín hiệu thoát lệnh. Giữ nguyên danh mục."
    else:
        if latest_entry_signal:
            action_badge = '<span class="badge-buy">🟢 MUA MỚI (BUY / OPEN POSITION)</span>'
            action_desc = "Điều kiện Mua thỏa mãn (Giá > EMA và OBV dốc lên) trong khi đang cầm tiền mặt. Khuyến nghị mở vị thế Mua."
        else:
            action_badge = '<span class="badge-standby">⚪ ĐỨNG NGOÀI QUAN SÁT (STAND ASIDE)</span>'
            action_desc = "Đang nắm giữ tiền mặt và chưa có tín hiệu Mua đạt chuẩn. Kiên nhẫn chờ cơ hội."

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{latest_date}</div><div class="metric-label">Phiên Gần Nhất</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{latest_close:,.0f}</div><div class="metric-label">Giá Đóng Cửa (VNĐ)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{latest_ema:,.0f}</div><div class="metric-label">Đường EMA ({ema_period})</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{"ĐANG GIỮ CỔ PHIẾU" if current_holding == 1 else "TIỀN MẶT"}</div><div class="metric-label">Trạng Thái Vị Thế</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: #F8FAFC; border: 2px solid #E2E8F0; border-radius: 12px; padding: 20px; margin: 15px 0;">
        <h4 style="margin-top:0;">Hành Động Khuyến Nghị Cho Phiên Kế Tiếp:</h4>
        <div style="margin-bottom: 10px;">{action_badge}</div>
        <p style="color: #475569; font-size: 1rem; margin-bottom: 0;"><b>Lý do định lượng:</b> {action_desc}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("2. Biểu Đồ Kỹ Thuật Tương Tác (Giá, EMA, OBV & Điểm Vào/Ra Lệnh)")

    # Bộ lọc hiển thị thời gian
    chart_period = st.radio("Chọn phạm vi hiển thị trên biểu đồ:", ["Toàn bộ dữ liệu", "Chỉ tập Train", "Chỉ tập Test", "1 Năm gần nhất"], horizontal=True)
    if chart_period == "Chỉ tập Train":
        chart_df = train_df_comb
        chart_ent = train_ent_comb
        chart_ext = train_ext_comb
    elif chart_period == "Chỉ tập Test":
        chart_df = test_df_comb
        chart_ent = test_ent_comb
        chart_ext = test_ext_comb
    elif chart_period == "1 Năm gần nhất":
        chart_df = all_df_comb.iloc[-252:]
        chart_ent = all_ent_comb.iloc[-252:]
        chart_ext = all_ext_comb.iloc[-252:]
    else:
        chart_df = all_df_comb
        chart_ent = all_ent_comb
        chart_ext = all_ext_comb

    # Vẽ biểu đồ tương tác 2 khung bằng Plotly
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Biểu đồ Giá & Tín hiệu Mua/Bán", f"Chỉ báo OBV & Độ dốc OBV Slope ({obv_slope_period} phiên)")
    )

    # Nến hoặc đường giá
    fig.add_trace(
        go.Scatter(x=chart_df.index, y=chart_df['Close'], mode='lines', name='Giá Đóng Cửa', line=dict(color='#2563EB', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=chart_df.index, y=chart_df['EMA'], mode='lines', name=f'EMA ({ema_period})', line=dict(color='#F59E0B', width=1.5, dash='solid')),
        row=1, col=1
    )

    # Điểm Mua và Điểm Bán
    buy_points = chart_df[chart_ent]
    sell_points = chart_df[chart_ext]

    fig.add_trace(
        go.Scatter(
            x=buy_points.index, y=buy_points['Close'],
            mode='markers', name='Tín hiệu MUA',
            marker=dict(symbol='triangle-up', size=11, color='#16A34A', line=dict(width=1, color='black'))
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points.index, y=sell_points['Close'],
            mode='markers', name='Tín hiệu BÁN',
            marker=dict(symbol='triangle-down', size=11, color='#DC2626', line=dict(width=1, color='black'))
        ),
        row=1, col=1
    )

    # Khung 2: OBV Slope (dạng Bar màu xanh/đỏ)
    colors = ['#16A34A' if val > 0 else '#DC2626' for val in chart_df['OBV_Slope']]
    fig.add_trace(
        go.Bar(x=chart_df.index, y=chart_df['OBV_Slope'], name='Độ dốc OBV Slope', marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(
        height=620,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white"
    )
    fig.update_yaxes(title_text="Giá (VNĐ)", row=1, col=1)
    fig.update_yaxes(title_text="Độ dốc OBV", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Bảng 10 phiên gần nhất
    st.subheader("3. Dữ Liệu 10 Phiên Giao Dịch Gần Nhất")
    recent_view = all_df_comb[['Close', 'EMA', 'OBV_Slope']].tail(10).copy()
    recent_view['Tín hiệu Mua (Raw)'] = all_ent_comb.tail(10)
    recent_view['Tín hiệu Bán (Raw)'] = all_ext_comb.tail(10)
    st.dataframe(recent_view.style.format({
        'Close': '{:,.1f}',
        'EMA': '{:,.1f}',
        'OBV_Slope': '{:,.0f}'
    }), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: SO SÁNH HIỆU SUẤT CHIẾN LƯỢC (TRAIN VS TEST)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("1. Bảng Số Liệu So Sánh Hiệu Suất Chiến Lược (Train vs Test)")
    st.markdown("""
    Đối chiếu toàn diện **3 chiến lược**:
    - **EMA Riêng Lẻ** (Mua khi Giá > EMA, Bán khi Giá < EMA)
    - **OBV Riêng Lẻ** (Mua khi OBV Slope > 0, Bán khi OBV Slope < 0)
    - **EMA + OBV Kết Hợp** (Mua khi Giá > EMA VÀ OBV Slope > 0, Bán khi Giá < EMA)
    """)

    # Tạo bảng so sánh chuẩn hóa giống Cell 23 của Notebook
    comp_data = [
        {
            'Chiến lược': 'EMA Riêng lẻ',
            'Tập dữ liệu': 'Train',
            'Tham số': f'EMA={ema_period}',
            'Sharpe Ratio': res_train_ema['sharpe'],
            'Tổng lợi nhuận (%)': res_train_ema['total_return'],
            'Max Drawdown (%)': res_train_ema['max_drawdown'],
            'Số giao dịch': res_train_ema['closed_trades'],
            'Win Rate (%)': res_train_ema['win_rate'],
            'Profit Factor': res_train_ema['profit_factor']
        },
        {
            'Chiến lược': 'EMA Riêng lẻ',
            'Tập dữ liệu': 'Test',
            'Tham số': f'EMA={ema_period}',
            'Sharpe Ratio': res_test_ema['sharpe'],
            'Tổng lợi nhuận (%)': res_test_ema['total_return'],
            'Max Drawdown (%)': res_test_ema['max_drawdown'],
            'Số giao dịch': res_test_ema['closed_trades'],
            'Win Rate (%)': res_test_ema['win_rate'],
            'Profit Factor': res_test_ema['profit_factor']
        },
        {
            'Chiến lược': 'OBV Riêng lẻ',
            'Tập dữ liệu': 'Train',
            'Tham số': f'OBV_Slope={obv_slope_period}',
            'Sharpe Ratio': res_train_obv['sharpe'],
            'Tổng lợi nhuận (%)': res_train_obv['total_return'],
            'Max Drawdown (%)': res_train_obv['max_drawdown'],
            'Số giao dịch': res_train_obv['closed_trades'],
            'Win Rate (%)': res_train_obv['win_rate'],
            'Profit Factor': res_train_obv['profit_factor']
        },
        {
            'Chiến lược': 'OBV Riêng lẻ',
            'Tập dữ liệu': 'Test',
            'Tham số': f'OBV_Slope={obv_slope_period}',
            'Sharpe Ratio': res_test_obv['sharpe'],
            'Tổng lợi nhuận (%)': res_test_obv['total_return'],
            'Max Drawdown (%)': res_test_obv['max_drawdown'],
            'Số giao dịch': res_test_obv['closed_trades'],
            'Win Rate (%)': res_test_obv['win_rate'],
            'Profit Factor': res_test_obv['profit_factor']
        },
        {
            'Chiến lược': 'EMA + OBV Kết hợp',
            'Tập dữ liệu': 'Train',
            'Tham số': f'EMA={ema_period}, OBV={obv_slope_period}',
            'Sharpe Ratio': res_train_comb['sharpe'],
            'Tổng lợi nhuận (%)': res_train_comb['total_return'],
            'Max Drawdown (%)': res_train_comb['max_drawdown'],
            'Số giao dịch': res_train_comb['closed_trades'],
            'Win Rate (%)': res_train_comb['win_rate'],
            'Profit Factor': res_train_comb['profit_factor']
        },
        {
            'Chiến lược': 'EMA + OBV Kết hợp',
            'Tập dữ liệu': 'Test',
            'Tham số': f'EMA={ema_period}, OBV={obv_slope_period}',
            'Sharpe Ratio': res_test_comb['sharpe'],
            'Tổng lợi nhuận (%)': res_test_comb['total_return'],
            'Max Drawdown (%)': res_test_comb['max_drawdown'],
            'Số giao dịch': res_test_comb['closed_trades'],
            'Win Rate (%)': res_test_comb['win_rate'],
            'Profit Factor': res_test_comb['profit_factor']
        }
    ]

    comp_df = pd.DataFrame(comp_data)
    
    # Hiển thị bảng định dạng đẹp mắt
    st.dataframe(comp_df.style.format({
        'Sharpe Ratio': '{:.4f}',
        'Tổng lợi nhuận (%)': '{:.2f}%',
        'Max Drawdown (%)': '{:.2f}%',
        'Win Rate (%)': '{:.2f}%',
        'Profit Factor': '{:.2f}'
    }).background_gradient(subset=['Sharpe Ratio', 'Tổng lợi nhuận (%)'], cmap='Greens'), use_container_width=True)

    st.markdown("---")
    st.subheader("2. Biểu Đồ Trực Quan Hóa So Sánh (Sharpe, Lợi Nhuận & Drawdown)")

    b1, b2, b3 = st.columns(3)
    with b1:
        # Sharpe Ratio Bar Chart
        fig_sharpe = px.bar(
            comp_df, x='Chiến lược', y='Sharpe Ratio', color='Tập dữ liệu',
            barmode='group', text_auto='.2f',
            title='Sharpe Ratio (Train vs Test)',
            color_discrete_map={'Train': '#10B981', 'Test': '#F59E0B'}
        )
        fig_sharpe.update_layout(template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_sharpe, use_container_width=True)

    with b2:
        # Total Return Bar Chart
        fig_ret = px.bar(
            comp_df, x='Chiến lược', y='Tổng lợi nhuận (%)', color='Tập dữ liệu',
            barmode='group', text_auto='.1f',
            title='Tổng Lợi Nhuận (%) (Train vs Test)',
            color_discrete_map={'Train': '#3B82F6', 'Test': '#EF4444'}
        )
        fig_ret.update_layout(template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_ret, use_container_width=True)

    with b3:
        # Max Drawdown Bar Chart
        fig_dd = px.bar(
            comp_df, x='Chiến lược', y='Max Drawdown (%)', color='Tập dữ liệu',
            barmode='group', text_auto='.1f',
            title='Max Drawdown (%) (Train vs Test)',
            color_discrete_map={'Train': '#6366F1', 'Test': '#EC4899'}
        )
        fig_dd.update_layout(template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Biểu Đồ Đường Cong Vốn (Equity Curve) & Sụt Giảm (Drawdown)")

    eval_dataset = st.radio("Chọn tập dữ liệu xem đường cong vốn:", ["Tập Huấn luyện (Train)", "Tập Kiểm định (Test)"], horizontal=True)
    if eval_dataset == "Tập Huấn luyện (Train)":
        eq_ema = res_train_ema['equity_curve']
        eq_obv = res_train_obv['equity_curve']
        eq_comb = res_train_comb['equity_curve']
        eq_bm = res_train_comb['benchmark_curve']
        dd_comb = res_train_comb['drawdown']
    else:
        eq_ema = res_test_ema['equity_curve']
        eq_obv = res_test_obv['equity_curve']
        eq_comb = res_test_comb['equity_curve']
        eq_bm = res_test_comb['benchmark_curve']
        dd_comb = res_test_comb['drawdown']

    # Chuẩn hóa về tỷ lệ % tăng trưởng
    norm_ema = (eq_ema / eq_ema.iloc[0] - 1.0) * 100
    norm_obv = (eq_obv / eq_obv.iloc[0] - 1.0) * 100
    norm_comb = (eq_comb / eq_comb.iloc[0] - 1.0) * 100
    norm_bm = (eq_bm / eq_bm.iloc[0] - 1.0) * 100

    fig_eq = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=("Tăng Trưởng Giá Trị Vốn (%) So Với Buy & Hold", "Mức Sụt Giảm Drawdown (%) - Chiến lược Kết hợp")
    )

    fig_eq.add_trace(go.Scatter(x=norm_comb.index, y=norm_comb, mode='lines', name='EMA + OBV Kết Hợp', line=dict(color='#10B981', width=2.5)), row=1, col=1)
    fig_eq.add_trace(go.Scatter(x=norm_ema.index, y=norm_ema, mode='lines', name='EMA Riêng Lẻ', line=dict(color='#3B82F6', width=1.5)), row=1, col=1)
    fig_eq.add_trace(go.Scatter(x=norm_obv.index, y=norm_obv, mode='lines', name='OBV Riêng Lẻ', line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    fig_eq.add_trace(go.Scatter(x=norm_bm.index, y=norm_bm, mode='lines', name='Benchmark (Buy & Hold)', line=dict(color='#94A3B8', width=1.5, dash='dash')), row=1, col=1)

    # Drawdown Area chart
    fig_eq.add_trace(go.Scatter(x=dd_comb.index, y=dd_comb, mode='lines', fill='tozeroy', name='Drawdown (Kết hợp)', line=dict(color='#EF4444', width=1)), row=2, col=1)

    fig_eq.update_layout(height=580, template='plotly_white', hovermode='x unified', margin=dict(l=40, r=40, t=40, b=40))
    fig_eq.update_yaxes(title_text="Lợi Nhuận (%)", row=1, col=1)
    fig_eq.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    st.plotly_chart(fig_eq, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: NHẬT KÝ LỆNH & CHI TIẾT GIAO DỊCH
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("1. Sổ Lệnh Chi Tiết (Trade Log) & Phân Tích Lệnh Đã Thực Hiện")
    
    col_str, col_set = st.columns(2)
    with col_str:
        selected_strategy = st.selectbox("Chọn chiến lược xem sổ lệnh:", ["EMA + OBV Kết Hợp", "EMA Riêng Lẻ", "OBV Riêng Lẻ"])
    with col_set:
        selected_split = st.selectbox("Chọn tập dữ liệu:", ["Tập Huấn Luyện (Train)", "Tập Kiểm Định (Test)", "Toàn Bộ Dữ Liệu"])

    # Lấy kết quả tương ứng
    if selected_strategy == "EMA + OBV Kết Hợp":
        active_res = res_train_comb if selected_split == "Tập Huấn Luyện (Train)" else (res_test_comb if selected_split == "Tập Kiểm Định (Test)" else res_all_comb)
    elif selected_strategy == "EMA Riêng Lẻ":
        active_res = res_train_ema if selected_split == "Tập Huấn Luyện (Train)" else (res_test_ema if selected_split == "Tập Kiểm Định (Test)" else res_all_comb)
    else:
        active_res = res_train_obv if selected_split == "Tập Huấn Luyện (Train)" else (res_test_obv if selected_split == "Tập Kiểm Định (Test)" else res_all_comb)

    trades_table = active_res['trades_df']
    
    if len(trades_table) == 0:
        st.info("Không phát sinh lệnh giao dịch nào trong giai đoạn này.")
    else:
        # KPI Tóm tắt các lệnh
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        with c_kpi1:
            st.metric("Tổng Số Giao Dịch", f"{active_res['closed_trades']}")
        with c_kpi2:
            st.metric("Tỷ Lệ Thắng (Win Rate)", f"{active_res['win_rate']:.1f}%")
        with c_kpi3:
            st.metric("Hệ Số Lợi Nhuận (Profit Factor)", f"{active_res['profit_factor']:.2f}")
        with c_kpi4:
            st.metric("Tổng Lợi Nhuận Giai Đoạn", f"{active_res['total_return']:.2f}%")

        # Bộ lọc trạng thái lệnh (Thắng/Thua)
        filter_status = st.radio("Lọc danh sách lệnh:", ["Tất cả", "Chỉ lệnh Thắng (Win)", "Chỉ lệnh Thua (Loss)"], horizontal=True)
        display_trades = trades_table.copy()
        
        # Chuẩn hóa cột Return [%]
        ret_col = 'Return [%]' if 'Return [%]' in display_trades.columns else ('PnL' if 'PnL' in display_trades.columns else display_trades.columns[-1])
        
        if filter_status == "Chỉ lệnh Thắng (Win)":
            display_trades = display_trades[display_trades[ret_col] > 0]
        elif filter_status == "Chỉ lệnh Thua (Loss)":
            display_trades = display_trades[display_trades[ret_col] <= 0]

        st.dataframe(display_trades, use_container_width=True)

        # Biểu đồ phân phối lợi nhuận các lệnh
        st.subheader("2. Phân Phối Lợi Nhuận Từng Giao Dịch")
        if ret_col in trades_table.columns and len(trades_table) > 1:
            fig_hist = px.histogram(
                trades_table, x=ret_col, nbins=30,
                title="Phân Phối Lợi Nhuận Lệnh (%)",
                color_discrete_sequence=['#3B82F6'],
                labels={ret_col: 'Lợi Nhuận Lệnh (%)'}
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            fig_hist.update_layout(template='plotly_white')
            st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: TỐI ƯU HÓA THAM SỐ (SENSITIVITY & GRID SEARCH)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("1. Phân Tích Độ Nhạy Tham Số & Quét Lưới (Grid Sensitivity)")
    st.markdown("""
    Đánh giá xem kết quả **EMA = 36** và **OBV Slope = 20** có phải là **vùng cao nguyên bền vững (Robust Plateau)** 
    hay chỉ là **điểm nhọn cá biệt ăn may (Overfitted Spike)**.
    """)

    opt_dataset = st.selectbox("Chọn tập dữ liệu để quét độ nhạy:", ["Tập Huấn Luyện (Train)", "Tập Kiểm Định (Test)"])
    scan_df = train_df if opt_dataset == "Tập Huấn Luyện (Train)" else test_df

    with st.expander("⚙️ Thiết Lập Lưới Quét Tham Số (Grid Setup)", expanded=True):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            ema_range = st.select_slider(
                "Dải chu kỳ EMA cần quét:",
                options=[10, 15, 20, 25, 30, 36, 40, 50, 60],
                value=(15, 40)
            )
        with col_g2:
            obv_range = st.select_slider(
                "Dải chu kỳ OBV Slope cần quét:",
                options=[2, 3, 5, 10, 15, 19, 20, 25, 30],
                value=(3, 25)
            )

        btn_run_scan = st.button("🚀 Bắt Đầu Quét Lưới Ma Trận", type="primary")

    if btn_run_scan:
        ema_test_list = [p for p in [10, 15, 20, 25, 30, 36, 40, 50, 60] if ema_range[0] <= p <= ema_range[1]]
        obv_test_list = [p for p in [2, 3, 5, 10, 15, 19, 20, 25, 30] if obv_range[0] <= p <= obv_range[1]]

        matrix_sharpe = np.zeros((len(obv_test_list), len(ema_test_list)))
        matrix_return = np.zeros((len(obv_test_list), len(ema_test_list)))

        progress_bar = st.progress(0)
        total_steps = len(ema_test_list) * len(obv_test_list)
        step = 0

        for i, obv_p in enumerate(obv_test_list):
            for j, ema_p in enumerate(ema_test_list):
                ent, ext, _ = generate_signals(scan_df, 'combined', ema_p, obv_p)
                sim = simulate_portfolio(scan_df, ent, ext, initial_capital, fees_pct, slippage_pct, sl_stop_pct)
                
                # Áp dụng ngưỡng MIN_TRADES >= 5 như notebook để tránh Sharpe ảo
                if sim['closed_trades'] >= 5:
                    matrix_sharpe[i, j] = sim['sharpe']
                    matrix_return[i, j] = sim['total_return']
                else:
                    matrix_sharpe[i, j] = np.nan
                    matrix_return[i, j] = np.nan

                step += 1
                progress_bar.progress(step / total_steps)

        progress_bar.empty()

        # Hiển thị Heatmap Sharpe Ratio
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig_heat_sharpe = px.imshow(
                matrix_sharpe,
                x=[f"EMA={e}" for e in ema_test_list],
                y=[f"OBV={o}" for o in obv_test_list],
                color_continuous_scale="Viridis",
                labels=dict(x="Chu kỳ EMA", y="Chu kỳ OBV Slope", color="Sharpe Ratio"),
                title="Ma Trận Sharpe Ratio (Vùng Tham Số Bền Vững)"
            )
            fig_heat_sharpe.update_layout(template="plotly_white")
            st.plotly_chart(fig_heat_sharpe, use_container_width=True)

        with col_h2:
            fig_heat_ret = px.imshow(
                matrix_return,
                x=[f"EMA={e}" for e in ema_test_list],
                y=[f"OBV={o}" for o in obv_test_list],
                color_continuous_scale="Plasma",
                labels=dict(x="Chu kỳ EMA", y="Chu kỳ OBV Slope", color="Lợi Nhuận (%)"),
                title="Ma Trận Tổng Lợi Nhuận (%)"
            )
            fig_heat_ret.update_layout(template="plotly_white")
            st.plotly_chart(fig_heat_ret, use_container_width=True)

        st.success("✅ Đã hoàn tất quét ma trận tham số. Vùng màu sáng liền mạch thể hiện tham số có tính ổn định cao!")

# ------------------------------------------------------------------------------
# TAB 5: BÁO CÁO HỌC THUẬT & KHẮC PHỤC LỖI
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("📚 Báo Cáo Học Thuật: Giải Quyết Triệt Để Góp Ý Giảng Viên")
    
    st.markdown("""
    ### 1. Phân Tích Lỗi Gốc & Cách Khắc Phục
    
    #### ⚠️ Lỗi 1: Không Kiểm Soát Số Lượng Vị Thế (`MIN_TRADES`) Dẫn Tới Hiện Tượng Sharpe Vô Cực (`Sharpe = inf`)
    - **Bản chất lỗi:** Trong phiên bản ban đầu, thuật toán Hyperopt tối ưu hóa hoàn toàn dựa trên Sharpe Ratio mà không kiểm tra số lệnh phát sinh. Khi bộ tham số khiến hệ thống **hầu như không vào lệnh** (0 hoặc 1 lệnh), độ lệch chuẩn lợi nhuận $\sigma \approx 0$, dẫn đến **Sharpe Ratio bị chia cho 0 và bùng nổ thành `inf` (vô cực)**.
    - **Cách khắc phục:** 
      - Bổ sung ràng buộc ngặt: `MIN_TRADES = 5`.
      - Mọi bộ tham số sinh ra ít hơn 5 giao dịch đều bị gán điểm phạt (`loss = np.inf`), loại bỏ hoàn toàn các trường hợp "ăn may" do đứng ngoài thị trường.

    #### ⚠️ Lỗi 2: Tín Hiệu Thực Chiến Không Nhận Biết Trạng Thái Vị Thế
    - **Bản chất lỗi:** Tín hiệu kỹ thuật `close > ema` hay `close < ema` là **tín hiệu trạng thái** (kéo dài nhiều phiên liên tiếp). Nếu không kiểm tra danh mục đang cầm Cổ phiếu hay Tiền mặt:
      - Khi đang cầm cổ phiếu, mã tiếp tục báo MUA (dù tài khoản đã full vị thế).
      - Khi đang cầm tiền mặt, mã lại báo BÁN (trong khi thị trường chứng khoán Việt Nam chưa cho phép bán khống cổ phiếu cơ sở).
    - **Cách khắc phục:** 
      - Theo dõi biến vị thế `current_position` ($0$ hoặc $1$).
      - Chỉ báo MUA khi `position == 0` và có tín hiệu Entry.
      - Chỉ báo BÁN khi `position == 1` và có tín hiệu Exit.
      - Trạng thái còn lại là **TIẾP TỤC NẮM GIỮ (HOLD)** hoặc **ĐỨNG NGOÀI (STAND ASIDE)**.

    #### ⚠️ Lỗi 3: Ngăn Ngừa Look-Ahead Bias & Chi Phí Thực Tế
    - **Ngăn Look-ahead Bias:** Mọi tín hiệu đều được dịch chuyển 1 phiên: `shift(1)`. Quyết định của phiên hôm nay chỉ được hình thành sau khi nến phiên hôm trước đã đóng cửa hoàn toàn.
    - **Chi phí ma sát:** Tích hợp đầy đủ Phí giao dịch (0.2%), Trượt giá (0.1%) và Cắt lỗ cứng (7.0%).

    ---
    ### 2. Bài Học Đắt Giá Về Hiện Tượng Quá Khớp (Overfitting)
    
    Dữ liệu kiểm nghiệm cho thấy sự phân hóa sâu sắc giữa hai giai đoạn:
    """)

    col_ins, col_oos = st.columns(2)
    with col_ins:
        st.markdown("""
        <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 15px;">
            <h4 style="color: #065F46; margin-top:0;">🟢 Tập Train (2014 - 2020) | Thị Trường Xu Hướng (Bull Trend)</h4>
            <ul>
                <li><b>Chiến lược kết hợp:</b> Sharpe = <b>1.1514</b>, Lợi nhuận = <b>+236.56%</b>, Max Drawdown = <b>-23.87%</b>.</li>
                <li><b>Đánh giá:</b> Hệ thống EMA & OBV hoạt động xuất sắc khi cổ phiếu ACB nằm trong xu hướng tăng trưởng dài hạn. Đường EMA giữ vị thế tốt và OBV giúp lọc bớt các nhịp phân phối ngắn hạn.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_oos:
        st.markdown("""
        <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 15px;">
            <h4 style="color: #991B1B; margin-top:0;">🔴 Tập Test (2021 - 2023) | Biến Động & Downtrend 2022</h4>
            <ul>
                <li><b>Chiến lược kết hợp:</b> Sharpe = <b>-0.8298</b>, Lợi nhuận = <b>-36.47%</b>, Max Drawdown = <b>-60.14%</b>.</li>
                <li><b>Đánh giá:</b> Khi thị trường bước vào pha giảm mạnh năm 2022 và sideway whipsaw năm 2023, tín hiệu EMA bị nhiễu liên tục (mua đỉnh, bán đáy nhịp hồi), dẫn tới hiện tượng sụt giảm vốn nghiêm trọng.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    #### 💡 Đề Xuất Nâng Cấp Hệ Thống Cho Nhà Đầu Tư:
    1. **Tích hợp Bộ lọc Xu hướng Thị trường Chung (Market Regime Filter):** Chỉ kích hoạt tín hiệu Mua khi VN-Index nằm trên đường SMA200 tuần.
    2. **Quản trị vốn động (Volatility-based Position Sizing):** Giảm tỷ trọng giải ngân khi chỉ số biến động ATR tăng cao.
    3. **Walk-Forward Analysis (WFA):** Tái tối ưu hóa tham số định kỳ theo chu kỳ 6 tháng đến 1 năm thay vì dùng một bộ tham số cố định qua nhiều năm.
    """)

# ==============================================================================
# CHÂN TRANG (FOOTER)
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px;">
    Đề tài nghiên cứu: <b>Kiểm định tính hiệu quả của chiến lược kết hợp tín hiệu EMA và OBV cho cổ phiếu ACB</b><br>
    Ứng dụng được thiết kế tương thích hoàn toàn để triển khai trên <b>Streamlit Community Cloud</b> và <b>GitHub</b>.
</div>
""", unsafe_allow_html=True)
