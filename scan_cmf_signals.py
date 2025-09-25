import sys
import os
import pandas as pd
import argparse
from strategies.cmf_peaks_troughs_strategy import ConfigurableCMFPeaksTroughsStrategy
from data_sources import data_source_manager
import requests  # Add this import

def get_data_for_symbol(symbol, period="6mo"):
    """
    Fetch data using the same logic as the app (best source for symbol).
    """
    return data_source_manager.fetch_data(symbol, period)

def send_ntfy_notification(topic, message):
    """
    Send a notification to ntfy.sh with the given topic and message.
    """
    url = f"https://ntfy.sh/{topic}"
    try:
        response = requests.post(url, data=message.encode('utf-8'))
        if response.status_code != 200:
            print(f"Failed to send ntfy notification: {response.status_code}")
    except Exception as e:
        print(f"Error sending ntfy notification: {e}")

def scan_symbols(symbol_df, recent_period=5):
    strategy = ConfigurableCMFPeaksTroughsStrategy()
    found = []
    for _, row in symbol_df.iterrows():
        symbol = row['symbol']
        period = row.get('period', '6mo')
        smooth = row.get('smooth', 5)
        prominence = row.get('prominence', 0.05)
        try:
            df = get_data_for_symbol(symbol, period=period)
            if df is None or len(df) < recent_period:
                continue
            has_signal, signals = strategy.scan_signals_in_period(
                df,
                recent_period=recent_period,
                period=period,
                wma_period=smooth,
                prominence=prominence
            )
            if has_signal:
                found.append({'symbol': symbol, 'signals': signals})
                print(f"{symbol}: {signals}")
                # Send ntfy notification for each signal found
                send_ntfy_notification(
                    topic="claylin_alert",  # You can customize this topic
                    message=f"{symbol}: {signals}"
                )
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    print(f"\nSymbols with signals in last {recent_period} bars: {len(found)}")
    for entry in found:
        print(entry['symbol'])
    return found

def main():
    parser = argparse.ArgumentParser(description="Scan symbols for recent CMF signals.")
    parser.add_argument("symbol_list", help="CSV file with columns: symbol, period, smooth, prominence")
    parser.add_argument("--recent", type=int, default=5, help="Recent period (bars) to scan for signals")
    args = parser.parse_args()

    symbol_df = pd.read_csv(args.symbol_list, comment='#')
    scan_symbols(symbol_df, recent_period=args.recent)

if __name__ == "__main__":
    main()
