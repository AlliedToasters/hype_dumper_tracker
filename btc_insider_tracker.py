import requests
import json
from typing import Dict, Any, List
import time
from datetime import datetime
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns

class HyperliquidShortTracker:
    def __init__(self, wallet_address: str):
        self.base_url = "https://api.hyperliquid.xyz/info"
        self.wallet_address = wallet_address
        self.snapshots_dir = "btc_short_snapshots"
        
        # Create snapshots directory if it doesn't exist
        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)
    
    def get_clearinghouse_state(self) -> Dict[str, Any]:
        """Get perpetuals account summary including positions"""
        payload = {
            "type": "clearinghouseState",
            "user": self.wallet_address
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching clearinghouse state: {e}")
            return {}
    
    def get_meta_and_asset_contexts(self) -> tuple[List[Dict], List[Dict]]:
        """Get perpetuals metadata and current prices"""
        payload = {
            "type": "metaAndAssetCtxs"
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data[0]["universe"], data[1]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching meta and asset contexts: {e}")
            return [], []
    
    def extract_btc_position(self, clearinghouse_data: Dict[str, Any], asset_contexts: List[Dict]) -> Dict[str, Any]:
        """Extract BTC position details from clearinghouse state"""
        result = {
            "btc_position_size": 0.0,
            "btc_entry_price": 0.0,
            "btc_mark_price": 0.0,
            "position_value_usd": 0.0,
            "unrealized_pnl": 0.0,
            "leverage": 0.0,
            "margin_used": 0.0,
            "account_value": 0.0,
            "direction": "None"
        }
        
        if not clearinghouse_data:
            return result
        
        # Get account value
        if "marginSummary" in clearinghouse_data:
            result["account_value"] = float(clearinghouse_data["marginSummary"].get("accountValue", 0))
        
        # Get BTC mark price from asset contexts
        btc_mark_price = 0.0
        for ctx in asset_contexts:
            if ctx:  # First entry might be BTC (index 0)
                btc_mark_price = float(ctx.get("markPx", 0))
                break
        result["btc_mark_price"] = btc_mark_price
        
        # Find BTC position
        if "assetPositions" not in clearinghouse_data:
            return result
        
        for asset_pos in clearinghouse_data["assetPositions"]:
            if "position" not in asset_pos:
                continue
            
            position = asset_pos["position"]
            coin = position.get("coin", "")
            
            if coin == "BTC":
                # Extract position details
                szi = float(position.get("szi", 0))
                result["btc_position_size"] = szi
                result["btc_entry_price"] = float(position.get("entryPx", 0))
                result["position_value_usd"] = float(position.get("positionValue", 0))
                result["unrealized_pnl"] = float(position.get("unrealizedPnl", 0))
                result["margin_used"] = float(position.get("marginUsed", 0))
                
                # Determine direction
                if szi > 0:
                    result["direction"] = "Long"
                elif szi < 0:
                    result["direction"] = "Short"
                else:
                    result["direction"] = "None"
                
                # Extract leverage
                if "leverage" in position:
                    leverage_info = position["leverage"]
                    result["leverage"] = float(leverage_info.get("value", 0))
                
                break
        
        return result
    
    def get_current_position(self) -> Dict[str, Any]:
        """Get current BTC position snapshot"""
        print(f"Fetching position data for {self.wallet_address[:10]}...")
        
        # Get clearinghouse state and market data
        clearinghouse_data = self.get_clearinghouse_state()
        universe, asset_contexts = self.get_meta_and_asset_contexts()
        
        # Extract BTC position
        position_data = self.extract_btc_position(clearinghouse_data, asset_contexts)
        
        # Add timestamp
        position_data["timestamp"] = int(time.time())
        position_data["wallet_address"] = self.wallet_address
        
        return position_data
    
    def save_snapshot(self, position_data: Dict[str, Any]) -> str:
        """Save snapshot with timestamp in filename"""
        timestamp = position_data["timestamp"]
        datetime_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
        filename = f"{self.snapshots_dir}/btc_short_{datetime_str}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(position_data, f, indent=2)
        
        print(f"Snapshot saved to '{filename}'")
        return filename
    
    def load_all_snapshots(self) -> List[Dict[str, Any]]:
        """Load all snapshot files and return them sorted by timestamp"""
        snapshot_files = glob.glob(f"{self.snapshots_dir}/btc_short_*.json")
        snapshots = []
        
        for file in snapshot_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    snapshots.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Sort by timestamp
        snapshots.sort(key=lambda x: x.get('timestamp', 0))
        return snapshots
    
    def plot_position_history(self, snapshots: List[Dict[str, Any]], use_last_n: int = 100):
        """Create a dark mode plot of BTC short position over time"""
        if not snapshots:
            print("No snapshots available for plotting")
            return
        
        if use_last_n is not None and use_last_n > 0:
            snapshots.sort(key=lambda x: x.get('timestamp', 0))
            snapshots = snapshots[-use_last_n:]
        
        # Set dark style
        plt.style.use('dark_background')
        sns.set_theme(style="dark")
        
        # Extract data for plotting
        timestamps = []
        position_sizes = []
        position_values = []
        unrealized_pnls = []
        btc_prices = []
        
        for snapshot in snapshots:
            if 'timestamp' in snapshot:
                timestamps.append(datetime.fromtimestamp(snapshot['timestamp']))
                position_sizes.append(abs(snapshot.get('btc_position_size', 0)))
                position_values.append(abs(snapshot.get('position_value_usd', 0)))
                unrealized_pnls.append(snapshot.get('unrealized_pnl', 0))
                btc_prices.append(snapshot.get('btc_mark_price', 0))
        
        if not timestamps:
            print("No valid timestamp data in snapshots")
            return
        
        # Create figure with 4 subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), facecolor='#0d1117')
        
        # Plot 1: BTC Position Size
        ax1.plot(timestamps, position_sizes, color='#f85149', linewidth=2, marker='o', label='Short Position Size')
        ax1.set_ylabel('BTC Position Size', fontsize=12, color='white')
        ax1.set_title('BTC Short Position Size', fontsize=16, color='white', pad=20, fontweight='bold')
        ax1.grid(True, alpha=0.2, color='#30363d')
        ax1.legend(loc='upper right', framealpha=0.9, facecolor='#161b22', labelcolor='white')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4f} BTC'))
        
        # Plot 2: Position Value USD
        ax2.plot(timestamps, position_values, color='#58a6ff', linewidth=2, marker='o')
        ax2.set_ylabel('Position Value (USD)', fontsize=12, color='white')
        ax2.set_title('BTC Short Position Value', fontsize=16, color='white', pad=20, fontweight='bold')
        ax2.grid(True, alpha=0.2, color='#30363d')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 3: Unrealized PnL
        colors = ['#3fb950' if pnl >= 0 else '#f85149' for pnl in unrealized_pnls]
        ax3.bar(timestamps, unrealized_pnls, color=colors, width=0.02)
        ax3.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.3)
        ax3.set_ylabel('Unrealized PnL (USD)', fontsize=12, color='white')
        ax3.set_title('Unrealized PnL', fontsize=16, color='white', pad=20, fontweight='bold')
        ax3.grid(True, alpha=0.2, color='#30363d')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 4: BTC Mark Price
        ax4.plot(timestamps, btc_prices, color='#ffa500', linewidth=2, marker='o')
        ax4.set_xlabel('Date/Time', fontsize=12, color='white')
        ax4.set_ylabel('BTC Price (USD)', fontsize=12, color='white')
        ax4.set_title('BTC Mark Price', fontsize=16, color='white', pad=20, fontweight='bold')
        ax4.grid(True, alpha=0.2, color='#30363d')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis dates for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', color='white')
            plt.setp(ax.yaxis.get_majorticklabels(), color='white')
            ax.set_facecolor('#0d1117')
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            
            # Make tick marks white
            ax.tick_params(colors='white')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"btc_short_position_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=150, facecolor='#0d1117', edgecolor='none')
        plt.savefig("btc_short_position_latest.png", dpi=150, facecolor='#0d1117', edgecolor='none')
        print(f"Plot saved to '{plot_filename}'")
        
        plt.show()
    
    def print_position(self, position_data: Dict[str, Any]):
        """Print formatted position data"""
        print("\n" + "="*80)
        print("BTC SHORT POSITION TRACKER")
        print("="*80)
        print(f"Wallet: {position_data['wallet_address']}")
        print(f"Timestamp: {datetime.fromtimestamp(position_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nBTC Mark Price: ${position_data['btc_mark_price']:,.2f}")
        print(f"\nPosition Direction: {position_data['direction']}")
        print(f"Position Size: {position_data['btc_position_size']:.6f} BTC")
        
        if position_data['direction'] != "None":
            print(f"Entry Price: ${position_data['btc_entry_price']:,.2f}")
            print(f"Position Value: ${position_data['position_value_usd']:,.2f}")
            print(f"Unrealized PnL: ${position_data['unrealized_pnl']:,.2f}")
            print(f"Leverage: {position_data['leverage']:.1f}x")
            print(f"Margin Used: ${position_data['margin_used']:,.2f}")
        
        print(f"\nAccount Value: ${position_data['account_value']:,.2f}")
        print("="*80 + "\n")


def main():
    # Wallet address to track
    wallet_address = "0xb317d2bc2d3d2df5fa441b5bae0ab9d8b07283ae"
    
    tracker = HyperliquidShortTracker(wallet_address)
    
    # Get current position
    position_data = tracker.get_current_position()
    
    # Print position
    tracker.print_position(position_data)
    
    # Save snapshot
    tracker.save_snapshot(position_data)
    
    # Load all historical snapshots and create plot
    print("Loading historical snapshots...")
    snapshots = tracker.load_all_snapshots()
    print(f"Found {len(snapshots)} snapshots")
    
    if len(snapshots) > 1:
        print("Creating position history plot...")
        tracker.plot_position_history(snapshots)
    
    # Also save to a simple filename for easy access
    with open('btc_short_position_latest.json', 'w') as f:
        json.dump(position_data, f, indent=2)
    print("Current position also saved to 'btc_short_position_latest.json'")


if __name__ == "__main__":
    frequency = 60 * 1  # Run every 1 minute
    
    print("Starting BTC Short Position Tracker...")
    print(f"Monitoring wallet: 0xb317d2bc2d3d2df5fa441b5bae0ab9d8b07283ae")
    print(f"Update frequency: {frequency} seconds\n")
    
    while True:
        try:
            main()
        except Exception as e:
            print(f"Error in main loop: {e}")
        
        print(f"\nWaiting {frequency} seconds until next run...\n")
        time.sleep(frequency)