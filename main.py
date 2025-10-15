import requests
import json
from typing import List, Dict, Any
import concurrent.futures
import threading
import time
from datetime import datetime
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd

class HyperliquidHypeTracker:
    def __init__(self):
        self.base_url = "https://api.hyperliquid.xyz/info"
        self.hype_token = "HYPE"  # HYPE appears as "coin": "HYPE" in balances
        self.snapshots_dir = "hype_snapshots"
        
        # Create snapshots directory if it doesn't exist
        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)

    def get_single_address_balance(self, address: str, include_staking: bool = False) -> tuple:
        """Get spot and staking balances for a single address
        
        Args:
            address: Wallet address to check
            include_staking: Whether to fetch staking data (delegated + undelegated + pending)
        
        Returns:
            Tuple of (address, spot_balance, staked_balance, undelegated_balance, pending_withdrawal)
        """
        # Get spot balance using existing method
        spot_balance = 0.0
        try:
            balance_data = self.get_spot_balances(address)
            spot_balance = self.extract_hype_balance(balance_data)
        except Exception as e:
            print(f"Error fetching spot balance for {address}: {e}")
        
        staked_balance = 0.0
        undelegated_balance = 0.0
        pending_withdrawal = 0.0
        
        if include_staking:
            try:
                # Get comprehensive staking summary using existing method
                staking_summary = self.get_staking_summary(address)
                
                if staking_summary:
                    # "delegated" = actively staked with validators
                    staked_balance = float(staking_summary.get("delegated", 0))
                    
                    # "undelegated" = in staking balance but not allocated to validators
                    undelegated_balance = float(staking_summary.get("undelegated", 0))
                    
                    # "totalPendingWithdrawal" = sum of all pending withdrawals (7 day lockup)
                    pending_withdrawal = float(staking_summary.get("totalPendingWithdrawal", 0))
                    
            except Exception as e:
                print(f"Error fetching staking data for {address}: {e}")
        
        return (address, spot_balance, staked_balance, undelegated_balance, pending_withdrawal)


    def get_all_holdings(self, addresses: List[str], staked_addresses: List[str] = None, 
                        delay_seconds: float = 1.0, max_retries: int = 3) -> Dict[str, Any]:
        """Get HYPE holdings for all addresses with throttling and retry logic
        
        Args:
            addresses: List of wallet addresses to check
            staked_addresses: List of addresses that have staked balances to include
            delay_seconds: Delay between requests to avoid rate limiting
            max_retries: Maximum number of retry attempts per address
        """
        if staked_addresses is None:
            staked_addresses = []
        
        print("Fetching all mid prices...")
        all_mids = self.get_all_prices()
        
        # Extract HYPE price - could be under "HYPE" or "@107"
        hype_price = 0.0
        for key in ["HYPE", "@107"]:
            if key in all_mids:
                try:
                    hype_price = float(all_mids[key])
                    print(f"Found HYPE price under key '{key}': ${hype_price:.4f}")
                    break
                except (ValueError, TypeError):
                    continue
        print(f"Found PURR price: ${all_mids['PURR']}")
        
        if hype_price == 0:
            print("Warning: Could not fetch HYPE price from allMids")
        
        print(f"\nFetching holdings for {len(addresses)} addresses sequentially (delay: {delay_seconds}s)...")
        if staked_addresses:
            print(f"Including staking data for {len(staked_addresses)} addresses: {', '.join([addr[:10] + '...' for addr in staked_addresses])}")
        
        results = {
            "hype_price": hype_price,
            "addresses": {},
            "total_spot_hype": 0.0,
            "total_staked_hype": 0.0,
            "total_undelegated_hype": 0.0,
            "total_pending_withdrawal": 0.0,
            "total_hype": 0.0,
            "total_usd_value": 0.0
        }
        
        # Process addresses sequentially with throttling
        for idx, addr in enumerate(addresses, 1):
            include_staking = addr in staked_addresses
            
            # Retry logic
            for attempt in range(max_retries):
                try:
                    address, spot_balance, staked_balance, undelegated_balance, pending_withdrawal = \
                        self.get_single_address_balance(addr, include_staking)
                    
                    # Total balance includes everything: spot + staked + undelegated + pending
                    total_balance = spot_balance + staked_balance + undelegated_balance + pending_withdrawal
                    usd_value = total_balance * hype_price
                    
                    results["addresses"][address] = {
                        "spot_balance": spot_balance,
                        "staked_balance": staked_balance,
                        "undelegated_balance": undelegated_balance,
                        "pending_withdrawal": pending_withdrawal,
                        "total_balance": total_balance,
                        "usd_value": usd_value
                    }
                    
                    results["total_spot_hype"] += spot_balance
                    results["total_staked_hype"] += staked_balance
                    results["total_undelegated_hype"] += undelegated_balance
                    results["total_pending_withdrawal"] += pending_withdrawal
                    results["total_hype"] += total_balance
                    results["total_usd_value"] += usd_value
                    
                    # Print progress with all balance types
                    if include_staking:
                        status_parts = [f"Spot: {spot_balance:,.2f}"]
                        if staked_balance > 0:
                            status_parts.append(f"Staked: {staked_balance:,.2f}")
                        if undelegated_balance > 0:
                            status_parts.append(f"Undelegated: {undelegated_balance:,.2f}")
                        if pending_withdrawal > 0:
                            status_parts.append(f"Pending: {pending_withdrawal:,.2f}")
                        print(f"Completed {idx}/{len(addresses)}: {address[:10]}... - {', '.join(status_parts)} HYPE")
                    else:
                        print(f"Completed {idx}/{len(addresses)}: {address[:10]}... - {spot_balance:,.2f} HYPE")
                    
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = delay_seconds * (attempt + 2)  # Exponential backoff
                        print(f"Error processing {addr[:10]}... (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error processing {addr[:10]}... after {max_retries} attempts: {e}")
                        results["addresses"][addr] = {
                            "spot_balance": 0.0,
                            "staked_balance": 0.0,
                            "undelegated_balance": 0.0,
                            "pending_withdrawal": 0.0,
                            "total_balance": 0.0,
                            "usd_value": 0.0
                        }
            
            # Throttle between requests (skip delay after last address)
            if idx < len(addresses):
                time.sleep(delay_seconds)
        
        # Recalculate totals to ensure accuracy
        results["total_spot_hype"] = sum(data["spot_balance"] for data in results["addresses"].values())
        results["total_staked_hype"] = sum(data["staked_balance"] for data in results["addresses"].values())
        results["total_undelegated_hype"] = sum(data["undelegated_balance"] for data in results["addresses"].values())
        results["total_pending_withdrawal"] = sum(data["pending_withdrawal"] for data in results["addresses"].values())
        results["total_hype"] = sum(data["total_balance"] for data in results["addresses"].values())
        results["total_usd_value"] = sum(data["usd_value"] for data in results["addresses"].values())

        return results
        
    def get_all_prices(self) -> Dict[str, str]:
        """Get all mid prices including HYPE using allMids endpoint"""
        payload = {"type": "allMids"}
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching all mids: {e}")
            return {}
    
    def get_spot_balances(self, address: str) -> Dict[str, Any]:
        """Get spot balances for a specific address"""
        payload = {
            "type": "spotClearinghouseState",
            "user": address
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching spot data for {address}: {e}")
            return {}
    
    def get_staking_delegations(self, address: str) -> List[Dict[str, Any]]:
        """Get staking delegations for a specific address"""
        payload = {
            "type": "delegations",
            "user": address
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching delegation data for {address}: {e}")
            return []
    
    def get_staking_summary(self, address: str) -> Dict[str, Any]:
        """Get staking summary including undelegated balance for a specific address"""
        payload = {
            "type": "delegatorSummary",
            "user": address
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching staking summary for {address}: {e}")
            return {}
    
    def extract_hype_balance(self, balance_data: Dict[str, Any]) -> float:
        """Extract HYPE balance from clearinghouse state"""
        if not balance_data or "balances" not in balance_data:
            return 0.0
        
        for balance in balance_data["balances"]:
            # Look for HYPE token using the coin name directly
            if balance.get("coin") == self.hype_token:
                try:
                    total_balance = float(balance.get("total", 0))
                    return total_balance
                except (ValueError, TypeError):
                    continue
        return 0.0
    
    def get_total_staked_hype(self, address: str) -> float:
        """Get total staked HYPE (delegated + undelegated in staking account)"""
        # Get staking summary
        summary = self.get_staking_summary(address)
        
        total_staked = 0.0
        
        # Add delegated amount
        if summary and "delegated" in summary:
            try:
                delegated = float(summary.get("delegated", 0))
                total_staked += delegated
            except (ValueError, TypeError):
                pass
        
        # Add undelegated amount (in staking account but not delegated to validators)
        if summary and "undelegated" in summary:
            try:
                undelegated = float(summary.get("undelegated", 0))
                total_staked += undelegated
            except (ValueError, TypeError):
                pass
        
        return total_staked
    
    def save_snapshot(self, results: Dict[str, Any]) -> str:
        """Save snapshot with timestamp in filename"""
        timestamp = int(time.time())
        datetime_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
        filename = f"{self.snapshots_dir}/hype_snapshot_{datetime_str}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Snapshot saved to '{filename}'")
        return filename
    
    def load_all_snapshots(self) -> List[Dict[str, Any]]:
        """Load all snapshot files and return them sorted by timestamp"""
        snapshot_files = glob.glob(f"{self.snapshots_dir}/hype_snapshot_*.json")
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
    
    def plot_balance_history(
            self, 
            snapshots: List[Dict[str, Any]],
            use_last_n: int = 100
        ):
        """Create a dark mode plot of HYPE balance over time with sell pressure metrics"""
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
        total_hype = []
        spot_hype = []
        staked_hype = []
        usd_values = []
        
        for snapshot in snapshots:
            if 'timestamp' in snapshot:
                timestamps.append(datetime.fromtimestamp(snapshot['timestamp']))
                total_hype.append(snapshot.get('total_hype', 0))
                spot_hype.append(snapshot.get('total_spot_hype', 0))
                staked_hype.append(snapshot.get('total_staked_hype', 0))
                usd_values.append(snapshot.get('total_usd_value', 0))
        
        if not timestamps:
            print("No valid timestamp data in snapshots")
            return
        
        # Calculate average sell pressure based on HYPE amounts only
        avg_sell_pressure_daily = None
        avg_sell_pressure_hourly = None
        hourly_sell_rate_hype = []  # Track HYPE sold per hour
        hourly_sell_rate_usd = []   # Track USD equivalent using current price
        avg_hype_price = None
        
        if len(total_hype) > 1 and len(timestamps) > 1:
            # Calculate total HYPE sold
            hype_sold = total_hype[0] - total_hype[-1]
            
            # Calculate average HYPE price over the period
            prices = []
            for i in range(len(total_hype)):
                if total_hype[i] > 0 and usd_values[i] > 0:
                    prices.append(usd_values[i] / total_hype[i])
            
            if prices:
                avg_hype_price = sum(prices) / len(prices)
                
                # Calculate time spans
                time_span_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
                time_span_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
                
                if time_span_hours > 0:
                    # Calculate average HYPE sold per hour/day
                    avg_hype_hourly = hype_sold / time_span_hours
                    avg_hype_daily = hype_sold / time_span_days
                    
                    # Convert to USD using average price
                    avg_sell_pressure_hourly = avg_hype_hourly * avg_hype_price
                    avg_sell_pressure_daily = avg_hype_daily * avg_hype_price
                    
                    # Calculate rolling hourly sell rate based on HYPE amounts
                    for i in range(1, len(timestamps)):
                        # HYPE sold (positive when balance decreases)
                        hype_sold_period = total_hype[i-1] - total_hype[i]
                        time_diff_hours = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                        
                        if time_diff_hours > 0:
                            # HYPE sold per hour
                            hype_rate = hype_sold_period / time_diff_hours
                            hourly_sell_rate_hype.append(hype_rate)
                            
                            # Convert to USD using current price at this point
                            if total_hype[i] > 0:
                                current_price = usd_values[i] / total_hype[i]
                            else:
                                current_price = avg_hype_price
                            hourly_sell_rate_usd.append(hype_rate * current_price)
                        else:
                            hourly_sell_rate_hype.append(0)
                            hourly_sell_rate_usd.append(0)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), facecolor='#0d1117')
        
        # Plot HYPE balances
        ax1.plot(timestamps, total_hype, color='#58a6ff', linewidth=2, label='Total HYPE', marker='o')
        
        ax1.set_xlabel('Date/Time', fontsize=12, color='white')
        ax1.set_ylabel('HYPE Balance', fontsize=12, color='white')
        ax1.grid(True, alpha=0.2, color='#30363d')
        ax1.legend(loc='upper right', framealpha=0.9, facecolor='#161b22', labelcolor='white')
        
        # Format y-axis for thousands
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Plot USD value
        ax2.plot(timestamps, usd_values, color='#3fb950', linewidth=2, marker='o')
        ax2.set_xlabel('Date/Time', fontsize=12, color='white')
        ax2.set_ylabel('USD Value', fontsize=12, color='white')
        ax2.set_title('Dumper USD Value Remaining', fontsize=16, color='white', pad=20, fontweight='bold')
        ax2.grid(True, alpha=0.2, color='#30363d')
        
        # Format y-axis for currency
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot sell pressure (if we have data)
        if hourly_sell_rate_usd and len(hourly_sell_rate_usd) > 0:
            # Create twin axis for daily scale
            ax3_daily = ax3.twinx()
            
            # Only show positive values (actual selling)
            hourly_sell_rate_positive = [max(0, rate) for rate in hourly_sell_rate_usd]
            
            # Cap at reasonable maximum for visualization
            hourly_sell_rate_clipped = [min(rate, 1_000_000) for rate in hourly_sell_rate_positive]
            
            line1 = ax3.plot(timestamps[1:], hourly_sell_rate_clipped, color='#f85149', linewidth=1.5, 
                            alpha=0.8, label='Hourly Rate')
            
            # Add average lines (only if positive - indicating net selling)
            lines = [line1[0]]
            if avg_sell_pressure_hourly and avg_sell_pressure_hourly > 0:
                line2 = ax3.axhline(y=min(avg_sell_pressure_hourly, 1_000_000), color='#ff6b6b', linestyle='--', 
                                    linewidth=2, label=f'Avg: ${min(avg_sell_pressure_hourly, 1_000_000):,.0f}/hr')
                lines.append(line2)
                
                if avg_sell_pressure_daily and avg_sell_pressure_daily > 0:
                    line3 = ax3_daily.axhline(y=avg_sell_pressure_daily, color='#ffa500', linestyle='--', 
                                            linewidth=2, label=f'Avg: ${avg_sell_pressure_daily:,.0f}/day')
                    lines.append(line3)
            
            # Labels and formatting
            ax3.set_xlabel('Date/Time', fontsize=12, color='white')
            ax3.set_ylabel('USD/Hour', fontsize=12, color='#ff6b6b')
            ax3_daily.set_ylabel('USD/Day', fontsize=12, color='#ffa500')
            
            if avg_sell_pressure_hourly and avg_sell_pressure_hourly > 0:
                sell_pressure_title = f'Sell Pressure: ${avg_sell_pressure_hourly:,.0f}/hr (${avg_sell_pressure_daily:,.0f}/day)'
            else:
                sell_pressure_title = 'Sell Pressure (Net Buying Detected)'
            ax3.set_title(sell_pressure_title, fontsize=16, color='white', pad=20, fontweight='bold')
            
            ax3.grid(True, alpha=0.2, color='#30363d')
            
            # Format y-axes for currency
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax3_daily.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Color the y-axis labels
            ax3.tick_params(axis='y', labelcolor='#ff6b6b')
            ax3_daily.tick_params(axis='y', labelcolor='#ffa500')
            
            # Combine legends
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper right', framealpha=0.9, 
                    facecolor='#161b22', labelcolor='white')
            
            # Add zero line for reference
            ax3.axhline(y=0, color='#30363d', linestyle='-', linewidth=1, alpha=0.5)
            
            # Style right axis
            ax3_daily.set_facecolor('#0d1117')
            for spine in ax3_daily.spines.values():
                spine.set_color('#30363d')
        else:
            # If no sell pressure data, show placeholder
            ax3.text(0.5, 0.5, 'Insufficient data for sell pressure analysis', 
                    transform=ax3.transAxes, ha='center', va='center', 
                    fontsize=14, color='#888888')
            ax3.set_title('Sell Pressure', fontsize=16, color='white', pad=20, fontweight='bold')
        
        # Format x-axis dates and make tick labels white
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', color='white')
            plt.setp(ax.yaxis.get_majorticklabels(), color='white')
            ax.set_facecolor('#0d1117')
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            
            # Make tick marks white
            ax.tick_params(colors='white')
        
        # Add current values as text
        if len(snapshots) > 0:
            latest = snapshots[-1]
            title_text = f"Remaining: {latest.get('total_hype', 0):,.2f} HYPE (${latest.get('total_usd_value', 0):,.2f})"
            ax1.set_title(title_text, fontsize=16, color='white', pad=20, fontweight='bold')

        plt.tight_layout()
        
        # Save plot
        plot_filename = f"old_pngs/hype_dumper_balance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=150, facecolor='#0d1117', edgecolor='none')
        plt.savefig("hype_dumper_balance_latest.png", dpi=150, facecolor='#0d1117', edgecolor='none')
        print(f"Plot saved to '{plot_filename}'")
        
        # Print sell pressure summary
        if avg_sell_pressure_hourly and avg_sell_pressure_daily and avg_sell_pressure_hourly > 0:
            print(f"\nSell Pressure Analysis:")
            print(f"  Average HYPE price: ${avg_hype_price:.2f}")
            print(f"  Total HYPE sold: {hype_sold:,.2f} HYPE")
            print(f"  Average hourly HYPE sold: {avg_hype_hourly:,.2f} HYPE")
            print(f"  Average hourly sell pressure: ${avg_sell_pressure_hourly:,.2f}")
            print(f"  Average daily sell pressure: ${avg_sell_pressure_daily:,.2f}")
            print(f"  Time period: {time_span_days:.1f} days ({time_span_hours:.1f} hours)")
            if avg_sell_pressure_hourly > 0 and len(usd_values) > 0:
                hours_until_depleted = usd_values[-1] / avg_sell_pressure_hourly
                days_until_depleted = hours_until_depleted / 24
                print(f"  Estimated time until depleted: {hours_until_depleted:.1f} hours ({days_until_depleted:.1f} days)")
        elif avg_sell_pressure_hourly and avg_sell_pressure_hourly < 0:
            print(f"\nNet Buying Detected:")
            print(f"  Holdings have increased over the monitoring period")
            print(f"  Net HYPE acquired: {abs(hype_sold):,.2f} HYPE")
        
        plt.show()
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results"""
        print("\n" + "="*80)
        print("HYPE HOLDINGS SUMMARY")
        print("="*80)
        print(f"HYPE Token Price: ${results['hype_price']:.4f}")
        print(f"Total Addresses Checked: {len(results['addresses'])}")
        print(f"Total HYPE Holdings: {results['total_hype']:,.2f} HYPE")
        print(f"  - Spot Balance: {results['total_spot_hype']:,.2f} HYPE")
        print(f"  - Staked Balance: {results['total_staked_hype']:,.2f} HYPE")
        print(f"Total USD Value: ${results['total_usd_value']:,.2f}")
        
        # Count addresses with holdings
        addresses_with_holdings = sum(1 for data in results['addresses'].values() if data['total_balance'] > 0)
        addresses_with_staking = sum(1 for data in results['addresses'].values() if data['staked_balance'] > 0)
        print(f"Addresses with HYPE: {addresses_with_holdings}/{len(results['addresses'])}")
        print(f"Addresses with staking: {addresses_with_staking}")
        
        print("\n" + "-"*80)
        print("INDIVIDUAL ADDRESS BREAKDOWN:")
        print("-"*80)
        
        # Sort addresses by total HYPE balance (descending)
        sorted_addresses = sorted(
            results["addresses"].items(), 
            key=lambda x: x[1]["total_balance"], 
            reverse=True
        )
        
        for address, data in sorted_addresses:
            spot_bal = data["spot_balance"]
            staked_bal = data["staked_balance"]
            total_bal = data["total_balance"]
            usd_val = data["usd_value"]
            
            if total_bal > 0:
                if staked_bal > 0:
                    print(f"{address}: {total_bal:,.2f} HYPE (${usd_val:,.2f})")
                    print(f"  └─ Spot: {spot_bal:,.2f} | Staked: {staked_bal:,.2f}")
                else:
                    print(f"{address}: {spot_bal:,.2f} HYPE (${usd_val:,.2f})")
            else:
                print(f"{address}: No HYPE holdings")
        
        print("\n" + "="*80)
        
        # Performance summary
        if results['total_hype'] > 0:
            avg_holding = results['total_hype'] / addresses_with_holdings if addresses_with_holdings > 0 else 0
            print(f"Average HYPE holding (non-zero addresses): {avg_holding:,.2f} HYPE")
            
            # Find largest holder
            largest = max(results['addresses'].items(), key=lambda x: x[1]['total_balance'])
            largest_percentage = (largest[1]['total_balance'] / results['total_hype']) * 100
            print(f"Largest holder: {largest[0][:10]}... with {largest[1]['total_balance']:,.2f} HYPE ({largest_percentage:.1f}%)")
            
            # Staking percentage
            if results['total_staked_hype'] > 0:
                staking_percentage = (results['total_staked_hype'] / results['total_hype']) * 100
                print(f"Percentage staked: {staking_percentage:.1f}%")
        print("="*80)

def main(
        wallets_json_path: str = "data/clean_unstakers.json",
        incl_staking: bool = True
    ):
    # List of wallet addresses
    with open(wallets_json_path, "r") as f:
        addresses = json.load(f) 

    # de-dupe
    addresses = list(set(addresses))
    
    # Addresses that have staked balances - add the ones you know have staking
    staked_addresses = []
    if incl_staking:
        staked_addresses = addresses  # include all, for use with unstaking queue tracker

    tracker = HyperliquidHypeTracker()
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Get current holdings
    results = tracker.get_all_holdings(addresses, staked_addresses)
    
    end_time = time.time()
    
    # Add timestamp and execution timex``
    results['execution_time'] = end_time - start_time
    results['timestamp'] = int(time.time())
    
    # Print results
    tracker.print_results(results)
    
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    
    # Save snapshot with timestamp
    tracker.save_snapshot(results)
    
    # Load all historical snapshots and create plot
    print("\nLoading historical snapshots...")
    snapshots = tracker.load_all_snapshots()
    print(f"Found {len(snapshots)} snapshots")
    
    if snapshots:
        print("Creating balance history plot...")
        tracker.plot_balance_history(snapshots)
    
    # Also save to the regular filename for compatibility
    with open('hype_holdings_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Current results also saved to 'hype_holdings_results.json'")

if __name__ == "__main__":
    frequency = 60 * 10  # Run every 5 minutes
    while True:
        main()
        print(f"\nWaiting {frequency/60:.1f} minutes until next run...\n")
        time.sleep(frequency)
