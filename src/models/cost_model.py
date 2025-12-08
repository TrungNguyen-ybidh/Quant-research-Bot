"""
Module 2: Cost Model
====================
Purpose: Determine if signals survive realistic trading costs.

A 52.7% accuracy means nothing if costs eat the edge.
This module answers: "Is the signal profitable after spreads, commissions, and slippage?"

Usage:
    from src.models.cost_model import CostModel, analyze_strategy_costs
    
    cost_model = CostModel()
    results = cost_model.analyze(
        predictions=predictions,
        actuals=actuals,
        prices=prices,
        symbol="EURUSD",
        timeframe="1h"
    )
    
    if results["survives_costs"]:
        print("Signal is tradeable!")
    else:
        print("Signal dies after costs.")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from src.utils.logger import info, success, warning, error


# =============================================================================
# COST CONFIGURATION
# =============================================================================

@dataclass
class CostConfig:
    """
    Trading cost configuration.
    
    All costs are in percentage terms (0.0001 = 1 pip = 0.01%)
    """
    
    # Spread costs by pair (in pips, will be converted to %)
    # Conservative retail estimates for IBKR
    SPREAD_PIPS: Dict[str, float] = field(default_factory=lambda: {
        # Majors - tightest spreads
        "EURUSD": 0.3,
        "USDJPY": 0.3,
        "GBPUSD": 0.4,
        "USDCHF": 0.5,
        "AUDUSD": 0.4,
        "NZDUSD": 0.5,
        "USDCAD": 0.5,
        
        # Crosses - wider spreads
        "EURJPY": 0.7,
        "GBPJPY": 1.0,
        "AUDJPY": 0.8,
        "CHFJPY": 0.9,
        
        # Commodities
        "XAUUSD": 3.5,  # Gold - much wider
    })
    
    # Pip values (how much 1 pip is worth in price terms)
    PIP_VALUES: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "AUDUSD": 0.0001,
        "NZDUSD": 0.0001,
        "USDCAD": 0.0001,
        "USDCHF": 0.0001,
        "USDJPY": 0.01,    # JPY pairs have different pip
        "EURJPY": 0.01,
        "GBPJPY": 0.01,
        "AUDJPY": 0.01,
        "CHFJPY": 0.01,
        "XAUUSD": 0.01,    # Gold
    })
    
    # Session-based spread multipliers
    SESSION_MULTIPLIERS: Dict[str, float] = field(default_factory=lambda: {
        "london": 1.0,           # Tightest spreads
        "new_york": 1.0,         # Tight spreads
        "overlap": 0.8,          # Best spreads (London+NY overlap)
        "asia": 1.5,             # Wider spreads
        "weekend": 3.0,          # Much wider (Sunday open)
        "news": 2.5,             # Around major news
    })
    
    # Commission (IBKR Pro: ~$2 per $100k, or 0.002%)
    commission_pct: float = 0.00002  # 0.002%
    
    # Slippage estimate (conservative)
    slippage_pips: float = 0.1  # 0.1 pip average slippage
    
    # Market impact (for larger positions, usually 0 for retail)
    market_impact_pct: float = 0.0
    
    # Minimum net Sharpe to consider tradeable
    min_net_sharpe: float = 0.3
    
    # Minimum edge after costs (in %)
    min_edge_pct: float = 0.01  # 0.01% minimum edge per trade


# =============================================================================
# COST CALCULATOR
# =============================================================================

class CostCalculator:
    """Calculates trading costs for FX pairs."""
    
    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()
    
    def get_spread_pct(self, symbol: str, session: str = "london") -> float:
        """
        Get spread as percentage of price.
        
        Args:
            symbol: Currency pair
            session: Trading session for multiplier
        
        Returns:
            Spread as decimal (0.0003 = 0.03% = 3 pips on EURUSD)
        """
        # Get base spread in pips
        spread_pips = self.config.SPREAD_PIPS.get(symbol, 1.0)
        
        # Get pip value
        pip_value = self.config.PIP_VALUES.get(symbol, 0.0001)
        
        # Convert to percentage
        spread_pct = spread_pips * pip_value
        
        # Apply session multiplier
        multiplier = self.config.SESSION_MULTIPLIERS.get(session, 1.0)
        spread_pct *= multiplier
        
        return spread_pct
    
    def get_slippage_pct(self, symbol: str) -> float:
        """Get slippage as percentage."""
        pip_value = self.config.PIP_VALUES.get(symbol, 0.0001)
        return self.config.slippage_pips * pip_value
    
    def get_total_cost_pct(self, symbol: str, session: str = "london") -> float:
        """
        Get total round-trip cost as percentage.
        
        Includes: spread (entry + exit) + commission + slippage
        """
        spread = self.get_spread_pct(symbol, session)
        slippage = self.get_slippage_pct(symbol)
        commission = self.config.commission_pct
        
        # Round trip = 2x spread (entry + exit) + 2x slippage + 2x commission
        total = 2 * (spread + slippage + commission)
        
        return total
    
    def get_cost_breakdown(self, symbol: str, session: str = "london") -> Dict:
        """Get detailed cost breakdown."""
        spread = self.get_spread_pct(symbol, session)
        slippage = self.get_slippage_pct(symbol)
        commission = self.config.commission_pct
        
        return {
            "symbol": symbol,
            "session": session,
            "spread_one_way_pct": spread * 100,
            "spread_round_trip_pct": spread * 2 * 100,
            "slippage_one_way_pct": slippage * 100,
            "slippage_round_trip_pct": slippage * 2 * 100,
            "commission_one_way_pct": commission * 100,
            "commission_round_trip_pct": commission * 2 * 100,
            "total_round_trip_pct": self.get_total_cost_pct(symbol, session) * 100,
            "total_round_trip_bps": self.get_total_cost_pct(symbol, session) * 10000,
        }


# =============================================================================
# STRATEGY ANALYZER
# =============================================================================

class StrategyAnalyzer:
    """Analyzes strategy performance before and after costs."""
    
    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calc = cost_calculator
    
    def calculate_returns(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        lookahead: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate strategy returns based on predictions.
        
        Args:
            predictions: Binary predictions (1=long, 0=short/flat)
            prices: Price series
            lookahead: Holding period in bars
        
        Returns:
            Tuple of (positions, returns)
        """
        n = len(predictions)
        
        # Convert predictions to positions: 1=long, -1=short, 0=flat
        # For binary: 1 -> +1 (long), 0 -> -1 (short)
        positions = np.where(predictions == 1, 1, -1)
        
        # Calculate forward returns
        forward_returns = np.zeros(n)
        for i in range(n - lookahead):
            forward_returns[i] = (prices[i + lookahead] - prices[i]) / prices[i]
        
        # Strategy returns = position * market return
        strategy_returns = positions * forward_returns
        
        # Trim last `lookahead` bars (no forward return available)
        strategy_returns = strategy_returns[:-lookahead]
        positions = positions[:-lookahead]
        
        return positions, strategy_returns
    
    def apply_costs(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
        cost_per_trade: float
    ) -> np.ndarray:
        """
        Apply trading costs to returns.
        
        Costs are incurred when position changes.
        """
        net_returns = returns.copy()
        
        # Detect position changes (trades)
        position_changes = np.diff(positions, prepend=positions[0])
        trades = position_changes != 0
        
        # Apply cost on each trade
        net_returns[trades] -= cost_per_trade
        
        return net_returns
    
    def calculate_metrics(
        self,
        returns: np.ndarray,
        annualization_factor: float = 252 * 24  # Hourly data
    ) -> Dict:
        """Calculate performance metrics."""
        
        # Handle edge cases
        if len(returns) == 0 or np.std(returns) == 0:
            return {
                "mean_return": 0.0,
                "std_return": 0.0,
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "num_trades": 0
            }
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        # Sharpe ratio (annualized)
        sharpe = (mean_ret / std_ret) * np.sqrt(annualization_factor) if std_ret > 0 else 0
        
        # Win rate
        wins = np.sum(returns > 0)
        total = len(returns)
        win_rate = wins / total if total > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = np.abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Total return
        total_return = np.sum(returns)
        
        return {
            "mean_return_per_trade": float(mean_ret),
            "std_return": float(std_ret),
            "sharpe_ratio": float(sharpe),
            "total_return_pct": float(total_return * 100),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_dd * 100),
            "num_trades": int(total)
        }


# =============================================================================
# MAIN COST MODEL
# =============================================================================

class CostModel:
    """
    Main cost model for analyzing strategy viability.
    
    Answers: "Does this signal survive trading costs?"
    """
    
    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()
        self.cost_calc = CostCalculator(self.config)
        self.analyzer = StrategyAnalyzer(self.cost_calc)
    
    def analyze(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        symbol: str,
        timeframe: str = "1h",
        lookahead: int = 4,
        session: str = "london"
    ) -> Dict:
        """
        Full cost analysis for a strategy.
        
        Args:
            predictions: Binary predictions (1=up, 0=down)
            prices: Price series (same length as predictions)
            symbol: Currency pair
            timeframe: Data timeframe
            lookahead: Holding period in bars
            session: Trading session assumption
        
        Returns:
            Complete analysis with gross/net metrics and recommendation
        """
        info(f"Analyzing costs for {symbol} {timeframe}...")
        
        # Get annualization factor based on timeframe
        annualization = self._get_annualization(timeframe)
        
        # Calculate gross returns (before costs)
        positions, gross_returns = self.analyzer.calculate_returns(
            predictions, prices, lookahead
        )
        
        # Get cost per trade
        cost_per_trade = self.cost_calc.get_total_cost_pct(symbol, session)
        
        # Calculate net returns (after costs)
        net_returns = self.analyzer.apply_costs(gross_returns, positions, cost_per_trade)
        
        # Calculate metrics
        gross_metrics = self.analyzer.calculate_metrics(gross_returns, annualization)
        net_metrics = self.analyzer.calculate_metrics(net_returns, annualization)
        
        # Cost breakdown
        cost_breakdown = self.cost_calc.get_cost_breakdown(symbol, session)
        
        # Count actual trades (position changes)
        position_changes = np.diff(positions, prepend=positions[0])
        num_trades = np.sum(position_changes != 0)
        
        # Calculate cost drag
        cost_drag = gross_metrics["sharpe_ratio"] - net_metrics["sharpe_ratio"]
        cost_drag_pct = (cost_drag / gross_metrics["sharpe_ratio"] * 100 
                        if gross_metrics["sharpe_ratio"] > 0 else 0)
        
        # Determine if strategy survives costs
        # Must have positive net edge AND net Sharpe above threshold
        net_edge = net_metrics["mean_return_per_trade"]
        edge_after_costs = gross_metrics["mean_return_per_trade"] - cost_per_trade
        
        survives = (
            net_metrics["sharpe_ratio"] >= self.config.min_net_sharpe and
            edge_after_costs > 0 and  # Must have positive edge after costs
            net_metrics["mean_return_per_trade"] > self.config.min_edge_pct / 100
        )
        
        # Breakeven analysis
        breakeven = self._calculate_breakeven(gross_metrics, cost_per_trade)
        
        # Build result
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "lookahead": lookahead,
            "session": session,
            "num_samples": len(predictions),
            "num_trades": int(num_trades),
            "trades_per_day": self._trades_per_day(num_trades, len(predictions), timeframe),
            
            # Gross metrics (before costs)
            "gross": {
                "sharpe": gross_metrics["sharpe_ratio"],
                "total_return_pct": gross_metrics["total_return_pct"],
                "win_rate": gross_metrics["win_rate"],
                "profit_factor": gross_metrics["profit_factor"],
                "mean_return_per_trade_pct": gross_metrics["mean_return_per_trade"] * 100,
                "max_drawdown_pct": gross_metrics["max_drawdown_pct"]
            },
            
            # Net metrics (after costs)
            "net": {
                "sharpe": net_metrics["sharpe_ratio"],
                "total_return_pct": net_metrics["total_return_pct"],
                "win_rate": net_metrics["win_rate"],
                "profit_factor": net_metrics["profit_factor"],
                "mean_return_per_trade_pct": net_metrics["mean_return_per_trade"] * 100,
                "max_drawdown_pct": net_metrics["max_drawdown_pct"]
            },
            
            # Cost analysis
            "costs": {
                "cost_per_trade_pct": cost_per_trade * 100,
                "cost_per_trade_bps": cost_per_trade * 10000,
                "total_costs_pct": (num_trades * cost_per_trade) * 100,
                "cost_drag_sharpe": cost_drag,
                "cost_drag_pct": cost_drag_pct,
                **cost_breakdown
            },
            
            # Breakeven analysis
            "breakeven": breakeven,
            
            # Verdict
            "survives_costs": survives,
            "recommendation": self._get_recommendation(
                gross_metrics, net_metrics, cost_drag_pct, 
                breakeven["edge_consumed_by_costs_pct"]
            )
        }
        
        return result
    
    def _get_annualization(self, timeframe: str) -> float:
        """Get annualization factor based on timeframe."""
        factors = {
            "1m": 252 * 24 * 60,   # Minutes per year
            "5m": 252 * 24 * 12,   # 5-min bars per year
            "15m": 252 * 24 * 4,   # 15-min bars per year
            "1h": 252 * 24,        # Hours per year
            "4h": 252 * 6,         # 4-hour bars per year
            "1d": 252,             # Days per year
        }
        return factors.get(timeframe, 252 * 24)
    
    def _trades_per_day(self, num_trades: int, num_bars: int, timeframe: str) -> float:
        """Calculate average trades per day."""
        bars_per_day = {
            "1m": 24 * 60,
            "5m": 24 * 12,
            "15m": 24 * 4,
            "1h": 24,
            "4h": 6,
            "1d": 1,
        }
        bpd = bars_per_day.get(timeframe, 24)
        days = num_bars / bpd
        return num_trades / days if days > 0 else 0
    
    def _calculate_breakeven(self, gross_metrics: Dict, cost_per_trade: float) -> Dict:
        """Calculate breakeven requirements."""
        gross_edge = gross_metrics["mean_return_per_trade"]
        
        # Breakeven win rate (assuming equal win/loss size)
        # With costs: win_rate * avg_win - (1 - win_rate) * avg_loss - cost = 0
        # Simplified: need edge > cost
        breakeven_winrate = 0.5 + (cost_per_trade / (2 * abs(gross_edge))) if gross_edge != 0 else 1.0
        breakeven_winrate = min(breakeven_winrate, 1.0)
        
        # How much edge is consumed by costs
        edge_consumed_pct = (cost_per_trade / gross_edge * 100) if gross_edge > 0 else 100
        
        # Maximum cost that would still be profitable
        max_viable_cost = gross_edge * 0.7  # Keep 30% of edge
        
        return {
            "breakeven_winrate": float(breakeven_winrate),
            "edge_consumed_by_costs_pct": float(edge_consumed_pct),
            "max_viable_cost_pct": float(max_viable_cost * 100),
            "current_cost_pct": float(cost_per_trade * 100),
            "cost_headroom_pct": float((max_viable_cost - cost_per_trade) * 100)
        }
    
    def _get_recommendation(
        self, 
        gross: Dict, 
        net: Dict, 
        cost_drag_pct: float,
        edge_consumed_pct: float = 0
    ) -> str:
        """Generate recommendation based on analysis."""
        
        net_sharpe = net["sharpe_ratio"]
        gross_sharpe = gross["sharpe_ratio"]
        net_edge = net["mean_return_per_trade"]
        
        # Primary check: is there positive edge after costs?
        if edge_consumed_pct >= 100:
            return "REJECT - Costs consume 100%+ of edge. Strategy is breakeven or losing."
        
        if net_edge <= 0:
            return "REJECT - No positive edge after costs"
        
        if edge_consumed_pct >= 70:
            return "MARGINAL - Costs consume >70% of edge. Very thin margin for error."
        
        if edge_consumed_pct >= 50:
            return "PROCEED WITH CAUTION - Costs consume >50% of edge. Consider lower frequency."
        
        if net_sharpe >= 0.5 and edge_consumed_pct < 50:
            return "STRONG PROCEED - Signal survives costs with good margin"
        elif net_sharpe >= 0.3:
            return "PROCEED - Signal survives costs"
        else:
            return "INVESTIGATE - Edge exists but metrics are weak"
    
    def print_report(self, result: Dict):
        """Print formatted cost analysis report."""
        
        print("\n" + "=" * 70)
        print(f"COST ANALYSIS REPORT: {result['symbol']} {result['timeframe']}")
        print("=" * 70)
        
        print(f"\nSamples: {result['num_samples']:,} | "
              f"Trades: {result['num_trades']:,} | "
              f"Trades/Day: {result['trades_per_day']:.1f}")
        
        print("\n" + "-" * 70)
        print("PERFORMANCE COMPARISON")
        print("-" * 70)
        print(f"{'Metric':<30} {'Gross':>15} {'Net':>15} {'Impact':>15}")
        print("-" * 70)
        
        g, n = result["gross"], result["net"]
        
        print(f"{'Sharpe Ratio':<30} {g['sharpe']:>15.3f} {n['sharpe']:>15.3f} "
              f"{n['sharpe']-g['sharpe']:>+15.3f}")
        print(f"{'Total Return %':<30} {g['total_return_pct']:>15.2f} {n['total_return_pct']:>15.2f} "
              f"{n['total_return_pct']-g['total_return_pct']:>+15.2f}")
        print(f"{'Win Rate':<30} {g['win_rate']:>15.1%} {n['win_rate']:>15.1%} "
              f"{n['win_rate']-g['win_rate']:>+15.1%}")
        print(f"{'Profit Factor':<30} {g['profit_factor']:>15.2f} {n['profit_factor']:>15.2f} "
              f"{n['profit_factor']-g['profit_factor']:>+15.2f}")
        print(f"{'Max Drawdown %':<30} {g['max_drawdown_pct']:>15.2f} {n['max_drawdown_pct']:>15.2f} "
              f"{n['max_drawdown_pct']-g['max_drawdown_pct']:>+15.2f}")
        
        print("\n" + "-" * 70)
        print("COST BREAKDOWN")
        print("-" * 70)
        c = result["costs"]
        print(f"  Spread (round-trip):     {c['spread_round_trip_pct']:.4f}%")
        print(f"  Slippage (round-trip):   {c['slippage_round_trip_pct']:.4f}%")
        print(f"  Commission (round-trip): {c['commission_round_trip_pct']:.4f}%")
        print(f"  Total per trade:         {c['cost_per_trade_pct']:.4f}% ({c['cost_per_trade_bps']:.2f} bps)")
        print(f"  Total costs paid:        {c['total_costs_pct']:.2f}%")
        print(f"  Cost drag on Sharpe:     {c['cost_drag_pct']:.1f}%")
        
        print("\n" + "-" * 70)
        print("BREAKEVEN ANALYSIS")
        print("-" * 70)
        b = result["breakeven"]
        print(f"  Breakeven win rate:      {b['breakeven_winrate']:.1%}")
        print(f"  Edge consumed by costs:  {b['edge_consumed_by_costs_pct']:.1f}%")
        print(f"  Cost headroom:           {b['cost_headroom_pct']:.4f}%")
        
        print("\n" + "=" * 70)
        print(f"SURVIVES COSTS: {'YES ✓' if result['survives_costs'] else 'NO ✗'}")
        print(f"RECOMMENDATION: {result['recommendation']}")
        print("=" * 70 + "\n")
    
    def save_results(self, result: Dict, output_dir: str = "outputs/costs"):
        """Save results to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{result['symbol']}_{result['timeframe']}_cost_analysis.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        success(f"Results saved to {filepath}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_strategy_costs(
    predictions: np.ndarray,
    prices: np.ndarray,
    symbol: str,
    timeframe: str = "1h",
    lookahead: int = 4,
    print_report: bool = True,
    save_results: bool = True
) -> Dict:
    """
    Convenience function to run full cost analysis.
    
    Args:
        predictions: Binary predictions (1=up, 0=down)
        prices: Price series
        symbol: Currency pair
        timeframe: Data timeframe
        lookahead: Holding period
        print_report: Print formatted report
        save_results: Save to JSON file
    
    Returns:
        Analysis results dictionary
    """
    model = CostModel()
    result = model.analyze(predictions, prices, symbol, timeframe, lookahead)
    
    if print_report:
        model.print_report(result)
    
    if save_results:
        model.save_results(result)
    
    return result


def get_cost_summary_all_pairs() -> pd.DataFrame:
    """Get cost summary for all configured pairs."""
    config = CostConfig()
    calc = CostCalculator(config)
    
    rows = []
    for symbol in config.SPREAD_PIPS.keys():
        breakdown = calc.get_cost_breakdown(symbol, "london")
        rows.append({
            "symbol": symbol,
            "spread_rt_pct": breakdown["spread_round_trip_pct"],
            "total_rt_pct": breakdown["total_round_trip_pct"],
            "total_rt_bps": breakdown["total_round_trip_bps"]
        })
    
    df = pd.DataFrame(rows).sort_values("total_rt_bps")
    return df


# =============================================================================
# TEST WITH BASELINE RESULTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COST MODEL TEST")
    print("=" * 70)
    
    # Show cost summary for all pairs
    print("\nCost Summary (All Pairs):")
    print("-" * 50)
    cost_df = get_cost_summary_all_pairs()
    print(cost_df.to_string(index=False))
    
    # Load baseline results and run cost analysis
    print("\n" + "=" * 70)
    print("Running cost analysis on EURUSD baseline predictions...")
    print("=" * 70)
    
    # Load data
    from pathlib import Path
    import pandas as pd
    
    # Load raw data for prices
    raw_path = Path("data/raw/clock/EURUSD_1h.parquet")
    if raw_path.exists():
        raw_df = pd.read_parquet(raw_path)
        prices = raw_df['close'].values
        
        # For testing: create mock predictions based on simple momentum
        # In real use, these would come from baseline_predictor
        returns = pd.Series(prices).pct_change().values
        mock_predictions = (returns > 0).astype(int)
        mock_predictions = np.roll(mock_predictions, 1)  # Lag by 1 to avoid lookahead
        mock_predictions[0] = 1
        
        # Run analysis
        result = analyze_strategy_costs(
            predictions=mock_predictions,
            prices=prices,
            symbol="EURUSD",
            timeframe="1h",
            lookahead=4,
            print_report=True,
            save_results=True
        )
    else:
        print(f"Data file not found: {raw_path}")
        print("Run baseline_predictor.py first to generate predictions.")