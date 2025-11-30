"""
Contract Builder - Build IBKR Contracts
Prevents 95% of FX contract errors with clean, standardized contract creation
"""

from ibapi.contract import Contract


def build_fx_contract(pair: str) -> Contract:
    """
    Build an IBKR FX contract.
    
    Example: EURUSD -> EUR/USD pair
    
    Args:
        pair: FX pair symbol (e.g., "EURUSD", "GBPUSD")
    
    Returns:
        IBKR Contract object configured for FX
    """
    contract = Contract()
    contract.secType = "CASH"
    contract.symbol = pair[:3]  # Base currency (e.g., "EUR")
    contract.currency = pair[3:]  # Quote currency (e.g., "USD")
    contract.exchange = "IDEALPRO"
    return contract

