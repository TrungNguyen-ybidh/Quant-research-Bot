"""
Main Entry Point for Quantitative Research Bot
"""

from src.utils.config_loader import ConfigLoader
from src.data.collector import update_all_raw
from src.utils.logger import info, success, error


def main():
    """
    Main entry point for the quantitative research bot.
    
    Loads configuration and updates all raw data.
    """
    try:
        info("Starting Quantitative Research Bot")
        
        # Load configuration
        config_loader = ConfigLoader(config_path="config/config.yaml")
        cfg = config_loader.config
        
        if not cfg:
            error("Failed to load configuration")
            return
        
        success("Configuration loaded successfully")
        
        # Update all raw data
        info("Beginning data collection process")
        update_all_raw(cfg)
        
        success("Quantitative Research Bot completed successfully")
        
    except Exception as e:
        error(f"Fatal error in main(): {e}")
        raise


if __name__ == "__main__":
    main()

