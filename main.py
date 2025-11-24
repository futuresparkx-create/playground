# main.py
"""
Main entry point for the AI Code Improvement Playground.
Provides a safe, human-supervised AI system for code generation and analysis.
"""

import sys
import signal
from pathlib import Path
from typing import Optional

from config.config_manager import ConfigManager, ConfigurationError
from orchestrator.graph import ImprovementGraph
from memory.lpra_integration import LPRAManager, LPRAIntegratedGraph
from ui.dashboard.monitor import Dashboard
from models.model_factory import ModelFactory
from security.validators import InputValidator, RateLimiter
from utils.logging_config import setup_logging, get_logger
from utils.exceptions import PlaygroundException

# Setup logging
setup_logging(
    log_level="INFO",
    log_file=Path("logs/playground.log"),
    json_format=False,
    enable_security_filter=True
)

logger = get_logger(__name__)


class PlaygroundApp:
    """
    Main application class for the AI Code Improvement Playground.
    
    Manages the application lifecycle, configuration, and user interaction.
    """
    
    def __init__(self, config_dir: Optional[Path] = None, enable_lpra: bool = True):
        """
        Initialize the playground application.
        
        Args:
            config_dir: Optional custom configuration directory
            enable_lpra: Whether to enable LPRA architecture (default: True)
        """
        self.config_manager = ConfigManager(config_dir)
        self.graph: Optional[ImprovementGraph] = None
        self.lpra_manager: Optional[LPRAManager] = None
        self.dashboard: Optional[Dashboard] = None
        self.rate_limiter = RateLimiter(max_calls=100, time_window=3600)  # 100 calls per hour
        self.enable_lpra = enable_lpra
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"PlaygroundApp initialized (LPRA: {'enabled' if enable_lpra else 'disabled'})")
    
    def initialize(self) -> None:
        """
        Initialize the application components.
        
        Raises:
            ConfigurationError: If configuration loading fails
            PlaygroundException: If initialization fails
        """
        try:
            logger.info("Initializing playground components...")
            
            # Load and validate configuration
            config = self.config_manager.load_config()
            logger.info("Configuration loaded successfully")
            
            # Initialize LPRA if enabled
            if self.enable_lpra:
                try:
                    self.lpra_manager = LPRAManager()
                    self.graph = LPRAIntegratedGraph(config, self.lpra_manager)
                    logger.info("LPRA architecture initialized successfully")
                except Exception as e:
                    logger.warning(f"LPRA initialization failed, falling back to standard graph: {e}")
                    self.graph = ImprovementGraph(config)
                    self.enable_lpra = False
            else:
                self.graph = ImprovementGraph(config)
            
            self.dashboard = Dashboard()
            
            logger.info("Playground initialization completed")
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise PlaygroundException(f"Failed to initialize playground: {e}")
    
    def run_interactive(self) -> None:
        """
        Run the interactive command-line interface.
        
        Provides a loop for user interaction with safety checks and rate limiting.
        """
        if not self.graph or not self.dashboard:
            raise PlaygroundException("Application not initialized. Call initialize() first.")
        
        logger.info("Starting interactive mode")
        print("🚀 AI Code Improvement Playground")
        print("=" * 50)
        print("Enter coding tasks for AI-assisted development.")
        print("Type 'help' for commands, 'quit' to exit.")
        print("=" * 50)
        
        while True:
            try:
                # Check rate limiting
                if not self.rate_limiter.is_allowed():
                    remaining = self.rate_limiter.get_remaining_calls()
                    print(f"⚠️  Rate limit exceeded. {remaining} calls remaining in current window.")
                    continue
                
                # Get user input
                task = input("\n📝 Enter coding task: ").strip()
                
                if not task:
                    continue
                
                # Handle special commands
                if task.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif task.lower() == 'help':
                    self._show_help()
                    continue
                elif task.lower() == 'stats':
                    self._show_stats()
                    continue
                elif task.lower() == 'config':
                    self._show_config()
                    continue
                elif task.lower() == 'summary':
                    self._show_lpra_summary()
                    continue
                
                # Process the task
                self._process_task(task)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                print(f"❌ Error: {e}")
                print("Please try again or type 'help' for assistance.")
    
    def _process_task(self, task: str) -> None:
        """
        Process a user task through the improvement cycle.
        
        Args:
            task: User task description
        """
        try:
            # Validate input
            sanitized_task = InputValidator.sanitize_task_input(task)
            
            logger.info(f"Processing task: {sanitized_task[:100]}...")
            print(f"🔄 Processing: {sanitized_task[:100]}...")
            
            # Run improvement cycle
            result = self.graph.cycle(sanitized_task)
            
            # Display results
            self.dashboard.display_cycle(result)
            
            # Safety reminder
            print("\n⚠️  SAFETY REMINDER:")
            print("   • Review all generated code before use")
            print("   • Test in a safe environment")
            print("   • Never execute untrusted code")
            
            logger.info("Task processing completed successfully")
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            print(f"❌ Task processing failed: {e}")
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = f"""
📚 Available Commands:
   help     - Show this help message
   stats    - Show system statistics
   config   - Show current configuration
   {'summary  - Show LPRA system summary' if self.enable_lpra else ''}
   quit/exit/q - Exit the application

💡 Usage Tips:
   • Be specific in your task descriptions
   • Review all generated code carefully
   • Use the static analysis results to improve code quality
   • Remember: this system never executes code automatically

🛡️ Safety Features:
   • All code generation is human-supervised
   • Static analysis only (no code execution)
   • Input validation and sanitization
   • Rate limiting to prevent abuse
   • Comprehensive logging for audit trails
   {'• LPRA persistent reasoning and memory' if self.enable_lpra else ''}

🧠 LPRA Features {'(Enabled)' if self.enable_lpra else '(Disabled)'}:
   {'• Semantic graph for knowledge representation' if self.enable_lpra else ''}
   {'• Structured state management and versioning' if self.enable_lpra else ''}
   {'• Dynamic context compression for LLMs' if self.enable_lpra else ''}
   {'• Human-readable progress summaries' if self.enable_lpra else ''}
"""
        print(help_text)
    
    def _show_stats(self) -> None:
        """Show system statistics."""
        try:
            if self.graph:
                stats = self.graph.get_stats()
                print("\n📊 System Statistics:")
                print(f"   • Total cycles: {stats.get('total_cycles', 0)}")
                print(f"   • Successful cycles: {stats.get('successful_cycles', 0)}")
                print(f"   • Average cycle time: {stats.get('average_cycle_time', 0):.2f}s")
            
            # Model factory stats
            model_stats = ModelFactory.get_memory_usage()
            print(f"   • Cached models: {model_stats['total_models']}")
            print(f"   • Memory usage: {model_stats['process_memory_mb']:.1f} MB")
            
            # Rate limiter stats
            remaining = self.rate_limiter.get_remaining_calls()
            print(f"   • Rate limit remaining: {remaining} calls")
            
            # LPRA stats if enabled
            if self.enable_lpra and self.lpra_manager:
                lpra_stats = self.lpra_manager.get_system_statistics()
                lpra_manager_stats = lpra_stats.get("lpra_manager", {})
                print(f"   • LPRA nodes created: {lpra_manager_stats.get('nodes_created', 0)}")
                print(f"   • LPRA edges created: {lpra_manager_stats.get('edges_created', 0)}")
                print(f"   • LPRA states computed: {lpra_manager_stats.get('states_computed', 0)}")
                print(f"   • LPRA contexts generated: {lpra_manager_stats.get('contexts_generated', 0)}")
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            print(f"❌ Failed to get statistics: {e}")
    
    def _show_config(self) -> None:
        """Show current configuration."""
        try:
            config = self.config_manager.load_config()
            print("\n⚙️  Current Configuration:")
            print(f"   • Model: {config['model'].name}")
            print(f"   • Engine: {config['model'].engine}")
            print(f"   • Max tokens: {config['model'].max_tokens}")
            print(f"   • Max cycles: {config['cycles'].max_cycles}")
            print(f"   • Human approval: {config['cycles'].require_human_approval}")
            
        except Exception as e:
            logger.error(f"Failed to show config: {e}")
            print(f"❌ Failed to show configuration: {e}")
    
    def _show_lpra_summary(self) -> None:
        """Show LPRA system summary."""
        if not self.enable_lpra or not self.lpra_manager:
            print("❌ LPRA is not enabled or not initialized")
            return
        
        try:
            summary = self.lpra_manager.get_human_summary()
            print("\n" + summary)
            
        except Exception as e:
            logger.error(f"Failed to get LPRA summary: {e}")
            print(f"❌ Failed to get LPRA summary: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        print(f"\n🛑 Received shutdown signal, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self) -> None:
        """
        Cleanup application resources.
        
        Should be called before application shutdown.
        """
        logger.info("Starting application cleanup...")
        
        try:
            # Cleanup model factory
            ModelFactory.cleanup_all()
            
            # Cleanup LPRA if initialized
            if self.lpra_manager:
                self.lpra_manager.cleanup()
            
            # Cleanup graph if initialized
            if self.graph:
                self.graph.cleanup()
            
            logger.info("Application cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """
    Main entry point for the playground application.
    
    Handles initialization, execution, and cleanup.
    """
    app = None
    
    try:
        # Initialize application
        app = PlaygroundApp()
        app.initialize()
        
        # Run interactive mode
        app.run_interactive()
        
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please check your configuration files and try again.")
        sys.exit(1)
        
    except PlaygroundException as e:
        print(f"❌ Playground Error: {e}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
        
    finally:
        # Ensure cleanup happens
        if app:
            app.cleanup()


if __name__ == "__main__":
    main()
