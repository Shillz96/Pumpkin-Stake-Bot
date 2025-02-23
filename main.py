# Add these back at the top of the file
import os
import time
import random
import base58
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction
from solders.message import Message
from solders.hash import Hash
from solana.rpc.api import Client
from solana.rpc.commitment import Commitment
from cryptography.fernet import Fernet, InvalidToken
import logging
from dotenv import load_dotenv
from functools import wraps
import asyncio
import json

# Move these class definitions to the top, after imports but before other code
class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.profits = []
        self.start_balance = 0
        
    async def track_trade(self, trade_type, amount, token, price):
        self.trades.append({
            "timestamp": time.time(),
            "type": trade_type,
            "amount": amount,
            "token": token,
            "price": price
        })
    
    async def calculate_metrics(self):
        return {
            "total_profit": sum(self.profits),
            "trade_count": len(self.trades),
            "win_rate": len([p for p in self.profits if p > 0]) / len(self.profits) if self.profits else 0,
            "average_profit": sum(self.profits) / len(self.profits) if self.profits else 0
        }

class UserPreferences:
    def __init__(self):
        self.risk_tolerance = "medium"  # low, medium, high
        self.auto_compound = True
        self.min_apy = 5.0
        self.max_stake_per_pool = 0.3  # 30% of total
        self.notification_preferences = {
            "price_alerts": True,
            "apy_changes": True,
            "auto_switches": True
        }

class UserStats:
    def __init__(self):
        self.total_staked = 0
        self.total_earned = 0
        self.best_trade = 0
        self.worst_trade = 0
        self.trade_history = []
        
    async def add_trade(self, trade_type, amount, token, price, profit=0):
        self.trade_history.append({
            "timestamp": time.time(),
            "type": trade_type,
            "amount": amount,
            "token": token,
            "price": price,
            "profit": profit
        })
        
        if trade_type == "stake":
            self.total_staked += amount
        elif trade_type == "profit":
            self.total_earned += profit
            self.best_trade = max(self.best_trade, profit)
            self.worst_trade = min(self.worst_trade, profit)

# Global variables - ALL must be declared here first
CHAT_ID = None
WALLET = None
SOLANA_CLIENT = None
TELEGRAM_TOKEN = None
DEV_WALLET = None
SOLANA_PRIVATE_KEY = None
MOCK_STAKE_PUBKEY = None
MOCK_STAKE_PRIVATE_KEY = None
ENCRYPTION_KEY = None
SOLANA_RPC_URL = "https://api.devnet.solana.com"
WALLET_FILE = "wallet.dat"

# Global state tracking
stakes = {"$PKIN": {"amount": 0.0, "pools": {}}}
current_stake = {"pool_id": None, "apy": 0.0, "token": "$PKIN", "amount": 0.0}
start_time = time.time()
wallet_name = "Default Wallet"
current_pnl_period = 24
custom_stake_options = [0.1, 0.5, 1.0, "all"]

# Initialize these after environment variables are loaded
cipher_suite = None
logger = None

# After global variable declarations but before check_better_opportunities
# Mock data for trending tokens and staking pools
trending_tokens = [
    {"name": "SOL", "volume": "1.2M", "price": "$150"},
    {"name": "USDC", "volume": "800K", "price": "$1"},
    {"name": "SRM", "volume": "500K", "price": "$0.50"}
]

mock_pools = [
    {
        "pool_id": "1", 
        "token": "$PKIN", 
        "apy": "5.2", 
        "price": 0.10,
        "sol_conversion_rate": 0.5,
        "price_1h_ago": 0.0995,
        "price_6h_ago": 0.099, 
        "price_12h_ago": 0.0985, 
        "price_24h_ago": 0.098
    },
    {
        "pool_id": "2", 
        "token": "$TEST", 
        "apy": "8.5",  # Higher APY
        "price": 0.15,
        "sol_conversion_rate": 0.4,
        "price_1h_ago": 0.145,
        "price_6h_ago": 0.14, 
        "price_12h_ago": 0.135, 
        "price_24h_ago": 0.13
    }
]

# Initialize trackers and preferences
performance_tracker = PerformanceTracker()
user_preferences = UserPreferences()
user_stats = UserStats()

# Add environment variable validation
def validate_env_vars():
    """Validate required environment variables are set."""
    required_vars = [
        "TELEGRAM_TOKEN", 
        "DEV_WALLET", 
        "SOLANA_PRIVATE_KEY",
        "MOCK_STAKE_PUBKEY",
        "MOCK_STAKE_PRIVATE_KEY"
    ]
    # Add debug prints
    print("Checking environment variables...")
    for var in required_vars:
        value = os.getenv(var)
        print(f"{var}: {'[SET]' if value else '[MISSING]'}")
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Add before load_dotenv()
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Looking for .env file at: {env_path}")
if not os.path.exists(env_path):
    raise FileNotFoundError(f".env file not found at {env_path}")

print("Loading environment variables...")
load_dotenv(env_path)  # Explicitly specify the path
validate_env_vars()

# Add these debug prints
print(f"Loaded TELEGRAM_TOKEN: {os.getenv('TELEGRAM_TOKEN')}")
print(f"Direct TELEGRAM_TOKEN value: {TELEGRAM_TOKEN}")

# Configure logging (console only, no file)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Environment variables for sensitive data
WALLET_FILE = "wallet.dat"  # File to store encrypted wallet private key
# Mock stake account for testing (replace with real Devnet keypair)
MOCK_STAKE_PUBKEY = Pubkey.from_string(os.getenv("MOCK_STAKE_PUBKEY"))
MOCK_STAKE_PRIVATE_KEY = os.getenv("MOCK_STAKE_PRIVATE_KEY")

# Encryption setup
try:
    print("Loading or generating encryption key...")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    cipher_suite = Fernet(ENCRYPTION_KEY)
    print("Encryption key loaded successfully.")
except Exception as e:
    logger.error(f"Encryption key error: {str(e)}", exc_info=True)
    exit(1)

# Function to load wallet from file or environment
def load_wallet():
    """Load wallet from file if exists, otherwise use SOLANA_PRIVATE_KEY."""
    try:
        if os.path.exists(WALLET_FILE):
            # Check file permissions (Unix-like systems)
            if hasattr(os, 'chmod'):
                current_perms = os.stat(WALLET_FILE).st_mode
                if current_perms & 0o077:  # Check if group/others have any access
                    logger.warning("Wallet file has unsafe permissions. Fixing...")
                    os.chmod(WALLET_FILE, 0o600)  # Set to user read/write only
            
            with open(WALLET_FILE, "rb") as f:
                encrypted_key = f.read()
            try:
                private_key = cipher_suite.decrypt(encrypted_key).decode()
                print(f"Loaded wallet from {WALLET_FILE} (redacted): {private_key[:8]}...")
            except (InvalidToken, ValueError) as e:
                logger.error(f"Failed to decrypt wallet file: {str(e)}. Falling back to environment variable.")
                print(f"Failed to decrypt wallet file: {str(e)}. Falling back to environment variable.")
                private_key = os.getenv("SOLANA_PRIVATE_KEY")
                if not private_key:
                    raise ValueError("SOLANA_PRIVATE_KEY environment variable is not set")
                print("Using environment variable for private key")
        else:
            private_key = os.getenv("SOLANA_PRIVATE_KEY")
            if not private_key:
                raise ValueError("SOLANA_PRIVATE_KEY environment variable is not set")
            print("Using environment variable for private key")
            
        wallet = Keypair.from_base58_string(private_key)
        print(f"Wallet public key: {wallet.pubkey()}")
        return wallet
        
    except Exception as e:
        logger.error(f"Failed to load wallet: {str(e)}", exc_info=True)
        print(f"Wallet load error: {str(e)}")
        exit(1)

# Function to save wallet to file
def save_wallet(wallet):
    """Save encrypted wallet private key to file."""
    try:
        private_key_bytes = wallet.secret()
        private_key = base58.b58encode(private_key_bytes).decode()
        encrypted_key = cipher_suite.encrypt(private_key.encode())
        with open(WALLET_FILE, "wb") as f:
            f.write(encrypted_key)
        logger.info(f"Wallet saved to {WALLET_FILE} (redacted): {private_key[:8]}...")
    except AttributeError:
        try:
            private_key_bytes = wallet.to_bytes() if hasattr(wallet, 'to_bytes') else wallet.sk.to_bytes()
            private_key = base58.b58encode(private_key_bytes).decode()
            encrypted_key = cipher_suite.encrypt(private_key.encode())
            with open(WALLET_FILE, "wb") as f:
                f.write(encrypted_key)
            logger.info(f"Wallet saved to {WALLET_FILE} (redacted): {private_key[:8]}...")
        except AttributeError:
            raise AttributeError("Unable to serialize Keypair to Base58 string or bytes")
    except Exception as e:
        logger.error(f"Failed to save wallet: {str(e)}", exc_info=True)
        raise

# Initialize Solana wallet and client
try:
    print("Initializing wallet...")
    WALLET = load_wallet()
    SOLANA_CLIENT = Client(SOLANA_RPC_URL, commitment=Commitment("confirmed"))
except Exception as e:
    logger.error(f"Failed to initialize wallet or client: {str(e)}", exc_info=True)
    exit(1)

# Add after the mock_pools definition

# Add rate limiting decorator
def rate_limit(max_per_second):
    """Rate limiting decorator."""
    min_interval = 1.0 / float(max_per_second)
    last_called = {}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            key = func.__name__
            if key in last_called:
                elapsed = now - last_called[key]
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
            last_called[key] = time.time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Apply rate limiting to API calls
@rate_limit(2)  # 2 calls per second max
async def get_sol_balance():
    """Retrieve the SOL balance of the wallet."""
    try:
        response = SOLANA_CLIENT.get_balance(WALLET.pubkey())
        balance = response.value / 1_000_000_000
        return balance
    except Exception as e:
        logger.error(f"Error fetching balance: {str(e)}", exc_info=True)
        return 0.0

def calculate_pnl(hours):
    """Calculate profit and loss for staked $PKIN over a given period with mock APY."""
    if not current_stake["amount"]:
        return 0.0
        
    total_pnl = 0
    elapsed_time = max((time.time() - start_time) / 3600, 0.1)  # At least 0.1 hours to show some progress
    
    # Handle single current stake
    pool = next((p for p in mock_pools if p["pool_id"] == current_stake["pool_id"]), None)
    if not pool:
        return 0.0
        
    amount = current_stake["amount"]
    apy = float(pool["apy"]) / 100  # Convert to decimal
    hourly_yield = apy / (365 * 24)  # Hourly yield
    reward = amount * hourly_yield * min(elapsed_time, hours)  # Cap at requested period
    
    # Calculate value change
    current_value = amount * pool["price"] + reward
    price_key = f"price_{hours}h_ago"
    old_price = pool.get(price_key, pool["price"])  # Fallback to current price if historical not available
    value_ago = amount * old_price
    
    total_pnl = current_value - value_ago
    return total_pnl

# Wallet Management
async def wallet_menu(message, context):
    """Display wallet information and options."""
    sol_balance = await get_sol_balance()
    keyboard = [
        [InlineKeyboardButton("Create New Wallet", callback_data="create_wallet"),
         InlineKeyboardButton("Upload Wallet", callback_data="upload_wallet")],
        [InlineKeyboardButton("Change Wallet Name", callback_data="change_wallet_name")],
        [InlineKeyboardButton("Close", callback_data="close")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await message.reply_text(
        "🤑 Wallet Info:\n"
        f"Name: {wallet_name}\n"
        f"Public Key: {str(WALLET.pubkey())}\n"
        f"SOL Balance: {sol_balance:.4f} SOL\n\n"
        "Options below:",
        reply_markup=reply_markup
    )

async def create_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a new Solana wallet and save it."""
    new_wallet = Keypair()
    private_key_bytes = new_wallet.secret()
    private_key = base58.b58encode(private_key_bytes).decode()
    private_key_encrypted = cipher_suite.encrypt(private_key.encode())
    context.user_data["temp_wallet"] = {"public_key": str(new_wallet.pubkey()), "private_key_encrypted": private_key_encrypted}
    global WALLET
    WALLET = new_wallet
    save_wallet(WALLET)
    keyboard = [[InlineKeyboardButton("Close", callback_data="close")]]
    await update.callback_query.message.reply_text(
        "🤑 New Wallet Created:\n"
        f"Name: {wallet_name}\n"
        f"Public Key: {str(new_wallet.pubkey())}\n"
        "Private Key (Encrypted): [Redacted]\n"
        "Note: Fund this wallet on Devnet to use it.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    logger.info(f"New wallet created and saved: {str(new_wallet.pubkey())[:8]}...")

async def upload_wallet_warning(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Warn user before uploading a wallet private key."""
    keyboard = [
        [InlineKeyboardButton("Proceed", callback_data="proceed_upload"),
         InlineKeyboardButton("Cancel", callback_data="close")]
    ]
    await update.callback_query.message.reply_text(
        "⚠️ **Security Warning:**\n\n"
        "Uploading a private key is sensitive. It will be encrypted in memory and saved securely. Proceed only if you trust this environment.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def upload_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Prompt user to send their private key."""
    await update.callback_query.message.reply_text(
        "Send your Solana private key (Base58) as text. It will be encrypted and saved.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="close")]])
    )
    context.user_data["waiting_for_private_key"] = True

async def change_wallet_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Prompt user to change wallet name."""
    await update.callback_query.message.reply_text(
        "Enter new wallet name:",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="cancel_wallet_name")]])
    )
    context.user_data["waiting_for_wallet_name"] = True

# Staking Functions
async def stake_instructions(message, context):
    """Show staking instructions."""
    keyboard = [
        [InlineKeyboardButton("Proceed to Stake", callback_data="stake_menu")],
        [InlineKeyboardButton("Close", callback_data="close")]
    ]
    await message.reply_text(
        "🏦 **Staking Instructions:**\n\n"
        "Stake SOL to auto-invest in $PKIN. The bot will optimize your stake across top-performing tokens.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def stake_menu(message, context):
    """Display staking options."""
    global custom_stake_options
    sol_balance = await get_sol_balance()
    keyboard = [
        [InlineKeyboardButton(f"{opt:.1f} SOL", callback_data=f"stake_amount_{opt}") if isinstance(opt, (int, float))
         else InlineKeyboardButton("All SOL", callback_data=f"stake_amount_{sol_balance:.4f}")
         for opt in custom_stake_options[i:i+2]]
        for i in range(0, len(custom_stake_options), 2)
    ]
    keyboard.append([InlineKeyboardButton("Settings", callback_data="settings_stake"),
                     InlineKeyboardButton("Close", callback_data="close")])
    await message.reply_text(
        f"🏦 Stake Amount (Available: {sol_balance:.4f} SOL):",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def get_best_pool():
    """Get the pool with the highest APY."""
    return max(mock_pools, key=lambda x: float(x["apy"]))

async def stake_amount(message, context, amount):
    """Process staking transaction with fee using mock stake account."""
    global current_stake, stakes
    
    # Add input validation
    if not isinstance(amount, (int, float)) or amount <= 0:
        await message.reply_text("🏦 Invalid stake amount. Must be a positive number.")
        return
        
    sol_balance = await get_sol_balance()
    if amount > sol_balance:
        await message.reply_text("🏦 Insufficient SOL balance.")
        return
    
    fee_amount = amount * 0.01  # 1% dev fee
    amount_after_fee = amount - fee_amount
    
    # Get the best pool for initial stake
    pool = await get_best_pool() if not current_stake["pool_id"] else mock_pools[0]
    logger.info(f"Selected pool: {pool['token']} with APY: {pool['apy']}%")
    
    try:
        # New checks
        market_conditions = await analyze_market_conditions()
        risk_metrics = await check_risk_metrics(pool)
        price_impact = await analyze_price_impact(pool, amount)
        
        if not price_impact["is_safe"] and user_preferences.risk_tolerance != "high":
            await message.reply_text(
                f"⚠️ Warning: Large price impact detected ({price_impact['impact_percentage']:.1f}%)\n"
                f"Recommended amount: {price_impact['recommended_amount']:.2f} SOL"
            )
            return
            
        if not risk_metrics["is_safe"] and user_preferences.risk_tolerance != "high":
            warnings = "\n".join(f"- {w}" for w in risk_metrics["warnings"])
            await message.reply_text(
                f"⚠️ Risk Warning:\n{warnings}\n"
                "Proceed with caution or adjust amount."
            )
            return
            
        # If auto-distribution is enabled and amount is large enough
        if user_preferences.max_stake_per_pool < 1.0 and amount > 1.0:
            distribution = await distribute_stake(amount)
            for stake in distribution:
                await stake_amount(message, context, stake["amount"])
            return

        # Convert SOL to token amount using conversion rate
        token_amount = amount_after_fee * pool["sol_conversion_rate"]
        
        # Create transfer instruction
        transfer_params = TransferParams(
            from_pubkey=WALLET.pubkey(),
            to_pubkey=MOCK_STAKE_PUBKEY,
            lamports=int(amount_after_fee * 1_000_000_000)
        )
        instruction = transfer(transfer_params)

        # Get recent blockhash
        recent_blockhash_response = SOLANA_CLIENT.get_latest_blockhash()
        blockhash = recent_blockhash_response.value.blockhash

        # Create message
        solana_message = Message.new_with_blockhash(
            instructions=[instruction],
            blockhash=blockhash,
            payer=WALLET.pubkey()
        )

        # Create and sign transaction
        transaction = Transaction(
            from_keypairs=[WALLET],
            message=solana_message,
            recent_blockhash=blockhash
        )

        # Send transaction
        result = SOLANA_CLIENT.send_transaction(transaction)
        
        # Update internal state
        stakes["$PKIN"]["pools"][pool["pool_id"]] = {
            "amount": stakes["$PKIN"]["pools"].get(pool["pool_id"], {"amount": 0})["amount"] + token_amount
        }
        current_stake = {
            "pool_id": pool["pool_id"],
            "apy": float(pool["apy"]),
            "token": pool["token"],
            "amount": current_stake["amount"] + token_amount
        }
        save_state()
        
        # Track the trade
        await performance_tracker.track_trade(
            "stake",
            token_amount,
            pool["token"],
            pool["price"]
        )
        
        # After successful stake, add to history
        add_to_history(pool["token"], token_amount, 0)  # Initial profit is 0
        save_state()
        
        # Show both SOL spent and tokens received in message
        await message.reply_text(
            f"🏦 Staked {amount_after_fee:.4f} SOL for {token_amount:.2f} {pool['token']}\n"
            f"Rate: 1 SOL = {pool['sol_conversion_rate']} {pool['token']}\n"
            f"Value: ${token_amount * pool['price']:.2f}\n"
            f"Pool: {pool['pool_id']} (APY: {pool['apy']}%)\n"
            f"Tx: {result.value}"
        )
        
    except Exception as e:
        logger.error(f"Staking error: {str(e)}", exc_info=True)
        await message.reply_text(f"🏦 Staking failed: {str(e)}")

async def unstake_menu(message, context):
    """Display unstaking options."""
    if not current_stake["pool_id"]:
        await message.reply_text("❌ No stakes to unstake.")
        return
    staked_amount = current_stake["amount"]
    keyboard = [
        [InlineKeyboardButton("0.1 $PKIN", callback_data="unstake_amount_0.1"),
         InlineKeyboardButton("0.5 $PKIN", callback_data="unstake_amount_0.5")],
        [InlineKeyboardButton("1 $PKIN", callback_data="unstake_amount_1.0"),
         InlineKeyboardButton("All", callback_data=f"unstake_amount_{staked_amount:.2f}")],
        [InlineKeyboardButton("Close", callback_data="close")]
    ]
    await message.reply_text(
        f"❌ Unstake Amount (Staked: {staked_amount:.2f} $PKIN):",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def unstake_amount(message, context, amount):
    """Process unstaking with fee using mock stake account."""
    global current_stake, stakes
    if not current_stake["pool_id"] or amount > current_stake["amount"]:
        await message.reply_text("❌ Insufficient staked amount or no active stakes.")
        return
    
    try:
        fee_amount = amount * 0.01
        amount_after_fee = amount - fee_amount
        pool_id = current_stake["pool_id"]
        
        # Get recent blockhash
        recent_blockhash_response = SOLANA_CLIENT.get_latest_blockhash()
        blockhash = recent_blockhash_response.value.blockhash

        # Create unstake instruction (transfer from mock stake account back to user)
        mock_stake_wallet = Keypair.from_base58_string(MOCK_STAKE_PRIVATE_KEY)
        transfer_params = TransferParams(
            from_pubkey=MOCK_STAKE_PUBKEY,
            to_pubkey=WALLET.pubkey(),
            lamports=int(amount_after_fee * 1_000_000_000)
        )
        instruction = transfer(transfer_params)

        # Create message
        solana_message = Message.new_with_blockhash(
            instructions=[instruction],
            blockhash=blockhash,
            payer=MOCK_STAKE_PUBKEY
        )

        # Create and sign transaction
        transaction = Transaction(
            from_keypairs=[mock_stake_wallet],
            message=solana_message,
            recent_blockhash=blockhash
        )

        # Send transaction
        result = SOLANA_CLIENT.send_transaction(transaction)
        
        # Update internal state
        stakes["$PKIN"]["pools"][pool_id]["amount"] -= amount
        current_stake["amount"] -= amount_after_fee
        save_state()
        
        if current_stake["amount"] <= 0:
            del stakes["$PKIN"]["pools"][pool_id]
            current_stake.update({"pool_id": None, "apy": 0.0, "amount": 0.0})
            msg = f"❌ Fully unstaked {amount_after_fee:.2f} $PKIN"
        else:
            msg = f"❌ Unstaked {amount_after_fee:.2f} $PKIN from Pool {pool_id}"
        
        # Calculate profit before unstaking
        pool = next((p for p in mock_pools if p["pool_id"] == current_stake["pool_id"]), None)
        if pool:
            initial_value = amount * pool["price_24h_ago"]
            final_value = amount * pool["price"]
            profit = final_value - initial_value
            add_to_history(pool["token"], amount, profit)
            save_state()
        
        await message.reply_text(
            f"{msg}. Fee: {fee_amount:.2f} SOL sent to dev wallet. Tx: {result.value}"
        )
        logger.info(f"Mock unstake successful: {amount_after_fee:.2f} SOL from {MOCK_STAKE_PUBKEY}")

    except Exception as e:
        logger.error(f"Mock unstaking error: {str(e)}", exc_info=True)
        await message.reply_text(f"❌ Unstaking failed: {str(e)}")

async def send_dev_fee(amount, sender_wallet):
    """Send 1% fee to dev wallet via Solana transaction."""
    try:
        dev_pubkey = Pubkey.from_string(DEV_WALLET)
        recent_blockhash_response = SOLANA_CLIENT.get_latest_blockhash()
        recent_blockhash = recent_blockhash_response.value.blockhash  # Ensure it's a Hash object
        
        transfer_params = TransferParams(
            from_pubkey=sender_wallet.pubkey(),
            to_pubkey=dev_pubkey,
            lamports=int(amount * 1_000_000_000)
        )
        instruction = transfer(transfer_params)
        
        print(f"Type of sender_wallet: {type(sender_wallet)}")  # Debug
        print(f"Type of sender_wallet.pubkey(): {type(sender_wallet.pubkey())}")  # Debug
        print(f"Value of sender_wallet.pubkey(): {sender_wallet.pubkey()}")  # Debug
        print(f"Type of recent_blockhash: {type(recent_blockhash)}")  # Debug
        print(f"Value of recent_blockhash: {recent_blockhash}")  # Debug
        
        payer_pubkey = sender_wallet.pubkey()  # Ensure it's a Pubkey
        print(f"Type of payer_pubkey before conversion: {type(payer_pubkey)}")  # Debug
        print(f"Value of payer_pubkey before conversion: {payer_pubkey}")  # Debug
        
        if not isinstance(payer_pubkey, Pubkey):
            if isinstance(payer_pubkey, str):
                payer_pubkey = Pubkey.from_string(payer_pubkey)
                print(f"Converted payer_pubkey to Pubkey: {payer_pubkey}")
            elif isinstance(payer_pubkey, Hash):
                raise ValueError(f"Unexpected Hash object as payer: {payer_pubkey}. Expected Pubkey from sender_wallet.pubkey()")
            else:
                raise ValueError(f"Unexpected type for payer_pubkey: {type(payer_pubkey)}. Value: {payer_pubkey}")
        
        # Final workaround: Ensure payer_pubkey is a Pubkey by string conversion
        payer_pubkey_str = str(payer_pubkey)
        payer_pubkey = Pubkey.from_string(payer_pubkey_str)
        
        print(f"Type of payer_pubkey after final conversion: {type(payer_pubkey)}")  # Debug
        print(f"Value of payer_pubkey after final conversion: {payer_pubkey}")  # Debug
        
        # Create Solana message (renamed to avoid conflict with Telegram message)
        solana_message = Message.new_with_blockhash(
            instructions=[instruction],
            blockhash=recent_blockhash,
            payer=payer_pubkey  # Use 'payer' correctly
        )
        # Create transaction directly from message and sign
        transaction = Transaction(solana_message)  # Initialize Transaction with Message
        transaction.sign([sender_wallet])  # Sign with sender_wallet keypair
        result = SOLANA_CLIENT.send_transaction(transaction)
        logger.info(f"Dev fee sent: {result.value} for {amount:.2f} SOL")
    except Exception as e:
        logger.error(f"Dev fee error: {str(e)}", exc_info=True)
        raise

async def simulate_transaction_fee(amount, sender_wallet):
    """Simulate transaction fee deduction (now unused, kept for reference)."""
    try:
        fee_lamports = 5000
        fee_sol = fee_lamports / 1_000_000_000
        balance = await get_sol_balance()
        if balance < (amount + fee_sol):
            raise ValueError(f"Insufficient balance: Need {amount + fee_sol:.6f} SOL, have {balance:.6f} SOL")
        logger.info(f"Fee simulated: {fee_sol:.6f} SOL for {amount:.2f} SOL")
    except Exception as e:
        logger.error(f"Fee simulation error: {str(e)}", exc_info=True)
        raise

# Dashboard and Info
async def show_dashboard(message, context):
    """Display the main dashboard."""
    global current_stake, stakes
    total_value = sum(data["amount"] * pool["price"] for pool_id, data in stakes["$PKIN"]["pools"].items()
                      for pool in mock_pools if pool["pool_id"] == pool_id)
    sol_balance = await get_sol_balance()
    pnl = calculate_pnl(current_pnl_period)
    pnl_text = f"📈 +${abs(pnl):.2f}" if pnl > 0 else f"📉 -${abs(pnl):.2f}"
    
    keyboard = [[InlineKeyboardButton("Close", callback_data="close")]]
    await message.reply_text(
        f"# 🎃 Pumpkin.fun Auto Staking\n\n"
        f"🚀 _First Auto-Staking Platform_\n\n"
        f"💰 Wallet: {wallet_name}\n"
        f"- SOL Balance: {sol_balance:.4f} SOL\n"
        f"- Staked: {current_stake['amount']:.2f} {current_stake['token']} ({pnl_text} for {current_pnl_period}h)\n\n"
        f"Total Value: ${total_value:.2f} USD",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )

async def show_holdings(message, context):
    """Display current holdings."""
    global current_stake
    if not current_stake["pool_id"]:
        await message.reply_text("No active stakes.")
        return
        
    pool = next((p for p in mock_pools if p["pool_id"] == current_stake["pool_id"]), None)
    if not pool:
        await message.reply_text("Pool information not found.")
        return
        
    value = current_stake["amount"] * pool["price"]
    keyboard = [[InlineKeyboardButton("Close", callback_data="close")]]
    
    await message.reply_text(
        f"💰 Current Holdings:\n\n"
        f"Token: {current_stake['token']}\n"
        f"Amount: {current_stake['amount']:.2f}\n"
        f"Value: ${value:.2f}\n"
        f"APY: {current_stake['apy']}%",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def pnl_menu(message, context):
    """Show PnL period selection menu."""
    keyboard = [
        [InlineKeyboardButton("1h", callback_data="pnl_1h"),
         InlineKeyboardButton("6h", callback_data="pnl_6h")],
        [InlineKeyboardButton("12h", callback_data="pnl_12h"),
         InlineKeyboardButton("24h", callback_data="pnl_24h")],
        [InlineKeyboardButton("Close", callback_data="close")]
    ]
    await message.reply_text(
        "📊 Select PnL Period:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def show_current_stake(update, context):
    """Show current stake information."""
    global current_stake
    if not current_stake["pool_id"]:
        if update.callback_query:
            await update.callback_query.message.reply_text("No active stakes.")
        else:
            await update.message.reply_text("No active stakes.")
        return
        
    pool = next((p for p in mock_pools if p["pool_id"] == current_stake["pool_id"]), None)
    if not pool:
        await update.message.reply_text("Pool information not found.")
        return
        
    value = current_stake["amount"] * pool["price"]
    pnl = calculate_pnl(current_pnl_period)
    keyboard = [[InlineKeyboardButton("Close", callback_data="close")]]
    
    message_text = (
        f"Current Stake:\n\n"
        f"Token: {current_stake['token']}\n"
        f"Amount: {current_stake['amount']:.2f}\n"
        f"Value: ${value:.2f}\n"
        f"APY: {current_stake['apy']}%\n"
        f"PnL ({current_pnl_period}h): ${pnl:.2f}"
    )
    
    if update.callback_query:
        await update.callback_query.message.reply_text(
            message_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.message.reply_text(
            message_text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

# Settings
async def settings_stake(message, context):
    """Show stake settings menu."""
    keyboard = [
        [InlineKeyboardButton("Add Custom Amount", callback_data="add_custom_stake"),
         InlineKeyboardButton("Remove Amount", callback_data="remove_stake_option")],
        [InlineKeyboardButton("Reset to Default", callback_data="reset_stake_options")],
        [InlineKeyboardButton("Close", callback_data="close")]
    ]
    current_options = ", ".join(str(opt) for opt in custom_stake_options)
    await message.reply_text(
        f"🔧 Stake Settings\n\nCurrent options: {current_options}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def add_custom_stake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add custom stake amount option."""
    await update.callback_query.message.reply_text(
        "Enter new stake amount in SOL:",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Cancel", callback_data="close")]])
    )

async def remove_stake_option(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Remove stake amount option."""
    keyboard = [
        [InlineKeyboardButton(f"Remove {opt}", callback_data=f"remove_{opt}") 
         for opt in custom_stake_options if opt != "all"]
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data="close")])
    await update.callback_query.message.reply_text(
        "Select amount to remove:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# Add this to your global variables section
token_history = []  # To store history of tokens and profits

# Add this function to track token history
def add_to_history(token, amount, profit):
    """Add token trade to history."""
    token_history.append({
        "token": token,
        "amount": amount,
        "profit": profit,
        "timestamp": time.time()
    })
    # Keep only last 5 entries
    while len(token_history) > 5:
        token_history.pop(0)

# Modify the save_state function to include token history
def save_state():
    """Save current state to file."""
    state = {
        "current_stake": current_stake,
        "stakes": stakes,
        "start_time": start_time,
        "wallet_name": wallet_name,
        "token_history": token_history  # Add this line
    }
    try:
        with open("bot_state.json", "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

# Modify the load_state function to load token history
def load_state():
    """Load state from file."""
    global current_stake, stakes, start_time, wallet_name, token_history
    try:
        with open("bot_state.json", "r") as f:
            state = json.load(f)
            current_stake = state["current_stake"]
            stakes = state["stakes"]
            start_time = state.get("start_time", time.time())
            wallet_name = state.get("wallet_name", "Default Wallet")
            token_history = state.get("token_history", [])
    except FileNotFoundError:
        # If no saved state exists, use defaults
        pass
    except Exception as e:
        logger.error(f"Failed to load state: {e}")

# Move this function before main()
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Bot start command."""
    global CHAT_ID, current_stake
    if update.message:
        CHAT_ID = update.message.chat_id
    elif update.callback_query:
        CHAT_ID = update.callback_query.message.chat_id
    else:
        logger.error("Unexpected update type: neither message nor callback_query")
        return

    sol_balance = await get_sol_balance()
    pnl = calculate_pnl(current_pnl_period)
    pnl_text = f"📈 +${abs(pnl):.2f}" if pnl > 0 else f"📉 -${abs(pnl):.2f}"
    
    # Format history text
    history_text = "\n📜 Token History:\n" if token_history else ""
    for entry in reversed(token_history):
        profit_text = f"📈 +${entry['profit']:.2f}" if entry['profit'] > 0 else f"📉 -${abs(entry['profit']):.2f}"
        history_text += f"• {entry['token']}: {entry['amount']:.2f} ({profit_text})\n"
    
    keyboard = [
        [InlineKeyboardButton("🏦 Stake", callback_data="stake_instructions"), 
         InlineKeyboardButton("❌ Unstake", callback_data="unstake_menu")],
        [InlineKeyboardButton("💪 Holdings", callback_data="holdings"), 
         InlineKeyboardButton("💰 PnL", callback_data="pnl_menu")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="settings_menu"), 
         InlineKeyboardButton("🤑 Wallet", callback_data="wallet_menu")],
        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh")]
    ]
    
    message_text = (
        f"# 🎃 Pumpkin.fun Auto Staking\n\n"
        f"🚀 _First Auto-Staking Platform_\n\n"
        f"💰 Wallet: {wallet_name}\n"
        f"• SOL Balance: {sol_balance:.4f} SOL\n"
        f"• Staked: {current_stake['amount']:.2f} {current_stake['token']}\n"
        f"• PnL ({current_pnl_period}h): {pnl_text}\n"
        f"{history_text}"
    )
    
    if update.message:
        await update.message.reply_text(
            message_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    else:
        await update.callback_query.message.reply_text(
            message_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
        await update.callback_query.answer()

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks."""
    query = update.callback_query
    data = query.data

    if data == "close":
        await query.message.delete()
    elif data == "refresh":
        await start(update, context)
    elif data == "stake_instructions":
        await stake_instructions(query.message, context)
    elif data == "stake_menu":
        await stake_menu(query.message, context)
    elif data == "unstake_menu":
        await unstake_menu(query.message, context)
    elif data == "holdings":
        await show_holdings(query.message, context)
    elif data == "pnl_menu":
        await pnl_menu(query.message, context)
    elif data == "settings_menu":
        await settings_menu(query.message, context)
    elif data == "wallet_menu":
        await wallet_menu(query.message, context)
    elif data.startswith("stake_amount_"):
        amount = float(data.split("_")[-1])
        await stake_amount(query.message, context, amount)
    elif data.startswith("unstake_amount_"):
        amount = float(data.split("_")[-1])
        await unstake_amount(query.message, context, amount)
    elif data.startswith("pnl_"):
        hours = int(data.split("_")[1].replace("h", ""))
        global current_pnl_period
        current_pnl_period = hours
        await show_current_stake(update, context)

    await query.answer()

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages."""
    text = update.message.text.lower()
    
    if context.user_data.get("waiting_for_wallet_name", False):
        global wallet_name
        wallet_name = text
        context.user_data["waiting_for_wallet_name"] = False
        await update.message.reply_text(f"Wallet name changed to: {wallet_name}")
        save_state()
    elif text.startswith("/"):
        await update.message.reply_text("Unknown command. Use /start to see available options.")
    else:
        await update.message.reply_text("Please use the menu buttons to interact with the bot.")

# Then the main function
def main():
    """Run the Telegram bot."""
    print("Starting bot...")
    load_state()  # Add this line
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("Bot handlers added, starting polling...")
    application.run_polling()

if __name__ == "__main__":
    try:
        TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        if not TELEGRAM_TOKEN:
            raise ValueError("TELEGRAM_TOKEN not found in environment variables")
        main()
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}", exc_info=True)
        print(f"Main execution error: {str(e)}")