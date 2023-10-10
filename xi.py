import MetaTrader5 as mt5
import time
import datetime as date


def kozero_modify(trading_data: dict):
    open_positions = mt5.positions_get(symbol=symbol)

    if open_positions == None:
        print("No positions for", symbol, ", error code =", mt5.last_error())
    elif len(open_positions) > 0:
        # iterate over open positions and set SL/TP
        gamma = trading_data['gamma']
        beta = trading_data['beta']

        if gamma < 0:
            gamma = gamma * -1

        if beta < 0:
            beta = beta * -1

        for position in open_positions:
            if position.type == mt5.ORDER_TYPE_BUY:
                price = mt5.symbol_info_tick(trading_data['epsilon']).bid
                sl = gamma * 125 - price
                tp = beta * 350 + price
                if sl < 0:
                    sl = sl * -1

                if tp < 0:
                    tp = tp * -1
                take_profit = round((tp + 200), 2)
                stop_loss = round((sl - 50), 2)

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": position.type,
                    "position": position.ticket,
                    "price": position.price_open,
                    "sl": stop_loss,
                    "tp": take_profit,
                    "magic": position.magic,
                    "comment": "KodifyZero",
                    "type_time": mt5.ORDER_TIME_GTC,  # good till cancelled
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }

                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print("Failed to modify position :(", result)

            elif position.type == mt5.ORDER_TYPE_SELL:
                price = mt5.symbol_info_tick(trading_data['epsilon']).ask
                sl = beta * 1250 + price
                tp = gamma * 3500 - price
                if sl < 0:
                    sl = sl * -1

                if tp < 0:
                    tp = tp * -1
                take_profit = round((tp - 200), 2)
                stop_loss = round((sl + 50), 2)

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": position.type,
                    "position": position.ticket,
                    "price": position.price_open,
                    "sl": stop_loss,
                    "tp": take_profit,
                    "magic": position.magic,
                    "comment": "KodifyZero",
                    "type_time": mt5.ORDER_TIME_GTC,  # good till cancelled
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }

                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print("Failed to modify position :(", result)
def calculate_margin_level():
    """
    Calculate the Margin Level using:
    Margin Level = (Equity / Used Margin) x 100%
    """
    account_info = mt5.account_info()._asdict()
    equity = account_info['equity']
    used_margin = account_info['margin']

    if used_margin == 0:
        # Handle the case when no trades are open and used margin is zero.
        return 0  # Return a very high margin level to indicate no open trades.
    else:
        margin_level = (equity / used_margin) * 100
        return margin_level


def calculate_lot_size(equity, contract_size, open_price, leverage):
    target_margin_level = 5  # 300%
    
    # Check for zero or invalid values to prevent division by zero or negative values
    if contract_size <= 0:
        raise ValueError("Contract Size must be positive and non-zero")
    
    if open_price <= 0:
        raise ValueError("Open Price must be positive and non-zero")
        
    if leverage <= 0:
        raise ValueError("Leverage must be positive and non-zero")
    
    # Calculating the volume in lots
    lot_size = (equity * leverage) / (target_margin_level * contract_size * open_price)
    lot_size = round(lot_size, 2)
    # Ensure the lot size does not exceed max permissible limits
    return lot_size

def kozero_order(trading_data: dict):
    epsilon = trading_data['epsilon']
    account_info = mt5.account_info()._asdict()
    symbol_info = mt5.symbol_info(trading_data['epsilon'])
    if symbol_info is None:
        print("[Xi]", trading_data['market'], "not found, can not call order_check()")
        return None
    equity = account_info['balance']
    open_positions = mt5.positions_get(symbol=symbol)
    if open_positions:
        print("[Xi] Existing position open. Skipping new trade.")
        return
    # If the symbol is not available in the MarketWatch, we add it
    if not symbol_info.visible:
        print("[Xi]", trading_data['epsilon'], "is not visible, trying to switch on")
        if not mt5.symbol_select(trading_data['epsilon'], True):
            print("[Xi] symbol_select({}) failed, exit", trading_data['epsilon'])
            return None

    account_info = mt5.account_info()
    quote = mt5.symbol_info(symbol)
    symbol_specification = mt5.symbol_info(symbol)
    # Parameters
    equity = account_info.equity  # Your current equity
    contract_size = symbol_specification.trade_contract_size  # Contract size for the symbol
    open_price = quote.bid  # Price at which the position will be opened
    margin_percentage = symbol_specification.margin_initial / contract_size  # Margin requirement set by the broker
    leverage = 300  # Your account leverage
    deviation = 20
    gamma = trading_data['gamma']
    beta = trading_data['beta']
    order = trading_data['alpha']
    print(beta)
    print(gamma)

    account_info = mt5.account_info()._asdict()

    if order == -1:
        print("[Xi] No trade directive received. Skipping trade execution.")
        return
    
    if order == 0:
        price = mt5.symbol_info_tick(trading_data['epsilon']).bid
        order_type = mt5.ORDER_TYPE_BUY
        sl = price - 200
        tp = price + 1000

    elif order == 1:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(trading_data['epsilon']).ask
        sl = price + 200
        tp = price - 1000

    else:
        print("[Xi] All Trades Closed")
        return

    
    take_profit = round(tp, 2)
    stop_loss = round(sl, 2)
    print("Take Profit: ", take_profit)
    print("Stop Loss: ", stop_loss)
    desired_margin_level = 300  # Example value

    xi = calculate_lot_size(equity, contract_size, open_price, leverage)
        
    #round((lot * 2), 2)
    print("Volume: ", xi)
    if xi < 0.01:
        xi = 0.01

    trade = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": epsilon,
        "volume": xi,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": deviation,
        "magic": 234000,
        "comment": "Ko:Zero",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Sending the buy
    result = mt5.order_send(trade)
    print("[Xi] 1. order_send(): by {} {} lots at Price {}. SL:{} TP:{} points".format(
        trading_data['epsilon'], xi, price, stop_loss, take_profit))
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("[Thread - orders] Failed {}: retcode={}".format(order_type, result.retcode))
        return None


def klozero(direction):
    open_positions = mt5.positions_get()

    if not open_positions:
        print('No positions currently open')
        return
    elif len(open_positions) == 0:
        print('No positions currently open')
        return

    # Iterate over all open positions
    for position in open_positions:
        # Check the type of the position (buy or sell)
        if position.type == direction:
            # Request to close the position
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "deviation": 20,
                "magic": 0,
                "comment": "KloZero",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            # Send request to close the position
            result = mt5.order_send(close_request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print("Failed to close position with ticket #:", position.ticket)
            else:
                print("Successfully closed position with ticket #:", position.ticket)
        else:
            print("Error in closing")

def thread_xi(kozeroshutdown, trading_data: dict, lock):
    mt5.initialize()
    global symbol
    lock.acquire()
    symbol = trading_data['epsilon']
    print("Symbol in Xi: ", symbol)
    lock.release()
    print("[Xi] - Working")
    account_info = mt5.account_info()._asdict()
    alpha = trading_data['alpha']
    print("y_pred: ", alpha)
    balance = account_info['balance']
    print("Balance: ", balance)
    modify = balance * 1.2
    closify = balance * 2
    direction = alpha
    print("Direction: ", direction)
    previous_direction = None  # Initialize with None
    last_operation = 0
    modify_operation = 0

    print("[Xi] - Starting Operations")
    while not kozeroshutdown.wait(0.1):
        alpha = trading_data['alpha']
        direction = alpha
        equity = mt5.account_info().equity  # Move this inside the loop to get the updated equity value
        if last_operation > 20 * 10:
            print("Equity: ", equity)
            print("[Xi] - Executing Trade")
            kozero_order(trading_data)
            previous_direction = direction  # Update the previous direction
            last_operation = 0
                
        # Before executing a trade, close any position that doesn't match the current direction
        open_positions = mt5.positions_get(symbol=symbol)
        if open_positions:
            for position in open_positions:
                if direction == -1:
                    pass
                    time.sleep(10)
                elif direction == -2:
                    klozero(1)
                    klozero(0)
                elif position.type != direction:
                    klozero(1 if direction == 0 else 0)


        last_operation += 1

        if modify_operation > 60 * 50:
            print("[Xi] - Checking to Modify")
            kozero_modify(trading_data)
            modify_operation = 0
            
        if equity >= closify:
            klozero(1)
            klozero(0)
        