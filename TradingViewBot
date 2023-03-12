mt5.initialize()

position = mt5.positions_get()
if position == None:
    print()

def shinkiro():
    positions = mt5.positions_get()
    for position in positions:
        zero(position)

def zero(position):

    tick = mt5.symbol_info_tick(position.symbol)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
        "price": tick.ask if position.type ==1 else tick.bid,
        "deviation": 20,
        "magic":,
        "comment": "Shinkiro Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        }
    result = mt5.order_send(request)
    print(result)
    if result != mt5.TRADE_RETCODE_CLOSE_ORDER_EXIST:
        print("No order")

def lancelot():
    mt5.initialize()
    symbol = "BTCUSD"
    acct_info = mt5.account_info()
    bal = acct_info.balance
    lot = bal / 8000.00
    volume = '%.2f' % lot
    vol = float(volume)
    rate = mt5.symbol_info(symbol).ask
    deviation = 20
    buy = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": vol,
        "type": mt5.ORDER_TYPE_BUY,
        "price": rate,
        "sl": 0.00,
        "tp": 0.00,
        "deviation": deviation,
        "magic": 24682,
        "comment": "Lancelot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(buy)
    print("[Shinkiro]: Sending the Buy, Price: ", rate, "Lot Size: ", vol)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Error opening buy order:", result)


def guren():
    mt5.initialize()
    symbol = "BTCUSD"
    acct_info = mt5.account_info()
    bal = acct_info.balance
    lot = bal / 800.00
    volume = '%.2f' % lot
    vol = float(volume)
    rate = mt5.symbol_info(symbol).bid
    deviation = 20
    sell = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": vol,
        "type": mt5.ORDER_TYPE_SELL,
        "price": rate,
        "sl": 0.00,
        "tp": 0.00,
        "deviation": deviation,
        "magic": ,
        "comment": "Guren",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(sell)
    print("[Shinkiro]: Snding the Sell, Price: ", rate, "Lot Size: ", vol)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Error opening buy order:", result)

    client.event
    async def on_message(message):
        if message.content == 'Shinkiro, Closing all trades':
            shinkiro()
            print("All Bots Have Disengaged")
            await message.channel.send("Shinkiro has confirmed the buy")
        elif mesage.content == 'Lancelot, taking the buy':
            print("Guren disengaged, Lancelot Engaging")
            lancelot()
            print("Lancelot Engaged")
            await message.channel.send("Shinkiro has confirmed the buy")
        elif message.content == 'Guren, taking the sell':
            print("Lancelot disengaged, ren Engaging")
            guren()
            print("Guren Engaged")
            await message.channel.send("Shinkiro has confirmed th sell")
