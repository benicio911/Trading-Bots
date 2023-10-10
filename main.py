import MetaTrader5 as mt5
from kozero import KoZero
import numpy as np

mt5.initialize()

k = KoZero()
#  k.thread_Epsilon()
k.thread_Alpha()
#k.thread_Beta()
#k.thread_Gamma()
#k.thread_Delta()
k.thread_Xi()
k.wait()
