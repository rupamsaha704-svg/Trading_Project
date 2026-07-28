//+------------------------------------------------------------------+
//|                                          ICT_GFT_Strategy.mq5    |
//|                        ICT/SMC Strategy for GFT $5K Challenge    |
//|                        Order Block + FVG + Liquidity Sweep        |
//+------------------------------------------------------------------+
#property copyright "Trading_Project"
#property version   "1.00"
#property strict

//--- Input Parameters
input double   RiskPercent         = 1.0;      // Risk per trade (%)
input double   MaxLotSize          = 0.06;     // Max lot size (GFT rule)
input double   RiskReward          = 2.0;      // Take Profit = RR x SL
input double   TrailingActivation  = 0.5;      // Trail activates at this RR
input double   TrailingFactor      = 0.5;      // Trail distance = factor x SL
input int      ATR_Period          = 14;       // ATR period
input int      EMA_Period          = 200;      // EMA period
input int      ADX_Period          = 14;       // ADX period
input double   ADX_Threshold       = 15.0;     // Min ADX for entry
input int      OB_Lookback         = 3;        // Order Block lookback
input double   DisplacementMult    = 1.5;      // Displacement = mult x ATR
input int      LiquidityLookback   = 20;       // Liquidity sweep lookback
input int      MaxBarsInTrade      = 20;       // Max hold time (bars)
input int      MinHoldBars         = 2;        // Min hold (2 bars = 10min on M5)
input double   MaxDailyLossPct     = 5.0;      // Daily drawdown limit (%)
input double   MaxOverallLossPct   = 10.0;     // Overall drawdown limit (%)
input int      MagicNumber         = 20250727; // EA magic number

//--- Global Variables
double AccountStartBalance;
double DailyStartBalance;
datetime LastDayChecked;
int ATR_Handle, EMA_Handle, ADX_Handle;
double TrailingStopLevel;
bool TrailingActivated;
int EntryBar;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   AccountStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   DailyStartBalance = AccountStartBalance;
   LastDayChecked = 0;
   
   // Create indicator handles
   ATR_Handle = iATR(_Symbol, PERIOD_CURRENT, ATR_Period);
   EMA_Handle = iMA(_Symbol, PERIOD_CURRENT, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   ADX_Handle = iADX(_Symbol, PERIOD_CURRENT, ADX_Period);
   
   if(ATR_Handle == INVALID_HANDLE || EMA_Handle == INVALID_HANDLE || ADX_Handle == INVALID_HANDLE)
   {
      Print("Failed to create indicator handles");
      return(INIT_FAILED);
   }
   
   Print("ICT GFT Strategy initialized. Balance: ", AccountStartBalance);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(ATR_Handle);
   IndicatorRelease(EMA_Handle);
   IndicatorRelease(ADX_Handle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only trade on new bar
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBar == lastBar) return;
   lastBar = currentBar;
   
   // --- Daily Reset ---
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime today = StringToTime(IntegerToString(dt.year) + "." + 
                                  IntegerToString(dt.mon) + "." + 
                                  IntegerToString(dt.day));
   if(today != LastDayChecked)
   {
      DailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      LastDayChecked = today;
   }
   
   // --- Check Drawdown Limits ---
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dailyLoss = DailyStartBalance - currentEquity;
   double dailyLossLimit = AccountStartBalance * (MaxDailyLossPct / 100.0);
   double overallFloor = AccountStartBalance * (1.0 - MaxOverallLossPct / 100.0);
   
   if(dailyLoss >= dailyLossLimit)
   {
      // Close all and stop for today
      CloseAllPositions();
      return;
   }
   
   if(currentEquity <= overallFloor)
   {
      CloseAllPositions();
      Print("HARD BREACH: Overall drawdown exceeded. Stopping.");
      ExpertRemove();
      return;
   }
   
   // --- Manage existing position (trailing stop) ---
   if(PositionsTotal() > 0)
   {
      ManageTrailingStop();
      CheckMaxHoldTime();
      return;  // One position at a time
   }
   
   // --- Get Indicator Values ---
   double atr[], ema[], adx[];
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(ema, true);
   ArraySetAsSeries(adx, true);
   
   if(CopyBuffer(ATR_Handle, 0, 0, 3, atr) < 3) return;
   if(CopyBuffer(EMA_Handle, 0, 0, 3, ema) < 3) return;
   if(CopyBuffer(ADX_Handle, 0, 0, 3, adx) < 3) return;
   
   double currentATR = atr[1];  // Previous completed bar
   double currentEMA = ema[1];
   double currentADX = adx[1];
   
   if(currentATR <= 0 || currentADX < ADX_Threshold) return;
   
   // --- Detect ICT Structures ---
   int signal = DetectICTSignal(currentATR, currentEMA);
   
   if(signal == 0) return;
   
   // --- Calculate Position Size ---
   double stopDistance = currentATR * 2.0;  // SL = 2x ATR
   double lotSize = CalculateLotSize(currentEquity, stopDistance);
   
   if(lotSize <= 0) return;
   
   // --- Check if trade risk is within daily limit ---
   double tradeRisk = stopDistance * lotSize * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE) 
                      / SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double remainingDaily = dailyLossLimit - dailyLoss;
   if(tradeRisk > remainingDaily) return;
   
   // --- Execute Trade ---
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   if(signal == 1)  // BUY
   {
      double sl = ask - stopDistance;
      double tp = ask + (stopDistance * RiskReward);
      
      MqlTradeRequest request = {};
      MqlTradeResult result = {};
      
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = lotSize;
      request.type = ORDER_TYPE_BUY;
      request.price = ask;
      request.sl = sl;
      request.tp = tp;
      request.magic = MagicNumber;
      request.comment = "ICT_BUY";
      request.deviation = 10;
      
      if(OrderSend(request, result))
      {
         TrailingStopLevel = sl;
         TrailingActivated = false;
         EntryBar = Bars(_Symbol, PERIOD_CURRENT);
         Print("BUY: ", lotSize, " lots at ", ask, " SL=", sl, " TP=", tp);
      }
   }
   else if(signal == -1)  // SELL
   {
      double sl = bid + stopDistance;
      double tp = bid - (stopDistance * RiskReward);
      
      MqlTradeRequest request = {};
      MqlTradeResult result = {};
      
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = lotSize;
      request.type = ORDER_TYPE_SELL;
      request.price = bid;
      request.sl = sl;
      request.tp = tp;
      request.magic = MagicNumber;
      request.comment = "ICT_SELL";
      request.deviation = 10;
      
      if(OrderSend(request, result))
      {
         TrailingStopLevel = sl;
         TrailingActivated = false;
         EntryBar = Bars(_Symbol, PERIOD_CURRENT);
         Print("SELL: ", lotSize, " lots at ", bid, " SL=", sl, " TP=", tp);
      }
   }
}

//+------------------------------------------------------------------+
//| Detect ICT Signal (Order Block + FVG + Liquidity Sweep)          |
//+------------------------------------------------------------------+
int DetectICTSignal(double atr, double ema)
{
   // Get recent OHLC data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, LiquidityLookback + OB_Lookback + 5, rates);
   if(copied < LiquidityLookback + 5) return 0;
   
   double close1 = rates[1].close;  // Last completed bar
   double open1 = rates[1].open;
   double high1 = rates[1].high;
   double low1 = rates[1].low;
   
   // --- Check for Bullish Order Block Retest ---
   bool bullishOB = false;
   double obHigh = 0, obLow = 0;
   
   for(int i = 2; i < OB_Lookback + 5; i++)
   {
      // Find displacement candle (strong bullish)
      double body = MathAbs(rates[i].close - rates[i].open);
      if(rates[i].close > rates[i].open && body > atr * DisplacementMult)
      {
         // Find last bearish candle before it
         for(int j = i + 1; j < i + OB_Lookback + 1 && j < copied; j++)
         {
            if(rates[j].close < rates[j].open)  // Bearish = OB
            {
               obHigh = rates[j].high;
               obLow = rates[j].low;
               bullishOB = true;
               break;
            }
         }
         if(bullishOB) break;
      }
   }
   
   // --- Check for Bearish Order Block ---
   bool bearishOB = false;
   double obHighS = 0, obLowS = 0;
   
   for(int i = 2; i < OB_Lookback + 5; i++)
   {
      double body = MathAbs(rates[i].close - rates[i].open);
      if(rates[i].close < rates[i].open && body > atr * DisplacementMult)
      {
         for(int j = i + 1; j < i + OB_Lookback + 1 && j < copied; j++)
         {
            if(rates[j].close > rates[j].open)  // Bullish = bearish OB
            {
               obHighS = rates[j].high;
               obLowS = rates[j].low;
               bearishOB = true;
               break;
            }
         }
         if(bearishOB) break;
      }
   }
   
   // --- Check for Liquidity Sweep ---
   double recentHigh = rates[1].high;
   double recentLow = rates[1].low;
   for(int i = 2; i <= LiquidityLookback; i++)
   {
      if(rates[i].high > recentHigh) recentHigh = rates[i].high;
      if(rates[i].low < recentLow) recentLow = rates[i].low;
   }
   
   bool bullishSweep = (low1 < recentLow && close1 > recentLow);
   bool bearishSweep = (high1 > recentHigh && close1 < recentHigh);
   
   // --- BULLISH SIGNAL ---
   // OB retest: price dipped into OB zone and closed bullish above it
   if(bullishOB && low1 <= obHigh && close1 > obHigh && close1 > open1)
   {
      if(close1 > ema)  // Above EMA
         return 1;
   }
   // Liquidity sweep + bullish close
   if(bullishSweep && close1 > open1 && close1 > ema)
   {
      return 1;
   }
   
   // --- BEARISH SIGNAL ---
   if(bearishOB && high1 >= obLowS && close1 < obLowS && close1 < open1)
   {
      if(close1 < ema)
         return -1;
   }
   if(bearishSweep && close1 < open1 && close1 < ema)
   {
      return -1;
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Calculate lot size with all GFT constraints                      |
//+------------------------------------------------------------------+
double CalculateLotSize(double equity, double stopDistance)
{
   if(stopDistance <= 0) return 0;
   
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   if(tickValue <= 0 || tickSize <= 0) return 0;
   
   // Risk amount = 1% of equity
   double riskAmount = equity * (RiskPercent / 100.0);
   
   // Dollar per point per lot
   double dollarPerPoint = tickValue / tickSize;
   
   // Lot size based on risk
   double lots = riskAmount / (stopDistance * dollarPerPoint);
   
   // Cap at max lot
   if(lots > MaxLotSize) lots = MaxLotSize;
   
   // Minimum lot
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   if(lots < minLot) lots = minLot;
   
   // Round to lot step
   lots = MathFloor(lots / lotStep) * lotStep;
   
   // Margin check (80% max)
   double marginRequired;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lots, SymbolInfoDouble(_Symbol, SYMBOL_ASK), marginRequired))
      return 0;
   
   double maxMargin = equity * (80.0 / 100.0);  // 80% margin limit
   if(marginRequired > maxMargin)
   {
      lots = lots * (maxMargin / marginRequired);
      lots = MathFloor(lots / lotStep) * lotStep;
   }
   
   if(lots < minLot) return 0;
   if(lots > MaxLotSize) lots = MaxLotSize;
   
   return lots;
}

//+------------------------------------------------------------------+
//| Manage Trailing Stop                                              |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      
      double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentTP = PositionGetDouble(POSITION_TP);
      double stopDistance = MathAbs(entryPrice - currentSL);
      
      if(stopDistance <= 0) continue;
      
      long posType = PositionGetInteger(POSITION_TYPE);
      double currentPrice = (posType == POSITION_TYPE_BUY) ? 
                            SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                            SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      
      double unrealizedRR;
      if(posType == POSITION_TYPE_BUY)
         unrealizedRR = (currentPrice - entryPrice) / stopDistance;
      else
         unrealizedRR = (entryPrice - currentPrice) / stopDistance;
      
      // Activate trailing
      if(unrealizedRR >= TrailingActivation && !TrailingActivated)
      {
         TrailingActivated = true;
         TrailingStopLevel = entryPrice;  // Move to break-even
      }
      
      if(TrailingActivated)
      {
         double newTrail;
         if(posType == POSITION_TYPE_BUY)
         {
            newTrail = currentPrice - (stopDistance * TrailingFactor);
            if(newTrail > TrailingStopLevel)
            {
               TrailingStopLevel = newTrail;
               ModifySL(ticket, TrailingStopLevel);
            }
         }
         else
         {
            newTrail = currentPrice + (stopDistance * TrailingFactor);
            if(newTrail < TrailingStopLevel)
            {
               TrailingStopLevel = newTrail;
               ModifySL(ticket, TrailingStopLevel);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check Max Hold Time                                               |
//+------------------------------------------------------------------+
void CheckMaxHoldTime()
{
   int currentBars = Bars(_Symbol, PERIOD_CURRENT);
   int barsHeld = currentBars - EntryBar;
   
   if(barsHeld >= MaxBarsInTrade)
   {
      CloseAllPositions();
      Print("Max hold time reached. Position closed.");
   }
}

//+------------------------------------------------------------------+
//| Modify Stop Loss                                                  |
//+------------------------------------------------------------------+
void ModifySL(ulong ticket, double newSL)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.symbol = _Symbol;
   request.sl = NormalizeDouble(newSL, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
   request.tp = PositionGetDouble(POSITION_TP);
   
   OrderSend(request, result);
}

//+------------------------------------------------------------------+
//| Close All Positions                                               |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      
      MqlTradeRequest request = {};
      MqlTradeResult result = {};
      
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = PositionGetDouble(POSITION_VOLUME);
      request.deviation = 10;
      request.magic = MagicNumber;
      
      long posType = PositionGetInteger(POSITION_TYPE);
      if(posType == POSITION_TYPE_BUY)
      {
         request.type = ORDER_TYPE_SELL;
         request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      }
      else
      {
         request.type = ORDER_TYPE_BUY;
         request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      }
      
      request.position = ticket;
      OrderSend(request, result);
   }
}
//+------------------------------------------------------------------+
