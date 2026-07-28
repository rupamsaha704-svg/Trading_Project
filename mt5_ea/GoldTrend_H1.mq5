//+------------------------------------------------------------------+
//|                                            GoldTrend_H1.mq5      |
//|     Simple Gold Trend EA - H1 Timeframe - Proven Approach        |
//|     EMA Cross + ATR Volatility + Session + Trailing              |
//|     Target: $7-8K profit in 6 months on $5K account              |
//+------------------------------------------------------------------+
#property copyright "Trading_Project"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//=== INPUTS ===
input group "=== Strategy ==="
input int      EMA_Fast        = 21;       // Fast EMA period
input int      EMA_Slow        = 55;       // Slow EMA period
input int      ATR_Period      = 14;       // ATR period
input double   ATR_MinFilter   = 3.0;      // Min ATR in price points to trade
input double   SL_ATR_Mult     = 1.5;      // SL = ATR x this
input double   TP_ATR_Mult     = 4.5;      // TP = ATR x this (RR = 3:1)

input group "=== Risk ==="
input double   RiskPct         = 1.0;      // Risk % per trade
input double   MaxLot          = 0.06;     // Max lot size
input double   DailyMaxLossPct = 4.5;      // Stop trading after this % loss/day
input double   MaxDDPct        = 9.0;      // Hard stop if DD reaches this %

input group "=== Session (Broker GMT) ==="
input int      SessionStart    = 7;        // Start hour (London open)
input int      SessionEnd      = 20;       // End hour (NY close)

input group "=== Trade Management ==="
input bool     UseTrailing     = true;     // Enable trailing stop
input double   Trail_Start_RR  = 1.0;      // Start trail at 1R profit
input double   Trail_Step_ATR  = 0.75;     // Trail distance = ATR x this
input int      MaxTradesPerDay = 3;        // Max trades per day
input int      MinBarsBetween  = 4;        // Min H1 bars between trades

input group "=== System ==="
input int      Magic           = 112233;   // Magic number

//=== GLOBALS ===
CTrade   trade;
int      hEmaFast, hEmaSlow, hATR;
double   gStartBalance;
double   gDayBalance;
datetime gLastDay;
int      gDayTrades;
int      gLastTradeBar;

//+------------------------------------------------------------------+
int OnInit()
{
   hEmaFast = iMA(_Symbol, PERIOD_H1, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlow = iMA(_Symbol, PERIOD_H1, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   hATR     = iATR(_Symbol, PERIOD_H1, ATR_Period);
   
   if(hEmaFast == INVALID_HANDLE || hEmaSlow == INVALID_HANDLE || hATR == INVALID_HANDLE)
   {
      Print("ERROR creating indicators");
      return(INIT_FAILED);
   }
   
   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(20);
   
   gStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   gDayBalance = gStartBalance;
   gLastDay = 0;
   gDayTrades = 0;
   gLastTradeBar = -100;
   
   Print("GoldTrend H1 EA started. Balance=", gStartBalance);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(hEmaFast != INVALID_HANDLE) IndicatorRelease(hEmaFast);
   if(hEmaSlow != INVALID_HANDLE) IndicatorRelease(hEmaSlow);
   if(hATR != INVALID_HANDLE)     IndicatorRelease(hATR);
}

//+------------------------------------------------------------------+
void OnTick()
{
   // Only act on new H1 bar
   static datetime sPrevBar = 0;
   datetime curBar = iTime(_Symbol, PERIOD_H1, 0);
   if(curBar == sPrevBar) return;
   sPrevBar = curBar;
   
   // Daily reset
   CheckDayReset();
   
   // DD protection
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double ddPct = (gStartBalance - equity) / gStartBalance * 100.0;
   if(ddPct >= MaxDDPct)
   {
      CloseAll();
      Print("MAX DD REACHED - ALL CLOSED");
      return;
   }
   
   // Daily loss check
   double dayLoss = (gDayBalance - equity) / gDayBalance * 100.0;
   if(dayLoss >= DailyMaxLossPct)
   {
      Print("Daily loss limit reached: ", dayLoss, "%");
      return;
   }
   
   // Manage trailing on existing positions
   if(UseTrailing) ManageTrailing();
   
   // Check if we have a position already
   if(HasPosition()) return;
   
   // Session check
   if(!InSession()) return;
   
   // Max trades per day
   if(gDayTrades >= MaxTradesPerDay) return;
   
   // Min bars between trades
   int curBarNum = Bars(_Symbol, PERIOD_H1);
   if(curBarNum - gLastTradeBar < MinBarsBetween) return;
   
   // Get indicators for COMPLETED bars (bar[1] and bar[2])
   double emaF[3], emaS[3], atr[2];
   ArraySetAsSeries(emaF, true);
   ArraySetAsSeries(emaS, true);
   ArraySetAsSeries(atr, true);
   
   if(CopyBuffer(hEmaFast, 0, 1, 3, emaF) < 3) return;
   if(CopyBuffer(hEmaSlow, 0, 1, 3, emaS) < 3) return;
   if(CopyBuffer(hATR, 0, 1, 2, atr) < 2) return;
   
   double curATR = atr[0];
   
   // ATR filter - need minimum volatility
   if(curATR < ATR_MinFilter) return;
   
   // EMA crossover detection on COMPLETED bars
   // bar[1] = last closed, bar[2] = one before
   bool crossUp   = (emaF[0] > emaS[0]) && (emaF[1] <= emaS[1]);  // Fast crossed above Slow
   bool crossDown = (emaF[0] < emaS[0]) && (emaF[1] >= emaS[1]);  // Fast crossed below Slow
   
   // Additional confirmation: price closed in direction of cross
   double close1 = iClose(_Symbol, PERIOD_H1, 1);
   double open1  = iOpen(_Symbol, PERIOD_H1, 1);
   
   int signal = 0;
   if(crossUp && close1 > open1)   signal = 1;   // BUY
   if(crossDown && close1 < open1) signal = -1;  // SELL
   
   if(signal == 0) return;
   
   // Execute trade
   OpenTrade(signal, curATR);
}

//+------------------------------------------------------------------+
void OpenTrade(int signal, double atr)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   int dig = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   double entry, sl, tp;
   ENUM_ORDER_TYPE type;
   
   double slDist = atr * SL_ATR_Mult;
   double tpDist = atr * TP_ATR_Mult;
   
   if(signal == 1)
   {
      type = ORDER_TYPE_BUY;
      entry = ask;
      sl = NormalizeDouble(entry - slDist, dig);
      tp = NormalizeDouble(entry + tpDist, dig);
   }
   else
   {
      type = ORDER_TYPE_SELL;
      entry = bid;
      sl = NormalizeDouble(entry + slDist, dig);
      tp = NormalizeDouble(entry - tpDist, dig);
   }
   
   // Position sizing
   double lots = CalcLots(slDist);
   if(lots <= 0) return;
   
   // Execute
   if(trade.PositionOpen(_Symbol, type, lots, entry, sl, tp, "GoldTrend"))
   {
      gDayTrades++;
      gLastTradeBar = Bars(_Symbol, PERIOD_H1);
      Print(signal==1 ? "BUY " : "SELL ", lots, " @ ", entry, " SL=", sl, " TP=", tp, " ATR=", atr);
   }
}

//+------------------------------------------------------------------+
double CalcLots(double slDist)
{
   if(slDist <= 0) return 0;
   
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmt = balance * RiskPct / 100.0;
   
   double tickVal = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSz  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   if(tickVal <= 0 || tickSz <= 0) return 0;
   
   double costPerLot = (slDist / tickSz) * tickVal;
   if(costPerLot <= 0) return 0;
   
   double lots = riskAmt / costPerLot;
   
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lots = MathFloor(lots / stepLot) * stepLot;
   if(lots < minLot) lots = minLot;
   if(lots > MaxLot) lots = MaxLot;
   
   return lots;
}

//+------------------------------------------------------------------+
void ManageTrailing()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != Magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double curSL = PositionGetDouble(POSITION_SL);
      double slDist = MathAbs(entry - curSL);
      if(slDist <= 0) continue;
      
      int posType = (int)PositionGetInteger(POSITION_TYPE);
      double curPrice = (posType == POSITION_TYPE_BUY) ?
                        SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                        SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      
      // Current RR
      double profit = (posType == POSITION_TYPE_BUY) ? (curPrice - entry) : (entry - curPrice);
      double rr = profit / slDist;
      
      // Only trail after reaching Trail_Start_RR
      if(rr < Trail_Start_RR) continue;
      
      // Get current ATR for trail distance
      double atrBuf[1];
      ArraySetAsSeries(atrBuf, true);
      if(CopyBuffer(hATR, 0, 0, 1, atrBuf) < 1) continue;
      double trailDist = atrBuf[0] * Trail_Step_ATR;
      
      int dig = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      double newSL;
      
      if(posType == POSITION_TYPE_BUY)
      {
         newSL = curPrice - trailDist;
         if(newSL > curSL + SymbolInfoDouble(_Symbol, SYMBOL_POINT))
         {
            trade.PositionModify(ticket, NormalizeDouble(newSL, dig), PositionGetDouble(POSITION_TP));
         }
      }
      else
      {
         newSL = curPrice + trailDist;
         if(newSL < curSL - SymbolInfoDouble(_Symbol, SYMBOL_POINT) || curSL == 0)
         {
            trade.PositionModify(ticket, NormalizeDouble(newSL, dig), PositionGetDouble(POSITION_TP));
         }
      }
   }
}

//+------------------------------------------------------------------+
bool HasPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != Magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
bool InSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   return (h >= SessionStart && h < SessionEnd);
}

//+------------------------------------------------------------------+
void CheckDayReset()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime today = StringToTime(StringFormat("%d.%02d.%02d", dt.year, dt.mon, dt.day));
   if(today != gLastDay)
   {
      gDayBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      gDayTrades = 0;
      gLastDay = today;
   }
}

//+------------------------------------------------------------------+
void CloseAll()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != Magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      trade.PositionClose(ticket);
   }
}
//+------------------------------------------------------------------+
