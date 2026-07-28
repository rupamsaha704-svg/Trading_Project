//+------------------------------------------------------------------+
//|                                          GoldMulti_M15.mq5       |
//|     Multi-Indicator Gold Scalper - M15 Timeframe                 |
//|     EMA Cross + RSI Reversal + Bollinger Breakout + ATR          |
//|     Target: $8000 in 6 months on $5K, GFT rules enforced        |
//+------------------------------------------------------------------+
#property copyright "Trading_Project"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//=== STRATEGY ===
input int      EMA_Fast        = 13;       // Fast EMA
input int      EMA_Slow        = 34;       // Slow EMA
input int      RSI_Period      = 14;       // RSI period
input double   RSI_BuyLevel    = 35.0;     // RSI buy zone (below this)
input double   RSI_SellLevel   = 65.0;     // RSI sell zone (above this)
input int      BB_Period       = 20;       // Bollinger period
input double   BB_Dev          = 2.0;      // Bollinger deviation
input int      ATR_Period      = 14;       // ATR period
input double   SL_ATR          = 1.2;      // SL = ATR x this
input double   TP_ATR          = 3.6;      // TP = ATR x this (RR 3:1)

//=== RISK ===
input double   RiskPct         = 0.8;      // Risk % per trade
input double   MaxLot          = 0.06;     // Max lot (GFT)
input double   DailyLossPct    = 4.5;      // Daily loss stop %
input double   MaxDDPct        = 9.0;      // Overall max DD %

//=== SESSION ===
input int      SessionStart    = 6;        // Trade from (hour)
input int      SessionEnd      = 20;       // Trade until (hour)
input int      MaxSpread       = 40;       // Max spread (points)

//=== TRADE MGMT ===
input bool     UseTrailing     = true;     // Trailing stop
input double   Trail_RR        = 1.0;      // Trail starts at 1R
input double   Trail_ATR       = 0.6;      // Trail distance = ATR x this
input int      MaxPerDay       = 5;        // Max trades per day
input int      CooldownBars    = 3;        // Min bars between trades

//=== SYSTEM ===
input int      Magic           = 554433;   // Magic number

//=== GLOBALS ===
CTrade   trade;
int      hEmaF, hEmaS, hRSI, hBB, hATR;
double   gStartBal, gDayBal;
datetime gLastDay;
int      gDayTrades, gLastBar;

//+------------------------------------------------------------------+
int OnInit()
{
   hEmaF = iMA(_Symbol, PERIOD_M15, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaS = iMA(_Symbol, PERIOD_M15, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   hRSI  = iRSI(_Symbol, PERIOD_M15, RSI_Period, PRICE_CLOSE);
   hBB   = iBands(_Symbol, PERIOD_M15, BB_Period, 0, BB_Dev, PRICE_CLOSE);
   hATR  = iATR(_Symbol, PERIOD_M15, ATR_Period);
   
   if(hEmaF==INVALID_HANDLE || hEmaS==INVALID_HANDLE || 
      hRSI==INVALID_HANDLE || hBB==INVALID_HANDLE || hATR==INVALID_HANDLE)
   {
      Print("ERROR: Indicators failed");
      return(INIT_FAILED);
   }
   
   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(15);
   gStartBal = AccountInfoDouble(ACCOUNT_BALANCE);
   gDayBal = gStartBal;
   gLastDay = 0;
   gDayTrades = 0;
   gLastBar = -100;
   
   Print("GoldMulti M15 started. Bal=", gStartBal);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(hEmaF!=INVALID_HANDLE) IndicatorRelease(hEmaF);
   if(hEmaS!=INVALID_HANDLE) IndicatorRelease(hEmaS);
   if(hRSI!=INVALID_HANDLE)  IndicatorRelease(hRSI);
   if(hBB!=INVALID_HANDLE)   IndicatorRelease(hBB);
   if(hATR!=INVALID_HANDLE)  IndicatorRelease(hATR);
}

//+------------------------------------------------------------------+
void OnTick()
{
   static datetime sPrev = 0;
   datetime cur = iTime(_Symbol, PERIOD_M15, 0);
   if(cur == sPrev) return;
   sPrev = cur;
   
   DayReset();
   
   // DD checks
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if((gStartBal - eq)/gStartBal*100.0 >= MaxDDPct){ CloseAll(); return; }
   if((gDayBal - eq)/gDayBal*100.0 >= DailyLossPct) return;
   
   // Trail existing
   if(UseTrailing) DoTrail();
   
   // Skip if position open
   if(HasPos()) return;
   
   // Filters
   if(!InSession()) return;
   if(gDayTrades >= MaxPerDay) return;
   int barNum = Bars(_Symbol, PERIOD_M15);
   if(barNum - gLastBar < CooldownBars) return;
   if(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > MaxSpread) return;
   
   // Get data (bar[1] = last closed)
   double ef[3], es[3], rsi[2], bbUp[2], bbLow[2], atr[2];
   ArraySetAsSeries(ef,true); ArraySetAsSeries(es,true);
   ArraySetAsSeries(rsi,true); ArraySetAsSeries(bbUp,true);
   ArraySetAsSeries(bbLow,true); ArraySetAsSeries(atr,true);
   
   if(CopyBuffer(hEmaF,0,1,3,ef)<3) return;
   if(CopyBuffer(hEmaS,0,1,3,es)<3) return;
   if(CopyBuffer(hRSI,0,1,2,rsi)<2) return;
   if(CopyBuffer(hBB,1,1,2,bbUp)<2) return;   // upper band
   if(CopyBuffer(hBB,2,1,2,bbLow)<2) return;  // lower band
   if(CopyBuffer(hATR,0,1,2,atr)<2) return;
   
   double close1 = iClose(_Symbol, PERIOD_M15, 1);
   double open1  = iOpen(_Symbol, PERIOD_M15, 1);
   double curATR = atr[0];
   if(curATR <= 0) return;
   
   int signal = 0;
   
   // === SIGNAL 1: EMA Crossover + RSI confirmation ===
   bool emaCrossUp   = (ef[0] > es[0]) && (ef[1] <= es[1]);
   bool emaCrossDown = (ef[0] < es[0]) && (ef[1] >= es[1]);
   
   if(emaCrossUp && rsi[0] < RSI_SellLevel && close1 > open1)
      signal = 1;
   if(emaCrossDown && rsi[0] > RSI_BuyLevel && close1 < open1)
      signal = -1;
   
   // === SIGNAL 2: RSI extreme + EMA trend ===
   if(signal == 0)
   {
      bool upTrend = (ef[0] > es[0]);
      bool dnTrend = (ef[0] < es[0]);
      
      if(upTrend && rsi[0] < RSI_BuyLevel && close1 > open1)
         signal = 1;
      if(dnTrend && rsi[0] > RSI_SellLevel && close1 < open1)
         signal = -1;
   }
   
   // === SIGNAL 3: Bollinger breakout + trend ===
   if(signal == 0)
   {
      bool upTrend = (ef[0] > es[0]);
      bool dnTrend = (ef[0] < es[0]);
      
      if(upTrend && close1 > bbUp[0] && close1 > open1)
         signal = 1;
      if(dnTrend && close1 < bbLow[0] && close1 < open1)
         signal = -1;
   }
   
   if(signal == 0) return;
   
   // Execute
   OpenTrade(signal, curATR);
}

//+------------------------------------------------------------------+
void OpenTrade(int sig, double atr)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   int dig = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   double slD = atr * SL_ATR;
   double tpD = atr * TP_ATR;
   double entry, sl, tp;
   ENUM_ORDER_TYPE type;
   
   if(sig == 1)
   {
      type = ORDER_TYPE_BUY;
      entry = ask;
      sl = NormalizeDouble(entry - slD, dig);
      tp = NormalizeDouble(entry + tpD, dig);
   }
   else
   {
      type = ORDER_TYPE_SELL;
      entry = bid;
      sl = NormalizeDouble(entry + slD, dig);
      tp = NormalizeDouble(entry - tpD, dig);
   }
   
   double lots = CalcLot(slD);
   if(lots <= 0) return;
   
   if(trade.PositionOpen(_Symbol, type, lots, entry, sl, tp, "GM15"))
   {
      gDayTrades++;
      gLastBar = Bars(_Symbol, PERIOD_M15);
      Print(sig==1?"BUY ":"SELL ", lots, " @ ", entry, " SL=", sl, " TP=", tp);
   }
}

//+------------------------------------------------------------------+
double CalcLot(double slD)
{
   if(slD <= 0) return 0;
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk = bal * RiskPct / 100.0;
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tv<=0 || ts<=0) return 0;
   double cpl = (slD/ts)*tv;
   if(cpl<=0) return 0;
   double l = risk/cpl;
   double mn = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double st = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   l = MathFloor(l/st)*st;
   if(l<mn) l=mn;
   if(l>MaxLot) l=MaxLot;
   return l;
}

//+------------------------------------------------------------------+
void DoTrail()
{
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk==0) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=Magic) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      
      double ent = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl = PositionGetDouble(POSITION_SL);
      double slD = MathAbs(ent-sl);
      if(slD<=0) continue;
      
      int pt = (int)PositionGetInteger(POSITION_TYPE);
      double pr = (pt==POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol,SYMBOL_BID) : SymbolInfoDouble(_Symbol,SYMBOL_ASK);
      double pft = (pt==POSITION_TYPE_BUY) ? (pr-ent) : (ent-pr);
      double rr = pft/slD;
      
      if(rr < Trail_RR) continue;
      
      double atrB[1];
      ArraySetAsSeries(atrB,true);
      if(CopyBuffer(hATR,0,0,1,atrB)<1) continue;
      double tDist = atrB[0]*Trail_ATR;
      int dig = (int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
      
      if(pt==POSITION_TYPE_BUY)
      {
         double ns = NormalizeDouble(pr-tDist, dig);
         if(ns>sl) trade.PositionModify(tk, ns, PositionGetDouble(POSITION_TP));
      }
      else
      {
         double ns = NormalizeDouble(pr+tDist, dig);
         if(ns<sl || sl==0) trade.PositionModify(tk, ns, PositionGetDouble(POSITION_TP));
      }
   }
}

//+------------------------------------------------------------------+
bool HasPos()
{
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong tk=PositionGetTicket(i);
      if(tk==0) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=Magic) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
bool InSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(),dt);
   return (dt.hour>=SessionStart && dt.hour<SessionEnd);
}

//+------------------------------------------------------------------+
void DayReset()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(),dt);
   datetime td = StringToTime(StringFormat("%d.%02d.%02d",dt.year,dt.mon,dt.day));
   if(td!=gLastDay){ gDayBal=AccountInfoDouble(ACCOUNT_BALANCE); gDayTrades=0; gLastDay=td; }
}

//+------------------------------------------------------------------+
void CloseAll()
{
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong tk=PositionGetTicket(i);
      if(tk==0) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=Magic) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      trade.PositionClose(tk);
   }
}
//+------------------------------------------------------------------+
