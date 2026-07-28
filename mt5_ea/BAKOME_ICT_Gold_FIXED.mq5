//+------------------------------------------------------------------+
//|                          BAKOME_ICT_Gold_FIXED.mq5               |
//|           Based on BAKOME Ultimate ICT Gold Scalper v3.0         |
//|           Fixed: removed templates, fixed MQL5 compat            |
//+------------------------------------------------------------------+
#property copyright "BAKOME - Fixed Build"
#property version   "3.01"
#property description "ICT Gold Scalper - FVG, Order Blocks, Silver Bullet"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\AccountInfo.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Risk Management ==="
input double RiskPercent            = 1.0;
input double MaxDailyRiskPercent    = 5.0;
input double MaxDailyProfitPercent  = 8.0;
input int    MaxPositions           = 2;
input int    MaxDailyTrades         = 10;

input group "=== XAUUSD Settings ==="
input double MinATR_Points          = 100.0;
input double MaxSpreadPoints        = 50.0;
input double ATR_SL_Multiplier      = 2.0;
input double ATR_TP_Multiplier      = 3.0;

input group "=== ICT Strategy ==="
input bool   UseLiquiditySweeps     = true;
input bool   UseFairValueGaps       = true;
input bool   UseOrderBlocks         = true;
input bool   UseSilverBullet        = true;
input int    LiquidityLookback      = 50;
input int    FVG_Lookback           = 20;
input double FVG_MinSizeATR         = 0.5;

input group "=== Session Settings ==="
input bool   TradeLondonSession     = true;
input bool   TradeNewYorkSession    = true;
input int    LondonStartHour        = 7;
input int    NewYorkStartHour       = 13;

input group "=== Silver Bullet ==="
input bool   LondonSilverBullet     = true;
input int    LondonKZStart          = 8;
input int    LondonKZEnd            = 9;
input bool   NYSilverBullet         = true;
input int    NYKZStart              = 15;
input int    NYKZEnd                = 16;

input group "=== Position Management ==="
input bool   UseBreakEven           = true;
input double BE_TriggerATR          = 1.0;
input bool   UseTrailingStop        = true;
input double Trail_StartATR         = 1.5;
input double Trail_StepATR          = 0.5;

input group "=== Execution ==="
input int    SlippagePoints         = 10;
input int    EA_MagicBase           = 888777;

//+------------------------------------------------------------------+
//| Structures                                                        |
//+------------------------------------------------------------------+
struct SLiquidityLevel
{
   double   price;
   datetime time;
   int      strength;
   bool     isHigh;
   bool     swept;
};

struct SFairValueGap
{
   double   topPrice;
   double   bottomPrice;
   datetime time;
   bool     isBullish;
   bool     filled;
};

struct SOrderBlock
{
   double   topPrice;
   double   bottomPrice;
   datetime time;
   bool     isBullish;
   bool     mitigated;
};

struct STrackedPos
{
   ulong    ticket;
   double   openPrice;
   double   originalSL;
   bool     breakEvenSet;
   bool     trailingActive;
};

//+------------------------------------------------------------------+
//| Global Variables                                                   |
//+------------------------------------------------------------------+
CTrade         g_trade;
CPositionInfo  g_pos;
CSymbolInfo    g_sym;
CAccountInfo   g_acc;

int            g_atrHandle;
int            g_emaH1Handle;
int            g_emaH4Handle;

SLiquidityLevel g_liqLevels[];
SFairValueGap   g_fvgList[];
SOrderBlock     g_obList[];
STrackedPos     g_tracked[];

double         g_dayStartBal;
int            g_todayTrades;
datetime       g_lastDay;
double         g_currentATR;
long           g_magic;
bool           g_ready;

//+------------------------------------------------------------------+
int OnInit()
{
   g_sym.Name(_Symbol);
   g_sym.Refresh();
   
   g_atrHandle   = iATR(_Symbol, PERIOD_M5, 14);
   g_emaH1Handle = iMA(_Symbol, PERIOD_H1, 34, 0, MODE_EMA, PRICE_CLOSE);
   g_emaH4Handle = iMA(_Symbol, PERIOD_H4, 200, 0, MODE_EMA, PRICE_CLOSE);
   
   if(g_atrHandle == INVALID_HANDLE || g_emaH1Handle == INVALID_HANDLE || g_emaH4Handle == INVALID_HANDLE)
   {
      Print("ERROR: Indicator creation failed");
      return(INIT_FAILED);
   }
   
   g_magic = EA_MagicBase;
   g_trade.SetExpertMagicNumber(g_magic);
   g_trade.SetDeviationInPoints(SlippagePoints);
   g_dayStartBal = g_acc.Balance();
   g_todayTrades = 0;
   g_lastDay = 0;
   g_currentATR = 0;
   g_ready = true;
   
   Print("BAKOME ICT Gold EA initialized. Magic=", g_magic);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_atrHandle != INVALID_HANDLE)   IndicatorRelease(g_atrHandle);
   if(g_emaH1Handle != INVALID_HANDLE) IndicatorRelease(g_emaH1Handle);
   if(g_emaH4Handle != INVALID_HANDLE) IndicatorRelease(g_emaH4Handle);
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(!g_ready) return;
   
   // New bar check
   static datetime sPrevBar = 0;
   datetime curBar = iTime(_Symbol, PERIOD_M5, 0);
   if(curBar == sPrevBar) return;
   sPrevBar = curBar;
   
   // Daily reset
   DailyReset();
   
   // Update ATR
   double atrBuf[];
   ArraySetAsSeries(atrBuf, true);
   if(CopyBuffer(g_atrHandle, 0, 0, 1, atrBuf) > 0)
      g_currentATR = atrBuf[0];
   if(g_currentATR <= 0) return;
   
   // Manage existing positions
   ManagePositions();
   
   // Check if we can trade
   if(!CanTrade()) return;
   
   // Update market structures
   UpdateLiquidity();
   UpdateFVG();
   UpdateOB();
   
   // Generate signal
   int signal = GetSignal();
   if(signal == 0) return;
   
   // Execute
   ExecuteTrade(signal);
}

//+------------------------------------------------------------------+
void DailyReset()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime today = StringToTime(StringFormat("%d.%02d.%02d", dt.year, dt.mon, dt.day));
   if(today != g_lastDay)
   {
      g_dayStartBal = g_acc.Balance();
      g_todayTrades = 0;
      g_lastDay = today;
   }
}

//+------------------------------------------------------------------+
bool CanTrade()
{
   // Daily trade limit
   if(g_todayTrades >= MaxDailyTrades) return false;
   
   // Daily P&L limits
   double equity = g_acc.Equity();
   double dailyPL = (equity - g_dayStartBal) / g_dayStartBal * 100.0;
   if(dailyPL <= -MaxDailyRiskPercent) return false;
   if(dailyPL >= MaxDailyProfitPercent) return false;
   
   // Position count
   int cnt = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(g_pos.SelectByIndex(i) && g_pos.Magic() == g_magic)
         cnt++;
   }
   if(cnt >= MaxPositions) return false;
   
   // Spread filter
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(spread > (long)MaxSpreadPoints) return false;
   
   // ATR minimum
   if(g_currentATR / g_sym.Point() < MinATR_Points) return false;
   
   // Session filter
   if(!InSession()) return false;
   
   return true;
}

//+------------------------------------------------------------------+
bool InSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   
   if(TradeLondonSession && h >= LondonStartHour && h < LondonStartHour + 4) return true;
   if(TradeNewYorkSession && h >= NewYorkStartHour && h < NewYorkStartHour + 4) return true;
   return false;
}

//+------------------------------------------------------------------+
bool InKillZone()
{
   if(!UseSilverBullet) return false;
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   if(LondonSilverBullet && h >= LondonKZStart && h < LondonKZEnd) return true;
   if(NYSilverBullet && h >= NYKZStart && h < NYKZEnd) return true;
   return false;
}

//+------------------------------------------------------------------+
int GetMarketBias()
{
   double ema[];
   ArraySetAsSeries(ema, true);
   if(CopyBuffer(g_emaH4Handle, 0, 0, 1, ema) <= 0) return 0;
   double price = iClose(_Symbol, PERIOD_M5, 1);
   if(price > ema[0]) return 1;
   if(price < ema[0]) return -1;
   return 0;
}

//+------------------------------------------------------------------+
void UpdateLiquidity()
{
   if(!UseLiquiditySweeps) return;
   ArrayResize(g_liqLevels, 0);
   
   for(int i = 3; i < LiquidityLookback && i < Bars(_Symbol, PERIOD_M5) - 3; i++)
   {
      double hi = iHigh(_Symbol, PERIOD_M5, i);
      if(hi > iHigh(_Symbol, PERIOD_M5, i-1) &&
         hi > iHigh(_Symbol, PERIOD_M5, i-2) &&
         hi > iHigh(_Symbol, PERIOD_M5, i+1) &&
         hi > iHigh(_Symbol, PERIOD_M5, i+2))
      {
         int sz = ArraySize(g_liqLevels);
         ArrayResize(g_liqLevels, sz+1);
         g_liqLevels[sz].price = hi;
         g_liqLevels[sz].isHigh = true;
         g_liqLevels[sz].swept = false;
         g_liqLevels[sz].time = iTime(_Symbol, PERIOD_M5, i);
         g_liqLevels[sz].strength = 1;
      }
      
      double lo = iLow(_Symbol, PERIOD_M5, i);
      if(lo < iLow(_Symbol, PERIOD_M5, i-1) &&
         lo < iLow(_Symbol, PERIOD_M5, i-2) &&
         lo < iLow(_Symbol, PERIOD_M5, i+1) &&
         lo < iLow(_Symbol, PERIOD_M5, i+2))
      {
         int sz = ArraySize(g_liqLevels);
         ArrayResize(g_liqLevels, sz+1);
         g_liqLevels[sz].price = lo;
         g_liqLevels[sz].isHigh = false;
         g_liqLevels[sz].swept = false;
         g_liqLevels[sz].time = iTime(_Symbol, PERIOD_M5, i);
         g_liqLevels[sz].strength = 1;
      }
   }
}

//+------------------------------------------------------------------+
void UpdateFVG()
{
   if(!UseFairValueGaps) return;
   ArrayResize(g_fvgList, 0);
   
   for(int i = 2; i < FVG_Lookback && i < Bars(_Symbol, PERIOD_M5) - 1; i++)
   {
      double prevHigh = iHigh(_Symbol, PERIOD_M5, i+1);
      double curLow   = iLow(_Symbol, PERIOD_M5, i-1);
      
      // Bullish FVG
      if(curLow > prevHigh)
      {
         double gap = curLow - prevHigh;
         if(gap >= g_currentATR * FVG_MinSizeATR)
         {
            int sz = ArraySize(g_fvgList);
            ArrayResize(g_fvgList, sz+1);
            g_fvgList[sz].topPrice = curLow;
            g_fvgList[sz].bottomPrice = prevHigh;
            g_fvgList[sz].isBullish = true;
            g_fvgList[sz].filled = false;
            g_fvgList[sz].time = iTime(_Symbol, PERIOD_M5, i);
         }
      }
      
      double prevLow  = iLow(_Symbol, PERIOD_M5, i+1);
      double curHigh  = iHigh(_Symbol, PERIOD_M5, i-1);
      
      // Bearish FVG
      if(curHigh < prevLow)
      {
         double gap = prevLow - curHigh;
         if(gap >= g_currentATR * FVG_MinSizeATR)
         {
            int sz = ArraySize(g_fvgList);
            ArrayResize(g_fvgList, sz+1);
            g_fvgList[sz].topPrice = prevLow;
            g_fvgList[sz].bottomPrice = curHigh;
            g_fvgList[sz].isBullish = false;
            g_fvgList[sz].filled = false;
            g_fvgList[sz].time = iTime(_Symbol, PERIOD_M5, i);
         }
      }
   }
}

//+------------------------------------------------------------------+
void UpdateOB()
{
   if(!UseOrderBlocks) return;
   ArrayResize(g_obList, 0);
   
   for(int i = 1; i < 50 && i < Bars(_Symbol, PERIOD_M5) - 1; i++)
   {
      double op_i = iOpen(_Symbol, PERIOD_M5, i);
      double cl_i = iClose(_Symbol, PERIOD_M5, i);
      double op_prev = iOpen(_Symbol, PERIOD_M5, i-1);
      double cl_prev = iClose(_Symbol, PERIOD_M5, i-1);
      
      // Bearish candle followed by bullish = bullish OB
      if(cl_i < op_i && cl_prev > op_prev)
      {
         int sz = ArraySize(g_obList);
         ArrayResize(g_obList, sz+1);
         g_obList[sz].topPrice = iHigh(_Symbol, PERIOD_M5, i);
         g_obList[sz].bottomPrice = iLow(_Symbol, PERIOD_M5, i);
         g_obList[sz].isBullish = true;
         g_obList[sz].mitigated = false;
         g_obList[sz].time = iTime(_Symbol, PERIOD_M5, i);
      }
      
      // Bullish candle followed by bearish = bearish OB
      if(cl_i > op_i && cl_prev < op_prev)
      {
         int sz = ArraySize(g_obList);
         ArrayResize(g_obList, sz+1);
         g_obList[sz].topPrice = iHigh(_Symbol, PERIOD_M5, i);
         g_obList[sz].bottomPrice = iLow(_Symbol, PERIOD_M5, i);
         g_obList[sz].isBullish = false;
         g_obList[sz].mitigated = false;
         g_obList[sz].time = iTime(_Symbol, PERIOD_M5, i);
      }
   }
}

//+------------------------------------------------------------------+
int GetSignal()
{
   int bias = GetMarketBias();
   if(bias == 0) return 0;
   
   double price = iClose(_Symbol, PERIOD_M5, 1);
   double low1  = iLow(_Symbol, PERIOD_M5, 1);
   double high1 = iHigh(_Symbol, PERIOD_M5, 1);
   
   bool inKZ = InKillZone();
   
   // Check Order Block retests
   for(int i = 0; i < ArraySize(g_obList); i++)
   {
      if(g_obList[i].mitigated) continue;
      
      // Bullish OB retest
      if(g_obList[i].isBullish && bias == 1)
      {
         if(low1 <= g_obList[i].topPrice && price > g_obList[i].topPrice)
         {
            if(!UseSilverBullet || inKZ)
            {
               g_obList[i].mitigated = true;
               return 1;
            }
         }
      }
      
      // Bearish OB retest
      if(!g_obList[i].isBullish && bias == -1)
      {
         if(high1 >= g_obList[i].bottomPrice && price < g_obList[i].bottomPrice)
         {
            if(!UseSilverBullet || inKZ)
            {
               g_obList[i].mitigated = true;
               return -1;
            }
         }
      }
   }
   
   // Check FVG fills
   for(int i = 0; i < ArraySize(g_fvgList); i++)
   {
      if(g_fvgList[i].filled) continue;
      
      if(g_fvgList[i].isBullish && bias == 1)
      {
         if(low1 <= g_fvgList[i].topPrice && price > g_fvgList[i].topPrice)
         {
            if(!UseSilverBullet || inKZ)
            {
               g_fvgList[i].filled = true;
               return 1;
            }
         }
      }
      
      if(!g_fvgList[i].isBullish && bias == -1)
      {
         if(high1 >= g_fvgList[i].bottomPrice && price < g_fvgList[i].bottomPrice)
         {
            if(!UseSilverBullet || inKZ)
            {
               g_fvgList[i].filled = true;
               return -1;
            }
         }
      }
   }
   
   // Liquidity sweep
   for(int i = 0; i < ArraySize(g_liqLevels); i++)
   {
      if(g_liqLevels[i].swept) continue;
      
      // Bullish sweep (wicked below, closed above)
      if(!g_liqLevels[i].isHigh && bias == 1)
      {
         if(low1 < g_liqLevels[i].price && price > g_liqLevels[i].price)
         {
            if(!UseSilverBullet || inKZ)
            {
               g_liqLevels[i].swept = true;
               return 1;
            }
         }
      }
      
      // Bearish sweep (wicked above, closed below)
      if(g_liqLevels[i].isHigh && bias == -1)
      {
         if(high1 > g_liqLevels[i].price && price < g_liqLevels[i].price)
         {
            if(!UseSilverBullet || inKZ)
            {
               g_liqLevels[i].swept = true;
               return -1;
            }
         }
      }
   }
   
   return 0;
}

//+------------------------------------------------------------------+
void ExecuteTrade(int signal)
{
   g_sym.Refresh();
   int dig = (int)g_sym.Digits();
   
   double entry, sl, tp;
   ENUM_ORDER_TYPE orderType;
   
   if(signal == 1)
   {
      entry = g_sym.Ask();
      sl = NormalizeDouble(entry - g_currentATR * ATR_SL_Multiplier, dig);
      tp = NormalizeDouble(entry + g_currentATR * ATR_TP_Multiplier, dig);
      orderType = ORDER_TYPE_BUY;
   }
   else
   {
      entry = g_sym.Bid();
      sl = NormalizeDouble(entry + g_currentATR * ATR_SL_Multiplier, dig);
      tp = NormalizeDouble(entry - g_currentATR * ATR_TP_Multiplier, dig);
      orderType = ORDER_TYPE_SELL;
   }
   
   // Lot calculation
   double riskAmt = g_acc.Balance() * RiskPercent / 100.0;
   double slDist = MathAbs(entry - sl);
   double tickVal = g_sym.TickValue();
   double tickSz  = g_sym.TickSize();
   
   if(tickVal <= 0 || tickSz <= 0 || slDist <= 0) return;
   
   double dollarPerLot = (slDist / tickSz) * tickVal;
   double lots = riskAmt / dollarPerLot;
   
   double minLot = g_sym.LotsMin();
   double maxLot = g_sym.LotsMax();
   double stepLot = g_sym.LotsStep();
   
   lots = MathFloor(lots / stepLot) * stepLot;
   if(lots < minLot) lots = minLot;
   if(lots > maxLot) lots = maxLot;
   if(lots > 0.06) lots = 0.06;  // GFT hard cap
   
   if(g_trade.PositionOpen(_Symbol, orderType, lots, entry, sl, tp, "BAKOME_ICT"))
   {
      g_todayTrades++;
      
      // Track position
      int sz = ArraySize(g_tracked);
      ArrayResize(g_tracked, sz+1);
      g_tracked[sz].ticket = g_trade.ResultOrder();
      g_tracked[sz].openPrice = entry;
      g_tracked[sz].originalSL = sl;
      g_tracked[sz].breakEvenSet = false;
      g_tracked[sz].trailingActive = false;
      
      Print(signal==1 ? "BUY" : "SELL", " ", lots, " @ ", entry, " SL=", sl, " TP=", tp);
   }
}

//+------------------------------------------------------------------+
void ManagePositions()
{
   for(int i = ArraySize(g_tracked) - 1; i >= 0; i--)
   {
      if(!g_pos.SelectByTicket(g_tracked[i].ticket))
      {
         // Position closed, remove from tracking
         ArrayRemove(g_tracked, i, 1);
         continue;
      }
      
      double entry = g_tracked[i].openPrice;
      double slDist = MathAbs(entry - g_tracked[i].originalSL);
      if(slDist <= 0) continue;
      
      int posType = (int)g_pos.PositionType();
      double curPrice = (posType == POSITION_TYPE_BUY) ? g_sym.Bid() : g_sym.Ask();
      double profit = (posType == POSITION_TYPE_BUY) ? (curPrice - entry) : (entry - curPrice);
      
      // Break-even
      if(UseBreakEven && !g_tracked[i].breakEvenSet)
      {
         if(profit >= g_currentATR * BE_TriggerATR)
         {
            double newSL = entry;
            int dig = (int)g_sym.Digits();
            g_trade.PositionModify(g_tracked[i].ticket, NormalizeDouble(newSL, dig), g_pos.TakeProfit());
            g_tracked[i].breakEvenSet = true;
         }
      }
      
      // Trailing stop
      if(UseTrailingStop && g_tracked[i].breakEvenSet)
      {
         if(profit >= g_currentATR * Trail_StartATR)
         {
            int dig = (int)g_sym.Digits();
            double newSL;
            if(posType == POSITION_TYPE_BUY)
            {
               newSL = curPrice - g_currentATR * Trail_StepATR;
               if(newSL > g_pos.StopLoss())
                  g_trade.PositionModify(g_tracked[i].ticket, NormalizeDouble(newSL, dig), g_pos.TakeProfit());
            }
            else
            {
               newSL = curPrice + g_currentATR * Trail_StepATR;
               if(newSL < g_pos.StopLoss() || g_pos.StopLoss() == 0)
                  g_trade.PositionModify(g_tracked[i].ticket, NormalizeDouble(newSL, dig), g_pos.TakeProfit());
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
