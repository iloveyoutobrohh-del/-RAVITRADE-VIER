import pandas as pd
import pandas_ta as ta
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Trading Indicator Analyzer")

# កត់ចំណាំ៖ កុំភ្លេចដាក់ OpenAI API Key របស់អ្នក
openai.api_key = "YOUR_OPENAI_API_KEY"

class MarketDataRequest(BaseModel):
    symbol: str               # ឧទាហរណ៍៖ "XAUUSD" ឬ "BTCUSDT"
    timeframe: str            # ឧទាហរណ៍៖ "1h", "4h"
    prices: list[dict]        # បញ្ជីទិន្នន័យ OHLCV (Open, High, Low, Close, Volume)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """គណនា Indicator តាមបែបបច្ចេកទេស"""
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['EMA_200'] = ta.ema(df['close'], length=200)
    return df

@app.post("/analyze")
async def analyze_market(data: MarketDataRequest):
    try:
        df = pd.DataFrame(data.prices)
        if df.empty or 'close' not in df.columns:
            raise HTTPException(status_code=400, detail="ទិន្នន័យតម្លៃមិនត្រឹមត្រូវ")

        # 1. គណនា Indicators
        df_calc = calculate_indicators(df)
        latest = df_calc.iloc[-1]

        # 2. រៀបចំ Prompt សម្រាប់ផ្ញើទៅ AI
        prompt_content = f"""
        អ្នកគឺជាអ្នកជំនាញវិភាគបច្ចេកទេស (Technical Analysis AI Agent) សម្រាប់ការជួញដូរ Forex & Crypto។
        សូមធ្វើការវិភាគលើគូររូបិយប័ណ្ណ/ទ្រព្យសកម្ម {data.symbol} នៅលើ Timeframe {data.timeframe} ផ្អែកលើ Indicators ខាងក្រោម៖
        
        - តម្លៃបច្ចុប្បន្ន (Current Close): {latest['close']}
        - RSI (14): {latest['RSI']:.2f}
        - MACD Value: {latest['MACD']:.4f} (Signal: {latest['MACD_signal']:.4f})
        - EMA 50: {latest['EMA_50']:.2f}
        - EMA 200: {latest['EMA_200']:.2f}

        សូមផ្តល់នូវលទ្ធផលវិភាគជាភាសាខ្មែរ ដោយរួមមាន៖
        1. និន្នាការទីផ្សារ (Trend Identification: Bullish / Bearish / Sideway)
        2. ការវិភាគ Indicators (RSI Overbought/Oversold, EMA Crossover)
        3. សញ្ញាជួញដូរ (Signal: Buy, Sell, or Wait)
        4. ការគ្រប់គ្រងហានិភ័យ (កំណត់តម្លៃ Entry, Stop-Loss, និង Take-Profit)
        """

        # 3. ហៅប្រើប្រាស់ AI Model
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional financial trader and AI indicator analyst."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.3
        )

        ai_analysis = response.choices[0].message.content

        return {
            "symbol": data.symbol,
            "timeframe": data.timeframe,
            "latest_indicators": {
                "close": latest['close'],
                "rsi": round(latest['RSI'], 2),
                "macd": round(latest['MACD'], 4),
                "ema50": round(latest['EMA_50'], 2),
                "ema200": round(latest['EMA_200'], 2)
            },
            "ai_analysis": ai_analysis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
