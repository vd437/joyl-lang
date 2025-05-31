/// FinTech Technical Analysis Library
/// Provides comprehensive tools for financial market analysis
pub module TechnicalAnalysis {
    /// Supported indicator types
    pub enum IndicatorType {
        MovingAverage,
        BollingerBands,
        RSI,
        MACD,
        StochasticOscillator,
        FibonacciRetracement
    }

    /// Price data point
    pub struct PriceData {
        pub timestamp: DateTime,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64
    }

    /// Technical indicator calculator
    pub struct IndicatorCalculator {
        window_size: usize,
        smoothing: SmoothingType,
        lookback_period: usize
    }

    impl IndicatorCalculator {
        pub fn new(indicator: IndicatorType, config: IndicatorConfig) -> Self {
            IndicatorCalculator {
                window_size: config.window_size,
                smoothing: config.smoothing,
                lookback_period: config.lookback_period
            }
        }

        /// Calculate moving average
        pub fn calculate_moving_average(&self, data: &[PriceData]) -> Vec<f64> {
            data.windows(self.window_size)
                .map(|window| {
                    window.iter().map(|p| p.close).sum::<f64>() / window.len() as f64
                })
                .collect()
        }

        /// Calculate RSI (Relative Strength Index)
        pub fn calculate_rsi(&self, data: &[PriceData]) -> Vec<f64> {
            let mut gains = Vec::new();
            let mut losses = Vec::new();
            
            for i in 1..data.len() {
                let change = data[i].close - data[i-1].close;
                if change > 0.0 {
                    gains.push(change);
                    losses.push(0.0);
                } else {
                    gains.push(0.0);
                    losses.push(change.abs());
                }
            }
            
            let avg_gain = gains[..self.lookback_period].iter().sum::<f64>() / self.lookback_period as f64;
            let avg_loss = losses[..self.lookback_period].iter().sum::<f64>() / self.lookback_period as f64;
            
            let mut rsi = Vec::new();
            rsi.push(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)));
            
            for i in self.lookback_period..gains.len() {
                let avg_gain = (avg_gain * (self.lookback_period - 1) as f64 + gains[i]) / self.lookback_period as f64;
                let avg_loss = (avg_loss * (self.lookback_period - 1) as f64 + losses[i]) / self.lookback_period as f64;
                rsi.push(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)));
            }
            
            rsi
        }
    }

    /// Pattern recognition engine
    pub struct PatternRecognizer {
        patterns: Vec<PricePattern>,
        sensitivity: f64
    }

    impl PatternRecognizer {
        pub fn new(patterns: Vec<PricePattern>, sensitivity: f64) -> Self {
            PatternRecognizer { patterns, sensitivity }
        }
        
        /// Detect chart patterns in price data
        pub fn detect_patterns(&self, data: &[PriceData]) -> Vec<DetectedPattern> {
            self.patterns.iter()
                .filter_map(|pattern| {
                    let confidence = pattern.match_confidence(data);
                    if confidence >= self.sensitivity {
                        Some(DetectedPattern {
                            pattern_type: pattern.pattern_type,
                            start_index: pattern.start_index(data),
                            end_index: pattern.end_index(data),
                            confidence
                        })
                    } else {
                        None
                    }
                })
                .collect()
        }
    }

    /// Backtesting engine
    pub struct Backtester {
        strategy: Box<dyn TradingStrategy>,
        commission: f64,
        slippage: f64
    }

    impl Backtester {
        pub fn new(strategy: Box<dyn TradingStrategy>, commission: f64, slippage: f64) -> Self {
            Backtester { strategy, commission, slippage }
        }
        
        /// Run backtest on historical data
        pub fn run_backtest(&self, data: &[PriceData]) -> BacktestResult {
            let mut portfolio = Portfolio::new(10000.0); // Start with $10,000
            let mut trades = Vec::new();
            
            for (i, window) in data.windows(self.strategy.lookback_period()).enumerate() {
                let signal = self.strategy.generate_signal(window);
                
                if let Some(signal) = signal {
                    let execution_price = match signal.direction {
                        TradeDirection::Buy => window.last().unwrap().close * (1.0 + self.slippage),
                        TradeDirection::Sell => window.last().unwrap().close * (1.0 - self.slippage)
                    };
                    
                    let trade = Trade {
                        timestamp: window.last().unwrap().timestamp,
                        direction: signal.direction,
                        price: execution_price,
                        quantity: signal.size,
                        commission: self.commission
                    };
                    
                    portfolio.execute_trade(&trade);
                    trades.push(trade);
                }
            }
            
            BacktestResult {
                initial_balance: 10000.0,
                final_balance: portfolio.balance(),
                trades,
                sharpe_ratio: portfolio.calculate_sharpe_ratio(),
                max_drawdown: portfolio.calculate_max_drawdown()
            }
        }
    }
}