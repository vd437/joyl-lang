/// FinTech Market Data Library
/// Provides real-time and historical market data connectivity
pub module MarketData {
    /// Supported data providers
    pub enum DataProvider {
        Bloomberg,
        Reuters,
        AlphaVantage,
        YahooFinance,
        Custom(String)
    }

    /// Market data feed
    pub struct MarketDataFeed {
        provider: DataProvider,
        ws_client: WebSocketClient,
        subscriptions: HashMap<String, Subscription>,
        buffer: DataBuffer
    }

    impl MarketDataFeed {
        pub async fn connect(provider: DataProvider, api_key: Option<&str>) -> Result<Self, MarketDataError> {
            let endpoint = match provider {
                DataProvider::Bloomberg => "wss://bloomberg-market-data.com/ws",
                DataProvider::Reuters => "wss://reuters-market-data.com/feed",
                DataProvider::AlphaVantage => format!("wss://alpha-vantage.com/stream?apikey={}", api_key.unwrap()),
                _ => return Err(MarketDataError::UnsupportedProvider)
            };
            
            let ws_client = WebSocketClient::connect(endpoint).await?;
            
            Ok(MarketDataFeed {
                provider,
                ws_client,
                subscriptions: HashMap::new(),
                buffer: DataBuffer::new(1000) // Keep last 1000 data points
            })
        }
        
        /// Subscribe to market data
        pub async fn subscribe(&mut self, symbol: &str, data_type: DataType) -> Result<String, MarketDataError> {
            let sub_id = Uuid::new_v4().to_string();
            let message = match self.provider {
                DataProvider::Bloomberg => json!({
                    "type": "subscribe",
                    "symbol": symbol,
                    "dataType": data_type.to_string()
                }),
                DataProvider::AlphaVantage => json!({
                    "function": data_type.to_alpha_vantage_function(),
                    "symbol": symbol,
                    "apikey": "YOUR_API_KEY"
                }),
                _ => return Err(MarketDataError::UnsupportedOperation)
            };
            
            self.ws_client.send(message.to_string()).await?;
            self.subscriptions.insert(sub_id.clone(), Subscription::new(symbol, data_type));
            
            Ok(sub_id)
        }
        
        /// Process incoming market data
        pub async fn process_data(&mut self) -> Result<(), MarketDataError> {
            while let Some(message) = self.ws_client.receive().await {
                let data: MarketDataMessage = serde_json::from_str(&message)?;
                self.buffer.push(data);
                
                // Handle real-time alerts
                if let Some(alert_conditions) = self.check_alerts(&data) {
                    self.handle_alerts(alert_conditions).await?;
                }
            }
            
            Ok(())
        }
    }

    /// Historical data client
    pub struct HistoricalDataClient {
        provider: DataProvider,
        http_client: HttpClient,
        cache: DataCache
    }

    impl HistoricalDataClient {
        pub fn new(provider: DataProvider) -> Self {
            HistoricalDataClient {
                provider,
                http_client: HttpClient::new(),
                cache: DataCache::new()
            }
        }
        
        /// Fetch historical price data
        pub async fn get_historical_data(
            &mut self,
            symbol: &str,
            interval: DataInterval,
            start: DateTime,
            end: DateTime
        ) -> Result<Vec<PriceData>, MarketDataError> {
            // Check cache first
            if let Some(cached) = self.cache.get(symbol, interval, start, end) {
                return Ok(cached);
            }
            
            let url = match self.provider {
                DataProvider::AlphaVantage => format!(
                    "https://www.alphavantage.co/query?function=TIME_SERIES_{}&symbol={}&apikey=YOUR_API_KEY",
                    interval.to_alpha_vantage_function(),
                    symbol
                ),
                DataProvider::YahooFinance => format!(
                    "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval={}&period1={}&period2={}",
                    symbol,
                    interval.to_yahoo_interval(),
                    start.timestamp(),
                    end.timestamp()
                ),
                _ => return Err(MarketDataError::UnsupportedOperation)
            };
            
            let response = self.http_client.get(&url).send().await?;
            let data = response.json::<HistoricalDataResponse>().await?;
            let prices = data.to_price_data();
            
            // Cache the results
            self.cache.store(symbol, interval, start, end, prices.clone());
            
            Ok(prices)
        }
    }

    /// Market data aggregator
    pub struct DataAggregator {
        sources: Vec<MarketDataFeed>,
        consolidation_strategy: ConsolidationStrategy,
        data_quality_checker: DataQualityChecker
    }

    impl DataAggregator {
        pub fn new(sources: Vec<MarketDataFeed>, strategy: ConsolidationStrategy) -> Self {
            DataAggregator {
                sources,
                consolidation_strategy: strategy,
                data_quality_checker: DataQualityChecker::new()
            }
        }
        
        /// Get consolidated market data
        pub async fn get_consolidated_data(&mut self, symbol: &str) -> Result<ConsolidatedData, MarketDataError> {
            let mut all_data = Vec::new();
            
            for source in &mut self.sources {
                if let Ok(data) = source.get_latest(symbol).await {
                    if self.data_quality_checker.check_quality(&data) {
                        all_data.push(data);
                    }
                }
            }
            
            match self.consolidation_strategy {
                ConsolidationStrategy::TimeWeighted => Ok(consolidate_time_weighted(all_data)),
                ConsolidationStrategy::VolumeWeighted => Ok(consolidate_volume_weighted(all_data)),
                ConsolidationStrategy::SourcePriority => Ok(consolidate_by_source_priority(all_data)),
                _ => Err(MarketDataError::InvalidConsolidationStrategy)
            }
        }
    }
}