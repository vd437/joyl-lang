/// FinTech Payment Processing Library
/// Provides secure payment processing capabilities
pub module PaymentProcessing {
    /// Supported payment methods
    pub enum PaymentMethod {
        CreditCard,
        BankTransfer,
        DigitalWallet,
        Cryptocurrency
    }

    /// Payment processor interface
    pub trait PaymentProcessor {
        fn process_payment(&self, payment: PaymentRequest) -> Result<PaymentResponse, PaymentError>;
        fn refund_payment(&self, refund: RefundRequest) -> Result<RefundResponse, PaymentError>;
        fn verify_transaction(&self, transaction_id: &str) -> Result<TransactionStatus, PaymentError>;
    }

    /// Credit card processor
    pub struct CreditCardProcessor {
        api_key: String,
        encryption_key: String,
        fraud_detector: FraudDetector
    }

    impl CreditCardProcessor {
        pub fn new(api_key: &str, encryption_key: &str) -> Self {
            CreditCardProcessor {
                api_key: api_key.to_string(),
                encryption_key: encryption_key.to_string(),
                fraud_detector: FraudDetector::new()
            }
        }
        
        /// Tokenize credit card for secure storage
        pub fn tokenize_card(&self, card: CreditCard) -> Result<String, PaymentError> {
            let encrypted = encrypt_card_data(card, &self.encryption_key)?;
            Ok(generate_token(&encrypted))
        }
    }

    impl PaymentProcessor for CreditCardProcessor {
        fn process_payment(&self, payment: PaymentRequest) -> Result<PaymentResponse, PaymentError> {
            // Fraud check
            if self.fraud_detector.is_high_risk(&payment) {
                return Err(PaymentError::FraudDetected);
            }
            
            // Process payment
            let response = process_credit_card_payment(
                &self.api_key,
                payment.amount,
                payment.currency,
                payment.card_token
            )?;
            
            Ok(PaymentResponse {
                transaction_id: response.transaction_id,
                status: response.status,
                timestamp: Utc::now()
            })
        }
        
        fn refund_payment(&self, refund: RefundRequest) -> Result<RefundResponse, PaymentError> {
            let response = process_credit_card_refund(
                &self.api_key,
                refund.transaction_id,
                refund.amount
            )?;
            
            Ok(RefundResponse {
                refund_id: response.refund_id,
                status: response.status,
                timestamp: Utc::now()
            })
        }
    }

    /// Payment gateway router
    pub struct PaymentGateway {
        processors: HashMap<PaymentMethod, Box<dyn PaymentProcessor>>,
        routing_strategy: RoutingStrategy,
        transaction_logger: TransactionLogger
    }

    impl PaymentGateway {
        pub fn new(strategy: RoutingStrategy) -> Self {
            PaymentGateway {
                processors: HashMap::new(),
                routing_strategy: strategy,
                transaction_logger: TransactionLogger::new()
            }
        }
        
        /// Add payment processor
        pub fn add_processor(&mut self, method: PaymentMethod, processor: Box<dyn PaymentProcessor>) {
            self.processors.insert(method, processor);
        }
        
        /// Process payment with automatic routing
        pub fn process(&mut self, payment: PaymentRequest) -> Result<PaymentResponse, PaymentError> {
            let processor = self.select_processor(&payment.method)?;
            let response = processor.process_payment(payment)?;
            
            // Log transaction
            self.transaction_logger.log(
                &response.transaction_id,
                response.status,
                response.timestamp
            );
            
            Ok(response)
        }
        
        fn select_processor(&self, method: &PaymentMethod) -> Result<&Box<dyn PaymentProcessor>, PaymentError> {
            match self.routing_strategy {
                RoutingStrategy::Primary => self.processors.get(method)
                    .ok_or(PaymentError::ProcessorNotFound),
                RoutingStrategy::Fallback => {
                    // Try primary first, then any available fallback
                    if let Some(processor) = self.processors.get(method) {
                        Ok(processor)
                    } else {
                        self.processors.values().next()
                            .ok_or(PaymentError::NoProcessorsAvailable)
                    }
                }
            }
        }
    }

    /// Recurring payment manager
    pub struct RecurringPaymentManager {
        scheduler: PaymentScheduler,
        retry_policy: RetryPolicy,
        subscription_repository: SubscriptionRepository
    }

    impl RecurringPaymentManager {
        pub fn new(retry_policy: RetryPolicy) -> Self {
            RecurringPaymentManager {
                scheduler: PaymentScheduler::new(),
                retry_policy,
                subscription_repository: SubscriptionRepository::new()
            }
        }
        
        /// Schedule recurring payment
        pub fn schedule_payment(
            &mut self,
            subscription: Subscription
        ) -> Result<(), PaymentError> {
            self.subscription_repository.save(&subscription)?;
            self.scheduler.schedule(
                subscription.next_payment_date(),
                subscription.id()
            );
            
            Ok(())
        }
        
        /// Process due payments
        pub async fn process_due_payments(&mut self, gateway: &mut PaymentGateway) -> Vec<PaymentResult> {
            let due = self.scheduler.get_due_payments();
            let mut results = Vec::new();
            
            for sub_id in due {
                if let Some(subscription) = self.subscription_repository.get(sub_id) {
                    let mut attempts = 0;
                    let mut result = None;
                    
                    while attempts < self.retry_policy.max_attempts && result.is_none() {
                        let payment = PaymentRequest {
                            amount: subscription.amount(),
                            currency: subscription.currency(),
                            method: subscription.payment_method(),
                            card_token: subscription.card_token()
                        };
                        
                        match gateway.process(payment) {
                            Ok(response) => {
                                result = Some(PaymentResult::Success {
                                    subscription_id: sub_id,
                                    transaction_id: response.transaction_id
                                });
                                
                                // Update next payment date if successful
                                let mut sub = subscription;
                                sub.update_next_payment();
                                self.subscription_repository.save(&sub)?;
                            },
                            Err(e) => {
                                attempts += 1;
                                if attempts >= self.retry_policy.max_attempts {
                                    result = Some(PaymentResult::Failure {
                                        subscription_id: sub_id,
                                        error: e
                                    });
                                } else {
                                    sleep(self.retry_policy.delay_between_attempts).await;
                                }
                            }
                        }
                    }
                    
                    if let Some(res) = result {
                        results.push(res);
                    }
                }
            }
            
            results
        }
    }
}