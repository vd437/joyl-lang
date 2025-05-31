// iot_protocols.joyl - IoT Protocol Abstraction Layer
pub struct ProtocolBroker {
    mqtt_client: Option<MqttClient>,
    coap_client: Option<CoapClient>,
    active_protocol: IoTProtocol,
    message_queue: PriorityQueue<IoTMessage>,
    qos_manager: QoSManager
}

impl ProtocolBroker {
    /// Initialize with preferred protocol
    pub fn new(protocol: IoTProtocol) -> ProtocolBroker {
        ProtocolBroker {
            mqtt_client: None,
            coap_client: None,
            active_protocol: protocol,
            message_queue: PriorityQueue::new(),
            qos_manager: QoSManager::new()
        }
    }

    /// Connect to broker/server
    pub async fn connect(
        &mut self,
        endpoint: string,
        credentials: ProtocolCredentials
    ) -> Result<(), ProtocolError> {
        match self.active_protocol {
            MQTT => {
                let client = MqttClient::connect(endpoint, credentials).await?;
                self.mqtt_client = Some(client);
            },
            CoAP => {
                let client = CoapClient::connect(endpoint, credentials).await?;
                self.coap_client = Some(client);
            }
        }
        Ok(())
    }

    /// Publish message with QoS
    pub async fn publish(
        &mut self,
        topic: string,
        payload: string,
        qos: QoSLevel
    ) -> Result<MessageId, ProtocolError> {
        let message = IoTMessage::new(topic, payload, qos);
        self.qos_manager.track_message(message.id);
        
        match self.active_protocol {
            MQTT => self.mqtt_client.as_mut().unwrap().publish(message).await,
            CoAP => self.coap_client.as_mut().unwrap().publish(message).await
        }
    }

    /// Subscribe to topic
    pub async fn subscribe(
        &mut self,
        topic: string,
        callback: fn(IoTMessage)
    ) -> Result<SubscriptionId, ProtocolError> {
        match self.active_protocol {
            MQTT => self.mqtt_client.as_mut().unwrap().subscribe(topic, callback).await,
            CoAP => self.coap_client.as_mut().unwrap().subscribe(topic, callback).await
        }
    }
}