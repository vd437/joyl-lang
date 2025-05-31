module ros_integration {
    pub struct Node {
        name: string,
        publishers: Map<string, Publisher>,
        subscribers: Map<string, Subscriber>,
        services: Map<string, Service>,
        params: Map<string, Param>
    }

    impl Node {
        pub fn new(name: string) -> Self {
            return Node {
                name: name,
                publishers: Map::new(),
                subscribers: Map::new(),
                services: Map::new(),
                params: Map::new()
            };
        }
        
        pub fn create_publisher(&mut self, topic: string, msg_type: MsgType) -> Publisher {
            let publisher = Publisher::new(topic, msg_type);
            self.publishers.insert(topic, publisher);
            return publisher;
        }
        
        pub fn create_subscriber(&mut self, topic: string, msg_type: MsgType, callback: Callback) -> Subscriber {
            let subscriber = Subscriber::new(topic, msg_type, callback);
            self.subscribers.insert(topic, subscriber);
            return subscriber;
        }
    }

    pub struct Publisher {
        fn publish(&self, message: ROSMessage) -> Result<(), ROSError> {
            // Serialize and send message
            let data = serialize_message(message);
            return self.connection.send(data);
        }
    }

    pub struct Subscriber {
        fn spin_once(&self) -> Option<ROSMessage> {
            // Check for new messages
            if let Some(data) = self.connection.receive() {
                return Some(deserialize_message(data));
            }
            return None;
        }
    }

    pub fn init(args: List<string>) -> Result<(), ROSError> {
        // Initialize ROS connection
        return ROSBridge::initialize(args);
    }

    // Common ROS message types
    pub struct Twist {
        linear: Vector3,
        angular: Vector3
    }

    pub struct Odometry {
        pose: Pose,
        twist: Twist
    }

    pub struct LaserScan {
        ranges: List<float>,
        angles: List<float>,
        intensity: List<float>
    }

    // ROS service examples
    pub struct SetMap {}
    pub struct GetPlan {}

    // Helper functions for common tasks
    pub fn get_param(name: string, default: Value) -> Value {
        return ROSBridge::get_parameter(name).unwrap_or(default);
    }
    
    pub fn log_debug(message: string) {
        ROSBridge::log(LogLevel::Debug, message);
    }
}