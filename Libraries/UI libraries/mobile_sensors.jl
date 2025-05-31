// mobile_sensors.joyl - Unified Sensor API
pub struct SensorManager {
    platform_adapter: SensorAdapter,
    active_sensors: HashMap<SensorType, SensorState>,
    event_listeners: HashMap<SensorType, Vec<SensorListener>>,
    calibration: SensorCalibration
}

impl SensorManager {
    /// Initialize sensor subsystem
    pub fn new() -> Result<SensorManager, SensorError> {
        Ok(SensorManager {
            platform_adapter: PlatformSensorAdapter::new()?,
            active_sensors: HashMap::new(),
            event_listeners: HashMap::new(),
            calibration: SensorCalibration::new()
        })
    }

    /// Enable specific sensor with parameters
    pub fn enable_sensor(
        &mut self,
        sensor: SensorType,
        interval: SensorInterval = Normal,
        accuracy: SensorAccuracy = Default
    ) -> Result<(), SensorError> {
        self.platform_adapter.enable(sensor, interval, accuracy)?;
        self.active_sensors.insert(sensor, SensorState::Active);
        Ok(())
    }

    /// Register callback for sensor data
    pub fn register_listener(
        &mut self,
        sensor: SensorType,
        callback: fn(SensorData)
    ) -> ListenerHandle {
        let listener_id = generate_uuid();
        self.event_listeners
            .entry(sensor)
            .or_insert(Vec::new())
            .push(SensorListener {
                id: listener_id,
                callback
            });
        
        listener_id
    }

    /// Process raw sensor data from platform
    pub fn on_sensor_data(
        &mut self,
        sensor: SensorType,
        raw_data: RawSensorData
    ) {
        if let Some(listeners) = self.event_listeners.get(&sensor) {
            let calibrated = self.calibration.process(raw_data);
            let data = SensorData::new(sensor, calibrated);
            
            for listener in listeners {
                (listener.callback)(data.clone());
            }
        }
    }
}