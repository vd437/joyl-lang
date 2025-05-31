// iot_sensors.joyl - Industrial IoT Sensor Integration
pub struct SensorHub {
    connected_sensors: HashMap<string, SensorDevice>,
    data_pipeline: SensorDataPipeline,
    calibration: SensorCalibration,
    sampling_interval: Duration
}

impl SensorHub {
    /// Initialize with sampling configuration
    pub fn new(interval: Duration) -> SensorHub {
        SensorHub {
            connected_sensors: HashMap::new(),
            data_pipeline: SensorDataPipeline::new(),
            calibration: SensorCalibration::new(),
            sampling_interval: interval
        }
    }

    /// Register new sensor
    pub fn register_sensor(
        &mut self,
        sensor: SensorDevice,
        calibration_profile: Option<CalibrationProfile>
    ) -> Result<SensorHandle, SensorError> {
        let calibrated_sensor = if let Some(profile) = calibration_profile {
            self.calibration.calibrate_sensor(sensor, profile)?
        } else {
            sensor
        };
        
        let handle = SensorHandle::new(calibrated_sensor.id.clone());
        self.connected_sensors.insert(calibrated_sensor.id.clone(), calibrated_sensor);
        Ok(handle)
    }

    /// Start continuous data stream
    pub async fn start_monitoring(
        &mut self,
        callback: fn(SensorDataBatch)
    ) -> Result<StreamHandle, SensorError> {
        let stream = self.data_pipeline.start_stream(
            &self.connected_sensors,
            self.sampling_interval,
            callback
        ).await?;
        
        Ok(stream)
    }

    /// Get single reading from sensor
    pub async fn read_sensor(
        &mut self,
        sensor: &SensorHandle
    ) -> Result<SensorReading, SensorError> {
        if let Some(sensor) = self.connected_sensors.get(sensor.id()) {
            let raw = sensor.read().await?;
            Ok(self.calibration.apply_calibration(raw))
        } else {
            Err(SensorError::SensorNotFound)
        }
    }
}