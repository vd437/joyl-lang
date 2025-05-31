// iot_connectivity.joyl - Unified IoT Connectivity
pub struct IoTConnectionManager {
    bluetooth: BluetoothController,
    wifi: WiFiController,
    current_protocol: IoProtocol,
    device_registry: HashMap<string, IoTDevice>,
    security: IoTSecurity
}

impl IoTConnectionManager {
    /// Initialize with default protocol
    pub fn new(default_protocol: IoProtocol) -> IoTConnectionManager {
        IoTConnectionManager {
            bluetooth: BluetoothController::new(),
            wifi: WiFiController::new(),
            current_protocol: default_protocol,
            device_registry: HashMap::new(),
            security: IoTSecurity::new()
        }
    }

    /// Scan for nearby devices
    pub async fn scan_devices(
        &mut self,
        timeout: Duration
    ) -> Result<Vec<IoTDevice>, IoError> {
        match self.current_protocol {
            Bluetooth => self.bluetooth.discover_devices(timeout).await,
            WiFi => self.wifi.discover_devices(timeout).await,
            _ => Err(IoError::UnsupportedProtocol)
        }
    }

    /// Connect to device with secure handshake
    pub async fn connect_device(
        &mut self,
        device_id: string,
        credentials: ConnectionCredentials
    ) -> Result<DeviceHandle, IoError> {
        let encrypted_creds = self.security.encrypt(credentials)?;
        
        let device = match self.current_protocol {
            Bluetooth => self.bluetooth.connect(device_id, encrypted_creds).await,
            WiFi => self.wifi.connect(device_id, encrypted_creds).await,
            _ => return Err(IoError::UnsupportedProtocol)
        }?;
        
        self.device_registry.insert(device_id.clone(), device.clone());
        Ok(DeviceHandle::new(device_id))
    }

    /// Send command to device
    pub async fn send_command(
        &mut self,
        device: &DeviceHandle,
        command: IoTCommand
    ) -> Result<CommandResponse, IoError> {
        let encrypted_cmd = self.security.encrypt_command(command)?;
        
        if let Some(device) = self.device_registry.get(device.id()) {
            match device.connection_type {
                Bluetooth => self.bluetooth.send(device, encrypted_cmd).await,
                WiFi => self.wifi.send(device, encrypted_cmd).await
            }
        } else {
            Err(IoError::DeviceNotConnected)
        }
    }
}