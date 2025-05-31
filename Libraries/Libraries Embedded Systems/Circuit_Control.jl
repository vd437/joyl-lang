/// Embedded Systems Circuit Control Library
/// Provides low-level hardware interface for circuit control
pub module CircuitControl {
    /// Supported microcontroller architectures
    pub enum Microcontroller {
        ARM_CortexM,
        AVR,
        PIC,
        RISC_V,
        ESP32
    }

    /// Circuit configuration
    pub struct CircuitConfig {
        pub clock_speed: u32,  // Hz
        pub voltage: f32,      // Volts
        pub gpio_banks: u8,
        pub adc_resolution: u8,
        pub pwm_channels: u8
    }

    /// Hardware Abstraction Layer
    pub struct HAL {
        mcu: Microcontroller,
        config: CircuitConfig,
        registers: HashMap<String, u32>,
        interrupt_handlers: Vec<InterruptHandler>
    }

    impl HAL {
        /// Initialize hardware abstraction layer
        pub fn init(mcu: Microcontroller, config: CircuitConfig) -> Result<Self, HardwareError> {
            let mut hal = HAL {
                mcu,
                config,
                registers: HashMap::new(),
                interrupt_handlers: Vec::new()
            };
            
            // Initialize registers based on MCU type
            match mcu {
                Microcontroller::ARM_CortexM => hal.init_cortex_m()?,
                Microcontroller::AVR => hal.init_avr()?,
                _ => return Err(HardwareError::UnsupportedMCU)
            }
            
            Ok(hal)
        }

        /// Read from memory-mapped register
        pub fn read_register(&self, name: &str) -> Result<u32, HardwareError> {
            self.registers.get(name)
                .copied()
                .ok_or(HardwareError::RegisterNotFound)
        }

        /// Write to memory-mapped register
        pub fn write_register(&mut self, name: &str, value: u32) -> Result<(), HardwareError> {
            let reg = self.registers.get_mut(name)
                .ok_or(HardwareError::RegisterNotFound)?;
            
            *reg = value;
            Ok(())
        }

        /// Set clock speed
        pub fn set_clock_speed(&mut self, speed: u32) -> Result<(), HardwareError> {
            match self.mcu {
                Microcontroller::ARM_CortexM => {
                    let pll_config = calculate_pll(speed, self.config.clock_speed);
                    self.write_register("PLL_CFGR", pll_config)?;
                },
                _ => return Err(HardwareError::UnsupportedOperation)
            }
            
            self.config.clock_speed = speed;
            Ok(())
        }

        /// Register interrupt handler
        pub fn register_interrupt(
            &mut self, 
            interrupt: InterruptType,
            handler: InterruptHandler
        ) -> Result<(), HardwareError> {
            let vector_num = get_interrupt_vector(interrupt, self.mcu)?;
            
            if vector_num >= self.interrupt_handlers.len() {
                self.interrupt_handlers.resize(vector_num + 1, dummy_interrupt_handler);
            }
            
            self.interrupt_handlers[vector_num] = handler;
            Ok(())
        }
    }

    /// Timer/Counter Controller
    pub struct Timer {
        hal: Rc<RefCell<HAL>>,
        timer_num: u8,
        mode: TimerMode
    }

    impl Timer {
        pub fn new(hal: Rc<RefCell<HAL>>, timer_num: u8) -> Result<Self, HardwareError> {
            let mut timer = Timer {
                hal,
                timer_num,
                mode: TimerMode::Disabled
            };
            
            timer.init()?;
            Ok(timer)
        }

        /// Start timer in specified mode
        pub fn start(&mut self, mode: TimerMode) -> Result<(), HardwareError> {
            let config = match mode {
                TimerMode::Periodic(interval) => {
                    let ticks = (interval * self.hal.borrow().config.clock_speed as f32) as u32;
                    (0x1 << 7) | (ticks & 0xFFFF_FFFF)
                },
                TimerMode::PWM(channel, duty) => {
                    let max = self.get_max_value()?;
                    let compare = (duty * max as f32) as u32;
                    (0x3 << 7) | ((channel as u32) << 5) | (compare & 0xFFFF)
                },
                _ => 0x0
            };
            
            self.hal.borrow_mut()
                .write_register(&format!("TIM{}_CR", self.timer_num), config)?;
            
            self.mode = mode;
            Ok(())
        }

        /// Get current timer value
        pub fn get_value(&self) -> Result<u32, HardwareError> {
            self.hal.borrow()
                .read_register(&format!("TIM{}_CNT", self.timer_num))
        }
    }

    /// Analog-to-Digital Converter (ADC)
    pub struct ADC {
        hal: Rc<RefCell<HAL>>,
        channel: u8,
        resolution: u8
    }

    impl ADC {
        pub fn new(hal: Rc<RefCell<HAL>>, channel: u8) -> Result<Self, HardwareError> {
            let resolution = hal.borrow().config.adc_resolution;
            let mut adc = ADC { hal, channel, resolution };
            
            adc.init()?;
            Ok(adc)
        }

        /// Read analog value
        pub fn read(&self) -> Result<u16, HardwareError> {
            let mut hal = self.hal.borrow_mut();
            
            // Start conversion
            hal.write_register("ADC_CR", 1 << self.channel)?;
            
            // Wait for completion
            while hal.read_register("ADC_SR")? & (1 << self.channel) == 0 {}
            
            // Read result
            let raw = hal.read_register(&format!("ADC_DR{}", self.channel))?;
            Ok((raw & ((1 << self.resolution) - 1)) as u16)
        }

        /// Convert to voltage
        pub fn read_voltage(&self, vref: f32) -> Result<f32, HardwareError> {
            let raw = self.read()?;
            Ok((raw as f32) / ((1 << self.resolution) as f32) * vref)
        }
    }
}