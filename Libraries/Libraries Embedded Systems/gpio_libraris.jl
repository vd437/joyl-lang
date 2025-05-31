/// Embedded Systems GPIO Library
/// Provides General Purpose Input/Output functionality
pub module GPIO {
    /// GPIO pin modes
    pub enum PinMode {
        Input,
        Output,
        InputPullUp,
        InputPullDown,
        Analog,
        AlternateFunction(u8)
    }

    /// GPIO pin speeds
    pub enum PinSpeed {
        Low,
        Medium,
        High,
        VeryHigh
    }

    /// GPIO pin configuration
    pub struct PinConfig {
        pub mode: PinMode,
        pub speed: PinSpeed,
        pub initial_state: bool
    }

    /// GPIO pin controller
    pub struct Pin {
        hal: Rc<RefCell<HAL>>,
        bank: char,
        num: u8,
        config: PinConfig
    }

    impl Pin {
        pub fn new(
            hal: Rc<RefCell<HAL>>,
            bank: char,
            num: u8,
            config: PinConfig
        ) -> Result<Self, HardwareError> {
            let mut pin = Pin { hal, bank, num, config };
            
            pin.configure()?;
            Ok(pin)
        }

        /// Configure pin according to settings
        fn configure(&mut self) -> Result<(), HardwareError> {
            let mode_val = match self.config.mode {
                PinMode::Input => 0b00,
                PinMode::Output => 0b01,
                PinMode::InputPullUp => 0b00, // PUPD configured separately
                PinMode::InputPullDown => 0b00,
                PinMode::Analog => 0b11,
                PinMode::AlternateFunction(_) => 0b10
            };
            
            let speed_val = match self.config.speed {
                PinMode::Low => 0b00,
                PinMode::Medium => 0b01,
                PinMode::High => 0b10,
                PinMode::VeryHigh => 0b11
            };
            
            // Calculate register values
            let moder = mode_val << (2 * self.num);
            let ospeedr = speed_val << (2 * self.num);
            
            // Apply configuration
            let mut hal = self.hal.borrow_mut();
            hal.write_register(&format!("GPIO{}_MODER", self.bank), moder)?;
            hal.write_register(&format!("GPIO{}_OSPEEDR", self.bank), ospeedr)?;
            
            // Set initial state for outputs
            if let PinMode::Output = self.config.mode {
                self.write(self.config.initial_state)?;
            }
            
            // Configure pull-up/pull-down
            if matches!(self.config.mode, PinMode::InputPullUp | PinMode::InputPullDown) {
                let pupdr = match self.config.mode {
                    PinMode::InputPullUp => 0b01,
                    PinMode::InputPullDown => 0b10,
                    _ => 0b00
                } << (2 * self.num);
                
                hal.write_register(&format!("GPIO{}_PUPDR", self.bank), pupdr)?;
            }
            
            Ok(())
        }

        /// Read pin state
        pub fn read(&self) -> Result<bool, HardwareError> {
            let idr = self.hal.borrow()
                .read_register(&format!("GPIO{}_IDR", self.bank))?;
            
            Ok((idr & (1 << self.num)) != 0)
        }

        /// Write pin state
        pub fn write(&mut self, state: bool) -> Result<(), HardwareError> {
            let mut hal = self.hal.borrow_mut();
            let reg = if state {
                format!("GPIO{}_BSRR", self.bank)
            } else {
                format!("GPIO{}_BRR", self.bank)
            };
            
            hal.write_register(&reg, 1 << self.num)?;
            Ok(())
        }

        /// Toggle pin state
        pub fn toggle(&mut self) -> Result<(), HardwareError> {
            let current = self.read()?;
            self.write(!current)
        }
    }

    /// GPIO Port controller
    pub struct Port {
        hal: Rc<RefCell<HAL>>,
        bank: char,
        pins: HashMap<u8, Pin>
    }

    impl Port {
        pub fn new(hal: Rc<RefCell<HAL>>, bank: char) -> Self {
            Port {
                hal,
                bank,
                pins: HashMap::new()
            }
        }
        
        /// Configure multiple pins at once
        pub fn configure_pins(
            &mut self,
            pins: Vec<(u8, PinConfig)>
        ) -> Result<(), HardwareError> {
            for (num, config) in pins {
                let pin = Pin::new(
                    self.hal.clone(),
                    self.bank,
                    num,
                    config
                )?;
                
                self.pins.insert(num, pin);
            }
            
            Ok(())
        }
        
        /// Read entire port
        pub fn read_port(&self) -> Result<u16, HardwareError> {
            self.hal.borrow()
                .read_register(&format!("GPIO{}_IDR", self.bank))
                .map(|v| v as u16)
        }
        
        /// Write entire port
        pub fn write_port(&mut self, value: u16) -> Result<(), HardwareError> {
            self.hal.borrow_mut()
                .write_register(&format!("GPIO{}_ODR", self.bank), value as u32)
        }
    }

    /// Interrupt-driven GPIO
    pub struct InterruptPin {
        pin: Pin,
        exti_line: u8,
        trigger: InterruptTrigger
    }

    impl InterruptPin {
        pub fn new(
            hal: Rc<RefCell<HAL>>,
            bank: char,
            num: u8,
            trigger: InterruptTrigger
        ) -> Result<Self, HardwareError> {
            let pin = Pin::new(
                hal.clone(),
                bank,
                num,
                PinConfig {
                    mode: PinMode::Input,
                    speed: PinSpeed::Low,
                    initial_state: false
                }
            )?;
            
            let exti_line = num;
            let mut interrupt_pin = InterruptPin { pin, exti_line, trigger };
            
            interrupt_pin.configure_interrupt()?;
            Ok(interrupt_pin)
        }
        
        fn configure_interrupt(&mut self) -> Result<(), HardwareError> {
            let mut hal = self.hal.borrow_mut();
            
            // Configure EXTI line
            let afr = match self.pin.bank {
                'A' => 0b0000,
                'B' => 0b0001,
                'C' => 0b0010,
                _ => return Err(HardwareError::UnsupportedOperation)
            };
            
            // Select EXTI line
            let exti_cr = hal.read_register("SYSCFG_EXTICR")?;
            let new_exti_cr = exti_cr | (afr << (4 * (self.exti_line % 4)));
            hal.write_register("SYSCFG_EXTICR", new_exti_cr)?;
            
            // Configure trigger
            let trigger_val = match self.trigger {
                InterruptTrigger::Rising => 0b01,
                InterruptTrigger::Falling => 0b10,
                InterruptTrigger::Both => 0b11
            };
            
            hal.write_register("EXTI_RTSR", trigger_val & 0b01)?;
            hal.write_register("EXTI_FTSR", (trigger_val >> 1) & 0b01)?;
            
            // Enable interrupt
            hal.write_register("EXTI_IMR", 1 << self.exti_line)?;
            
            Ok(())
        }
        
        /// Set interrupt callback
        pub fn set_callback<F>(&mut self, callback: F) -> Result<(), HardwareError>
        where
            F: Fn() + 'static
        {
            let vector = match self.exti_line {
                0..=4 => InterruptType::EXTI0 + self.exti_line as u8,
                5..=9 => InterruptType::EXTI9_5,
                10..=15 => InterruptType::EXTI15_10,
                _ => return Err(HardwareError::InvalidInterruptLine)
            };
            
            self.hal.borrow_mut()
                .register_interrupt(vector, Box::new(callback))?;
            
            Ok(())
        }
    }
}