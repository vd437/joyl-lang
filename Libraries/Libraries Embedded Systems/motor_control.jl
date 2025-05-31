/// Embedded Systems Motor Control Library
/// Provides precise control for various motor types
pub module MotorControl {
    /// Supported motor types
    pub enum MotorType {
        DC,
        Stepper,
        BrushlessDC,
        Servo
    }

    /// Motor configuration
    pub struct MotorConfig {
        pub motor_type: MotorType,
        pub max_voltage: f32,
        pub max_current: f32,
        pub steps_per_rev: Option<u32>,  // For steppers
        pub pwm_frequency: u32
    }

    /// DC Motor controller
    pub struct DCMotor {
        pwm_pin: PWM,
        in1: GPIO::Pin,
        in2: GPIO::Pin,
        config: MotorConfig,
        current_speed: f32  // -1.0 to 1.0
    }

    impl DCMotor {
        pub fn new(
            pwm: PWM,
            in1: GPIO::Pin,
            in2: GPIO::Pin,
            config: MotorConfig
        ) -> Result<Self, MotorError> {
            if config.motor_type != MotorType::DC {
                return Err(MotorError::InvalidMotorType);
            }
            
            let mut motor = DCMotor {
                pwm_pin: pwm,
                in1,
                in2,
                config,
                current_speed: 0.0
            };
            
            motor.brake()?;
            Ok(motor)
        }
        
        /// Set motor speed (-1.0 to 1.0)
        pub fn set_speed(&mut self, speed: f32) -> Result<(), MotorError> {
            let speed = speed.clamp(-1.0, 1.0);
            
            if speed.abs() < 0.001 {
                self.brake()?;
            } else if speed > 0.0 {
                self.in1.write(true)?;
                self.in2.write(false)?;
                self.pwm_pin.set_duty(speed as f32)?;
            } else {
                self.in1.write(false)?;
                self.in2.write(true)?;
                self.pwm_pin.set_duty(speed.abs() as f32)?;
            }
            
            self.current_speed = speed;
            Ok(())
        }
        
        /// Stop motor immediately (brake)
        pub fn brake(&mut self) -> Result<(), MotorError> {
            self.in1.write(true)?;
            self.in2.write(true)?;
            self.pwm_pin.set_duty(0.0)?;
            self.current_speed = 0.0;
            Ok(())
        }
    }

    /// Stepper Motor controller
    pub struct StepperMotor {
        pins: [GPIO::Pin; 4],
        config: MotorConfig,
        current_step: u32,
        step_delay_us: u32,
        microsteps: u8
    }

    impl StepperMotor {
        pub fn new(
            pins: [GPIO::Pin; 4],
            config: MotorConfig
        ) -> Result<Self, MotorError> {
            if config.motor_type != MotorType::Stepper || config.steps_per_rev.is_none() {
                return Err(MotorError::InvalidMotorType);
            }
            
            Ok(StepperMotor {
                pins,
                config,
                current_step: 0,
                step_delay_us: 0,
                microsteps: 1
            })
        }
        
        /// Set motor speed in RPM
        pub fn set_speed(&mut self, rpm: f32) -> Result<(), MotorError> {
            let steps_per_rev = self.config.steps_per_rev.unwrap() * self.microsteps as u32;
            let delay = (60_000_000.0 / (rpm * steps_per_rev as f32)) as u32;
            
            self.step_delay_us = delay;
            Ok(())
        }
        
        /// Move specified number of steps
        pub async fn step(&mut self, steps: i32) -> Result<(), MotorError> {
            let direction = steps.signum();
            let steps = steps.abs() as u32;
            
            for _ in 0..steps {
                self.current_step = (self.current_step as i32 + direction) as u32 % 4;
                self.do_step(self.current_step)?;
                Timer::delay_us(self.step_delay_us).await;
            }
            
            Ok(())
        }
        
        /// Execute single step
        fn do_step(&mut self, step: u32) -> Result<(), MotorError> {
            // Wave drive pattern
            let pattern = match step % 4 {
                0 => [true, false, false, false],
                1 => [false, true, false, false],
                2 => [false, false, true, false],
                3 => [false, false, false, true],
                _ => unreachable!()
            };
            
            for (pin, state) in self.pins.iter().zip(pattern.iter()) {
                pin.write(*state)?;
            }
            
            Ok(())
        }
    }

    /// Brushless DC Motor (BLDC) controller
    pub struct BLDCMotor {
        pwm: [PWM; 3],
        hall_sensors: [GPIO::Pin; 3],
        config: MotorConfig,
        current_speed: f32,
        electrical_angle: f32
    }

    impl BLDCMotor {
        pub fn new(
            pwm: [PWM; 3],
            hall_sensors: [GPIO::Pin; 3],
            config: MotorConfig
        ) -> Result<Self, MotorError> {
            if config.motor_type != MotorType::BrushlessDC {
                return Err(MotorError::InvalidMotorType);
            }
            
            Ok(BLDCMotor {
                pwm,
                hall_sensors,
                config,
                current_speed: 0.0,
                electrical_angle: 0.0
            })
        }
        
        /// Set motor speed (0.0 to 1.0)
        pub fn set_speed(&mut self, speed: f32) -> Result<(), MotorError> {
            self.current_speed = speed.clamp(0.0, 1.0);
            Ok(())
        }
        
        /// Update motor commutation (call in main loop)
        pub fn update(&mut self) -> Result<(), MotorError> {
            // Read hall sensors
            let hall_state = (
                self.hall_sensors[0].read()?,
                self.hall_sensors[1].read()?,
                self.hall_sensors[2].read()?
            );
            
            // Determine electrical angle (60 degree sectors)
            let sector = match hall_state {
                (false, false, true) => 0,
                (false, true, true) => 1,
                (false, true, false) => 2,
                (true, true, false) => 3,
                (true, false, false) => 4,
                (true, false, true) => 5,
                _ => return Err(MotorError::HallSensorFault)
            };
            
            self.electrical_angle = sector as f32 * 60.0;
            
            // Apply PWM based on sector
            let (a, b, c) = self.calculate_pwm(sector);
            
            self.pwm[0].set_duty(a)?;
            self.pwm[1].set_duty(b)?;
            self.pwm[2].set_duty(c)?;
            
            Ok(())
        }
        
        /// Calculate PWM values for sector
        fn calculate_pwm(&self, sector: u8) -> (f32, f32, f32) {
            let angle = self.electrical_angle.to_radians();
            let (a, b, c) = match sector {
                0 => (self.current_speed, 0.0, -self.current_speed),
                1 => (self.current_speed, -self.current_speed, 0.0),
                2 => (0.0, -self.current_speed, self.current_speed),
                3 => (-self.current_speed, 0.0, self.current_speed),
                4 => (-self.current_speed, self.current_speed, 0.0),
                5 => (0.0, self.current_speed, -self.current_speed),
                _ => (0.0, 0.0, 0.0)
            };
            
            (a.max(0.0), b.max(0.0), c.max(0.0))
        }
    }

    /// Servo Motor controller
    pub struct ServoMotor {
        pwm: PWM,
        config: MotorConfig,
        min_pulse: u16,
        max_pulse: u16
    }

    impl ServoMotor {
        pub fn new(
            pwm: PWM,
            config: MotorConfig
        ) -> Result<Self, MotorError> {
            if config.motor_type != MotorType::Servo {
                return Err(MotorError::InvalidMotorType);
            }
            
            Ok(ServoMotor {
                pwm,
                config,
                min_pulse: 1000,  // 1ms
                max_pulse: 2000   // 2ms
            })
        }
        
        /// Set servo angle (0.0 to 1.0)
        pub fn set_angle(&mut self, angle: f32) -> Result<(), MotorError> {
            let angle = angle.clamp(0.0, 1.0);
            let pulse = self.min_pulse + ((self.max_pulse - self.min_pulse) as f32 * angle) as u16;
            
            let duty = pulse as f32 / (1_000_000.0 / self.config.pwm_frequency as f32);
            self.pwm.set_duty(duty)
        }
        
        /// Calibrate servo pulse widths
        pub fn calibrate(&mut self, min_pulse: u16, max_pulse: u16) {
            self.min_pulse = min_pulse;
            self.max_pulse = max_pulse;
        }
    }
}